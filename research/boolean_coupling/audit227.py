"""Independent saved-array NumPy accounting and frozen scoring for 227."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def margin(x):
    return float(np.float64(x[1])-np.float64(x[0]))


def main(out):
    root = Path(__file__).resolve().parents[2]
    design = json.loads((out / "design.json").read_text())
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "complete" and status["counts"] == design["expected"]
    assert all(sha(root/p)==h for p,h in design["sources"].items())
    assert all(sha(p)==h for p,h in design["inputs"].items())
    weights = dict(np.load(out / "weights.npz"))
    down = weights["down"].astype(float)
    counts, maximum = defaultdict(int), defaultdict(float)
    raw_errors, rows = [], []

    def check(name, error, bound):
        error = float(error)
        assert np.isfinite(error) and error <= bound, (name, error, bound)
        counts[name] += 1
        maximum[name] = max(maximum[name], error)

    def head(x):
        x = x.astype(float)
        return (x/np.sqrt(np.mean(x*x)+float(weights["eps"])) * weights["norm"].astype(float)) @ weights["head"].astype(float).T

    for key in design["contexts"]:
        p = out / "cases" / (key+".npz")
        record = json.loads(p.with_suffix(".json").read_text())
        assert sha(p)==record["arrays_sha256"]
        for c in record["checks"]:
            check("runner_"+c["name"], c["error"], c["bound"])
        q = dict(np.load(p))
        def get(name, field):
            return q[name+"_"+field].astype(float)
        terms = dict(N=get("add", "product")-get("X", "product"),
            S=get("X", "product")-get("Y", "product"),
            P=get("Y", "product")-get("b", "product")-get("d", "product")+get("zero", "product"))
        terms["all"] = get("add", "product")-get("b", "product")-get("d", "product")+get("zero", "product")
        for name, v in terms.items():
            check("term_"+name, np.max(abs(q["R"+name]-v)), 1e-9)
        check("sum_terms", np.max(abs(terms["N"]+terms["S"]+terms["P"]-terms["all"])), 1e-9)
        check("norm_intervention", np.max(abs(q["X_norm"]-(get("b", "norm")+get("d", "norm")-get("zero", "norm")).astype(np.float32))), 0.)
        check("same_XY_norm", np.max(abs(q["X_norm"]-q["Y_norm"])), 0.)
        check("same_XY_up", np.max(abs(q["X_up"]-q["Y_up"])), 0.)
        check("phi_intervention", np.max(abs(q["Y_phi"]-(get("b", "phi")+get("d", "phi")-get("zero", "phi")).astype(np.float32))), 0.)
        check("mid_cast", np.max(abs(q["add_mid"]-(get("b", "mid")+get("d", "mid")-get("zero", "mid")).astype(np.float32))), 0.)
        for field in ("gate", "up"):
            err = np.max(abs(get("X", field)-get("b", field)-get("d", field)+get("zero", field)))
            check("additive_"+field+"_rounding", err, 1e-4)
        for name in ("zero", "b", "d", "both", "add", "X", "Y"):
            check("product_replay", np.max(abs(q[name+"_product"]-(q[name+"_phi"]*q[name+"_up"]))), 1e-4)
            if name != "Y":
                g = get(name, "gate")
                silu = g/(1+np.exp(-g))
                check("silu", np.max(abs(silu-get(name, "phi"))), 1e-4)
        signs = dict(add=1, b=-1, d=-1, zero=1)
        mixed_out = sum(sign*get(name,"out") for name,sign in signs.items())
        ideal = terms["all"] @ down.T
        input_round = sum(sign*get(name,"mid") for name,sign in signs.items())
        down_round = sum(sign*(get(name,"write")-get(name,"product")@down.T) for name,sign in signs.items())
        add_round = sum(sign*(get(name,"out")-get(name,"mid")-get(name,"write")) for name,sign in signs.items())
        raw = float(np.max(abs(mixed_out-ideal)))
        raw_errors.append(raw)
        check("rounding_ledger", np.max(abs(mixed_out-ideal-input_round-down_round-add_round)), 1e-9)
        for name in ("zero", "b", "d", "both"):
            check("original_head", np.max(abs(head(q["path_"+name][-1,-1])-q["scores_"+name])), .005)
        m = {name:margin(q["scores_"+name]) for name in ("zero", "b", "d", "both")}
        observed = m["both"]-m["b"]-m["d"]+m["zero"]
        final = {name:q["path_"+name][-1].astype(float) for name in ("zero", "b", "d", "both")}
        interaction = final["both"]-final["b"]-final["d"]+final["zero"]
        for name,term in terms.items():
            seed = int(hashlib.sha256((key+name+"227").encode()).hexdigest()[:16],16)
            random = np.random.default_rng(seed).standard_normal(term.shape)
            random *= np.linalg.norm(term)/np.linalg.norm(random)
            check("random_seed", np.max(abs(random-q["random_"+name])), 1e-9)
            check("random_norm", abs(np.linalg.norm(random)-np.linalg.norm(term)), 1e-9)
            result = dict(key=key, **record["group"], component=name,
                original_W=observed, original_vector_norm=float(np.linalg.norm(interaction)),
                product_norm=float(np.linalg.norm(term)),
                raw_write_error=raw, raw_write_pass=raw<=1e-4)
            for arm,vector in (("remove",term), ("random",random)):
                label = arm+"_"+name
                check("installed", np.max(abs(q[label+"_installed"]-(get("both","product")-vector).astype(np.float32))), 0.)
                trajectory = q[label+"_trajectory"].astype(float)
                sc = q[label+"_scores"]
                check("finite_head", np.max(abs(head(trajectory[-1,-1])-sc)), .005)
                ii = trajectory[-1]-final["b"]-final["d"]+final["zero"]
                result[arm+"_W"] = margin(sc)-m["b"]-m["d"]+m["zero"]
                result[arm+"_vector_norm"] = float(np.linalg.norm(ii))
            t = design["thresholds"]
            result["eligible"] = abs(observed)>t["guard"]
            result["pass"] = bool(result["eligible"]
                and abs(result["remove_W"])<=t["margin_ratio"]*abs(observed)
                and result["remove_vector_norm"]<=t["vector_ratio"]*result["original_vector_norm"]
                and abs(result["remove_W"])<abs(result["random_W"])
                and result["remove_vector_norm"]<result["random_vector_norm"])
            rows.append(result)
        for name in ("sham", "reinsert"):
            check("native_replay", np.max(abs(q[name+"_trajectory"]-q["path_both"][1:])), 1e-4)
            check("finite_head", np.max(abs(head(q[name+"_trajectory"][-1,-1])-q[name+"_scores"])), .005)
    summary = dict(experiment=227, retrospective=True, derivative_claim=False,
        counts=status["counts"], checks=dict(counts), maximum_errors=dict(maximum),
        raw_write_comparisons=dict(total=len(raw_errors), failures=sum(e>1e-4 for e in raw_errors), maximum=max(raw_errors)),
        components=[])
    for schema in ("generic", "descriptive"):
        for component in ("N", "S", "P", "all"):
            rr = [r for r in rows if r["schema"]==schema and r["component"]==component]
            order = [sum(r["pass"] for r in rr if r["reverse"]==i) for i in (0,1)]
            summary["components"].append(dict(schema=schema, component=component,
                eligible=sum(r["eligible"] for r in rr), passes=sum(order), order_passes=order,
                stable_descriptive=bool(schema=="descriptive" and sum(order)>=12 and min(order)>=6),
                **{k:float(np.median([r[k] for r in rr])) for k in
                   ("original_W", "remove_W", "random_W", "product_norm")},
                median_vector_ratio=float(np.median([r["remove_vector_norm"]/r["original_vector_norm"] for r in rr]))))
    with (out / "cases.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
