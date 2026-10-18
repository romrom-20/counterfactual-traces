"""NumPy-only reconstruction and scoring of experiment 226b; no model calls."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def margin(scores):
    return float(np.float64(scores[1])-np.float64(scores[0]))


def evaluate(out):
    root = Path(__file__).resolve().parents[2]
    design = json.loads((out / "design.json").read_text())
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "complete" and status["counts"] == design["expected"]
    assert all(sha(root/p) == h for p, h in design["sources"].items())
    assert all(sha(p) == h for p, h in design["inputs"].items())
    w = dict(np.load(out / "weights.npz"))
    old_root = Path(next(p for p in design["inputs"] if p.endswith("/design.json"))).parent
    original_seal = json.loads((old_root / "qualification-seal.json").read_text())
    assert all(sha(old_root/p) == h for p, h in original_seal["files"].items())
    errors = defaultdict(float)
    counts = defaultdict(int)

    def check(name, error, bound):
        error = float(error)
        assert np.isfinite(error) and error <= bound, (name, error, bound)
        errors[name] = max(errors[name], error)
        counts[name] += 1

    def scores(x):
        x = np.asarray(x, dtype=np.float64)
        return (x / np.sqrt(np.mean(x*x)+float(w["eps"])) * w["norm"].astype(float)) @ w["head"].astype(float).T

    rows, readouts = [], []
    for key in design["contexts"]:
        path = out / "cases" / (key + ".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["arrays_sha256"]
        for item in meta["checks"]:
            check("runner_" + item["name"], item["error"], item["bound"])
        q = dict(np.load(path))
        paths = {k: q["path_"+k].astype(float) for k in ("zero", "d", "b", "both")}
        groups = meta["group"]
        for name in paths:
            check("finite_head", np.max(abs(scores(paths[name][-1,-1])-q["scores_"+name])), .005)
        m = {k: margin(q["scores_"+k]) for k in paths}
        observed = m["both"]-m["b"]-m["d"]+m["zero"]
        final_mix = paths["both"][-1]-paths["b"][-1]-paths["d"][-1]+paths["zero"][-1]
        final_norm = float(np.linalg.norm(final_mix))
        check("final_add_cast", np.max(abs(q["readout_add_final"]-(paths["b"][-1]+paths["d"][-1]-paths["zero"][-1]).astype(np.float32))), 0.)
        check("finite_head", np.max(abs(scores(q["readout_add_final"][-1])-q["readout_add_scores"])), .005)
        r, d, b, both = (paths[k][-1,-1] for k in ("zero", "d", "b", "both"))
        hs = {k: margin(scores(x)) for k, x in (("r", r), ("d", d), ("b", b), ("both", both), ("add", b+d-r))}
        readout_mix = hs["add"]-hs["b"]-hs["d"]+hs["r"]
        residual_effect = hs["both"]-hs["add"]
        total = hs["both"]-hs["b"]-hs["d"]+hs["r"]
        check("head_split", abs(total-readout_mix-residual_effect), 1e-9)
        readouts.append(dict(key=key, **groups, observed=observed,
            promoted_head_total=total, readout_mixing=readout_mix,
            residual_interaction=residual_effect,
            readout_only_within_point1=bool(abs(residual_effect)<=.10),
            final_interaction_norm=final_norm))
        for j in range(5):
            for stage in ("attention", "mlp"):
                prefix = f"b{24+j}_{stage}_"
                if stage == "attention":
                    parents = {k: paths[k][j] for k in paths}
                    ys = {k: q["mid_"+k][j].astype(float) for k in paths}
                else:
                    parents = {k: q["mid_"+k][j].astype(float) for k in paths}
                    ys = {k: paths[k][j+1] for k in paths}
                local = q[prefix+"gadd"].astype(float)-ys["b"]-ys["d"]+ys["zero"]
                prop = ys["both"]-q[prefix+"gadd"].astype(float)
                total = ys["both"]-ys["b"]-ys["d"]+ys["zero"]
                for label, v in (("local", local), ("propagated", prop), ("total", total)):
                    check("reconstruct_"+label, np.max(abs(v-q[prefix+label])), 1e-9)
                check("signed_identity", np.max(abs(local+prop-total)), 1e-9)
                check("additive_cast", np.max(abs(q[prefix+"add_installed"]-(parents["b"]+parents["d"]-parents["zero"]).astype(np.float32))), 0.)
                check("incoming_interaction", np.max(abs(q[prefix+"incoming_interaction"]-(parents["both"]-parents["b"]-parents["d"]+parents["zero"]))), 1e-9)
                seed = int(hashlib.sha256((key+prefix+"226").encode()).hexdigest()[:16], 16)
                random = np.random.default_rng(seed).standard_normal(local.shape)
                random *= np.linalg.norm(local)/np.linalg.norm(random)
                check("random_seed", np.max(abs(q[prefix+"random"]-random)), 1e-9)
                check("random_norm", abs(np.linalg.norm(random)-np.linalg.norm(local)), 1e-9)
                result = dict(key=key, **groups, site=24+j, stage=stage,
                    observed=observed, final_interaction_norm=final_norm,
                    incoming_interaction_norm=float(np.linalg.norm(q[prefix+"incoming_interaction"])),
                    local_norm=float(np.linalg.norm(local)),
                    transported_norm=float(np.linalg.norm(prop)),
                    output_interaction_norm=float(np.linalg.norm(total)),
                    local_dot_transported=float(np.sum(local*prop)))
                for arm in ("remove", "random", "reinsert"):
                    proposed = ys["both"]-(random if arm == "random" else local)
                    if arm == "reinsert":
                        proposed += local
                    check("installed_"+arm, np.max(abs(q[prefix+arm+"_installed"]-proposed.astype(np.float32))), 0.)
                    trajectory = q[prefix+arm+"_trajectory"].astype(float)
                    sc = q[prefix+arm+"_scores"]
                    check("finite_head", np.max(abs(scores(trajectory[-1,-1])-sc)), .005)
                    final = trajectory[-1]-paths["b"][-1]-paths["d"][-1]+paths["zero"][-1]
                    value = margin(sc)-m["b"]-m["d"]+m["zero"]
                    result[arm+"_W"] = value
                    result[arm+"_final_interaction_norm"] = float(np.linalg.norm(final))
                    if arm == "reinsert":
                        check("reinsert_trajectory", np.max(abs(trajectory-paths["both"][j+1:])), 1e-4)
                        check("reinsert_head", np.max(abs(sc-q["scores_both"])), .005)
                criteria = design["thresholds"]
                result["eligible"] = abs(observed)>criteria["guard"]
                result["joint_reduction"] = bool(result["eligible"]
                    and abs(result["remove_W"])<=criteria["margin_ratio"]*abs(observed)
                    and result["remove_final_interaction_norm"]<=criteria["vector_ratio"]*final_norm
                    and abs(result["remove_W"])<abs(result["random_W"])
                    and result["remove_final_interaction_norm"]<result["random_final_interaction_norm"])
                rows.append(result)
    summary = dict(experiment="226b", retrospective=True, derivative_claim=False,
        counts=status["counts"], checks=dict(counts), maximum_errors=dict(errors),
        readout={}, sublayers=[])
    for schema in ("generic", "descriptive"):
        rr = [r for r in readouts if r["schema"]==schema]
        summary["readout"][schema] = dict(n=len(rr),
            within_point1=sum(r["readout_only_within_point1"] for r in rr),
            **{k:float(np.median([r[k] for r in rr])) for k in
               ("observed", "readout_mixing", "residual_interaction", "final_interaction_norm")})
        for site in range(24,29):
            for stage in ("attention", "mlp"):
                rr = [r for r in rows if r["schema"]==schema and r["site"]==site and r["stage"]==stage]
                order_passes = [sum(r["joint_reduction"] for r in rr if r["reverse"]==order) for order in (0,1)]
                summary["sublayers"].append(dict(schema=schema, site=site, stage=stage,
                    eligible=sum(r["eligible"] for r in rr), passes=sum(r["joint_reduction"] for r in rr),
                    order_passes=order_passes,
                    stable_descriptive=bool(schema=="descriptive" and sum(order_passes)>=12 and min(order_passes)>=6),
                    **{k:float(np.median([r[k] for r in rr])) for k in
                       ("local_norm", "transported_norm", "output_interaction_norm", "remove_W", "random_W")},
                    median_vector_ratio=float(np.median([r["remove_final_interaction_norm"]/r["final_interaction_norm"] for r in rr]))))
    (out / "audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    (out / "readout.json").write_text(json.dumps(readouts, indent=2)+"\n")
    with (out / "cases.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    evaluate(parser.parse_args().out.resolve())
