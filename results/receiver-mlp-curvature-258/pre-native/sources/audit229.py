"""Separate NumPy arithmetic, derivative, finite-head and outcome audit for 229."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PRIMITIVES = ("QK", "SM", "AVs", "AVd")
ARMS = ("native", "base", "repair_R", "repair_C", "repair_both", "random_R", "random_C",
        *("remove_"+k for k in PRIMITIVES), *("random_"+k for k in PRIMITIVES))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(x):
    e = np.exp(x-x.max(-1, keepdims=True))
    return e/e.sum(-1, keepdims=True)


def checked(checks, name, a, b, bound=1e-4):
    error = float(np.max(abs(a-b)))
    checks.append(dict(name=name, error=error, bound=bound))
    assert np.isfinite(error) and error <= bound, checks[-1]


def build(weights, data, state):
    """Independent NumPy reconstruction of the promoted attention graph."""
    width, groups = int(weights["width"]), int(weights["groups"])
    rms = np.sqrt(np.mean(state**2, axis=-1, keepdims=True)+weights["attn_eps"])
    z = state/rms*weights["attn_norm"]

    def heads(value):
        return value.reshape(value.shape[0], -1, width).transpose(1, 0, 2)

    def projection(name):
        return heads(z@weights[name+"_weight"].T+weights.get(name+"_bias", 0))

    cosine, sine = data["rope_cos"][0], data["rope_sin"][0]

    def rotate(value):
        a, b = np.split(value, 2, axis=-1)
        return value*cosine+np.concatenate((-b, a), axis=-1)*sine

    q, k = (rotate(projection(name)) for name in ("q", "k"))
    k, v = (np.repeat(value, groups, axis=0) for value in (k, projection("v")))
    logits = (q@k.transpose(0, 2, 1))*weights["scaling"]
    p = softmax(logits+data["mask"][0])
    weighted = p@v
    flat = weighted.transpose(1, 0, 2).reshape(state.shape)
    write = flat@weights["o_weight"].T+weights.get("o_bias", 0)
    return dict(z=z, rms=rms, q=q, k=k, v=v, logits=logits, p=p, weighted=weighted, write=write)


def derivative(weights, data, state, direction, original):
    """Explicit norm/QK/softmax/AV derivative, independent of torch AD."""
    width, groups = int(weights["width"]), int(weights["groups"])
    rms = original["rms"]
    dz = weights["attn_norm"]*(direction/rms-state*np.mean(state*direction, -1, keepdims=True)/rms**3)

    def project(name, rope=False):
        value = (dz@weights[name+"_weight"].T).reshape(state.shape[0], -1, width).transpose(1, 0, 2)
        if rope:
            a, b = np.split(value, 2, axis=-1)
            value = value*data["rope_cos"][0]+np.concatenate((-b, a), -1)*data["rope_sin"][0]
        return value

    dq = project("q", True)
    dk, dv = (np.repeat(project(name, name == "k"), groups, axis=0) for name in ("k", "v"))
    dl = (dq@original["k"].transpose(0, 2, 1)+original["q"]@dk.transpose(0, 2, 1))*weights["scaling"]
    dp = original["p"]*(dl-np.sum(original["p"]*dl, -1, keepdims=True))
    dy = dp@original["v"]+original["p"]@dv
    return direction+dy.transpose(1, 0, 2).reshape(state.shape)@weights["o_weight"].T


def head(weights, value):
    return (value/np.sqrt(np.mean(value**2)+weights["eps"])*weights["norm"])@weights["head"].T


def signed(component, error):
    return float(np.vdot(component[-1], error[-1])/np.vdot(error[-1], error[-1]))


def main(run):
    target = run/"audit.json"
    assert not target.exists(), "Never overwrite an audit."
    design = json.loads((run/"design.json").read_text())
    status = json.loads((run/"status.json").read_text())
    assert status["status"] == "complete" and status["counts"] == design["expected"]
    root = Path(__file__).resolve().parents[2]
    for path, digest in design["source_hashes"].items():
        assert sha(root/path) == digest
        assert sha(run/"sources"/Path(path).name) == digest
    assert sha(run/"weights.npz") == status["weights_sha256"]
    weights = {k:v.astype(float) for k, v in dict(np.load(run/"weights.npz")).items()}
    checks, rows, outcomes = [], [], []
    for entry in status["cases"]:
        path = run/"cases"/(entry["key"]+".npz")
        record = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == entry["arrays_sha256"] == record["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert all(np.isfinite(c["error"]) and c["error"] <= c["bound"] for c in record["checks"])
        data = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        start = record["start"]
        graphs = {name:build(weights, data, data[name]) for name in ("x", "source", "receiver", "both")}
        for k, name in (("zero", "x"), ("source", "source"), ("receiver", "receiver"), ("both", "both")):
            checked(checks, "numpy_graph", graphs[name]["write"][start:], data["smooth_"+k+"_write"], 1e-9)
        for kind in ("used", "sine", "cosine"):
            direction = np.zeros_like(data["x"])
            direction[start:] = data[kind+"_direction"]
            tangent = derivative(weights, data, data["x"], direction, graphs["x"])[start:]
            checked(checks, "numpy_derivative", tangent, data[kind+"_tangent"], 1e-9)
            checked(checks, "saved_central_FD", (data[kind+"_plus"]-data[kind+"_minus"])/2e-5, tangent)
        g0, gs, gd, gb = (graphs[k] for k in ("x", "source", "receiver", "both"))
        padd = softmax(gs["logits"]+gd["logits"]-g0["logits"]+data["mask"][0])
        sm = padd-gs["p"]-gd["p"]+g0["p"]
        contexts = dict(QK=(gb["p"]-padd)@gb["v"], SM=sm@gb["v"],
                        AVs=(gd["p"]-g0["p"])@(gs["v"]-g0["v"]),
                        AVd=(gs["p"]-g0["p"])@(gd["v"]-g0["v"]))
        for name, value in contexts.items():
            vec = value.transpose(1, 0, 2).reshape(data["x"].shape)@weights["o_weight"].T
            checked(checks, "numpy_primitive", vec[start:], data["smooth_vector_"+name], 1e-9)
            checked(checks, "actual_primitive_effect", data["actual_effect_"+name], vec[start:])
        e, r, c = data["E"], data["R"], data["C"]
        checked(checks, "RC", e, r+c, 1e-9)
        checked(checks, "primitive_plus_rounding", c,
                sum(data["smooth_vector_"+k] for k in PRIMITIVES)+data["smooth_rounding_remainder"], 1e-9)
        for name in ("R", "C", *PRIMITIVES):
            value = data[name] if name in ("R", "C") else data["actual_effect_"+name]
            seed = int(hashlib.sha256((entry["key"]+"229"+name).encode()).hexdigest()[:16], 16)
            random = np.random.default_rng(seed).standard_normal(value.shape)
            random *= np.linalg.norm(value)/np.linalg.norm(random)
            checked(checks, "random_vector", random, data["random_"+name], 1e-9)
        metrics = {}
        for arm in ARMS:
            checked(checks, "numpy_finite_head", head(weights, data[arm+"_final"][-1]), data[arm+"_scores"], .005)
            delta = data[arm+"_final"]-data["measured_final"]
            margin = np.diff(data[arm+"_scores"])[0]-np.diff(data["measured_scores"])[0]
            metrics[arm] = dict(vector_error=float(np.linalg.norm(delta[-1])), margin_error=float(abs(margin)),
                signed_margin_error=float(margin), all_receiving_error=float(np.linalg.norm(delta)))
            outcomes.append(dict(key=entry["key"], **record["group"], arm=arm, **metrics[arm]))
        for arm in ("native", "repair_both"):
            checked(checks, "native_recovery", data[arm+"_final"], data["measured_final"])
            checked(checks, "native_score_recovery", data[arm+"_scores"], data["measured_scores"], .005)
        norm = float(np.linalg.norm(e[-1]))
        row = dict(key=entry["key"], **record["group"], error_norm=norm,
            native_displacement_norm=float(np.linalg.norm((data["both_mid"]-data["zero_mid"])[-1])),
            eligible_local=norm >= .001,
            eligible_downstream=metrics["base"]["vector_error"] >= .001 and metrics["base"]["margin_error"] >= .10)
        for name, value in (("R", r), ("C", c), *((k, data["smooth_vector_"+k]) for k in PRIMITIVES)):
            row[name+"_norm"] = float(np.linalg.norm(value[-1]))
            row[name+"_all_receiving_norm"] = float(np.linalg.norm(value))
            row[name+"_signed_share"] = signed(value, e)
            row[name+"_remaining_ratio"] = float(np.linalg.norm((e-value)[-1])/norm)
            row[name+"_local_pass"] = bool(row["eligible_local"] and row[name+"_remaining_ratio"] <= .25)
            if name in ("R", "C"):
                m, random, base = metrics["repair_"+name], metrics["random_"+name], metrics["base"]
                row[name+"_downstream_pass"] = bool(row["eligible_downstream"] and all(
                    m[q] <= .5*base[q] and m[q] < random[q] for q in ("vector_error", "margin_error")))
        row["R_C_cosine"] = float(np.vdot(r[-1], c[-1])/(np.linalg.norm(r[-1])*np.linalg.norm(c[-1])))
        rows.append(row)
    for name, values in (("cases", rows), ("outcomes", outcomes)):
        with (run/(name+".csv")).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    medians = {k:float(np.median([r[k] for r in rows])) for k in rows[0]
               if k not in ("key", "assignment", "block", "reverse", "schema", "target")}
    decision_keys = [k for k in rows[0] if k.endswith("_pass")]
    strata = {}
    for schema in ("generic", "descriptive"):
        for order in (0, 1):
            subset = [r for r in rows if r["schema"] == schema and r["reverse"] == order]
            strata[f"{schema}_{order}"] = dict(n=len(subset), eligible_downstream=sum(r["eligible_downstream"] for r in subset),
                **{k:sum(r[k] for r in subset) for k in decision_keys})
    passes = {k:sum(r[k] for r in rows) for k in decision_keys}
    whole = {k:passes[k] >= 24 and all(s[k] >= 6 for s in strata.values()) for k in decision_keys}
    audit = dict(status="passed", experiment=229, retrospective=True, model_calls=0,
        design_sha256=sha(run/"design.json"), auditor_sha256=sha(__file__), checks=checks,
        case_count=len(rows), outcome_count=len(outcomes), counts=status["counts"], medians=medians,
        passes=passes, whole=whole, strata=strata,
        outcome_medians={arm:{k:float(np.median([r[k] for r in outcomes if r["arm"] == arm]))
                            for k in ("vector_error", "margin_error", "signed_margin_error", "all_receiving_error")} for arm in ARMS},
        runner_check_maxima={name:max(c["error"] for ent in status["cases"]
            for c in json.loads((run/"cases"/(ent["key"]+".json")).read_text())["checks"] if c["name"] == name)
            for name in {c["name"] for ent in status["cases"]
                         for c in json.loads((run/"cases"/(ent["key"]+".json")).read_text())["checks"]}})
    target.write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:audit[k] for k in ("status", "case_count", "passes", "whole", "medians", "outcome_medians")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    main(parser.parse_args().run.resolve())
