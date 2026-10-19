"""Independent NumPy graph/derivative/update audit and fixed final scoring for237."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import build, checked, derivative, sha
from .audit230 import corner
from .audit235 import finite, head
from .audit236 import geometry

POLICIES = ("1001", "1101", "1011", "1111")
MODES = ("legacy", *POLICIES, "public")
ARMS = tuple(f"{regime}_{mode}" for regime in ("forecast", "oracle") for mode in MODES)


def effects(a, b, c, d):
    return dict(QK=b-a, S=c-a, interaction=d-b-c+a)


def signed_rows(rows, key, group, regime, layer, values, target):
    for scope in ("all", "verdict"):
        pick = slice(None) if scope == "all" else -1
        t = target[pick]
        den = float(np.vdot(t, t))
        for name, value in values.items():
            v = value[pick]
            rows.append(dict(key=key, **group, regime=regime, layer=layer, scope=scope, component=name,
                target_norm=float(np.linalg.norm(t)), norm=float(np.linalg.norm(v)),
                signed_share=float(np.vdot(v, t)/den) if den > 1e-20 else 0.,
                remaining_relative=float(np.linalg.norm(t-v)/np.sqrt(den)) if den > 1e-20 else None))


def main(run):
    assert not (run/"audit.json").exists(), "Never overwrite evidence."
    root = Path(__file__).resolve().parents[2]
    design, status = (json.loads((run/n).read_text()) for n in ("design.json", "status.json"))
    assert status["status"] == "complete" and status["counts"] == design["expected"]
    previous, old = Path(design["input236"]), Path(design["input228"])
    assert sha(previous/"audit.json") == design["audit236_sha256"]
    for name, h in design["source_hashes"].items():
        assert sha(root/name) == sha(run/"sources"/Path(name).name) == h
    for name, h in design["input_hashes"].items():
        assert sha(old/name) == h
    assert sha(Path(design["weights235"])) == design["weights235_sha256"]
    assert sha(run/"weights.npz") == status["weights_sha256"]
    weights = {k:v.astype(float) for k, v in dict(np.load(run/"weights.npz")).items()}
    mw = {k:v.astype(float) for k, v in dict(np.load(design["weights235"])).items()}
    checks, outcomes, local_rows, pipeline_rows, decision_rows = [], [], [], [], []
    ncorner, nfinite, nlegacy, nprimitive = 0, 0, 0, 0
    for entry in status["cases"]:
        key = entry["key"]
        path = run/"cases"/(key+".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["arrays_sha256"] == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert sha(previous/"cases"/path.name) == meta["input236_arrays_sha256"]
        assert all(np.isfinite(c["error"]) and c["error"] <= c["bound"] for c in meta["checks"])
        data = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        f, o = (dict(np.load(old/phase/path.name)) for phase in ("forecasts", "observations"))
        p = {k:v.astype(float) for k, v in dict(np.load(previous/"cases"/path.name)).items()}
        start = meta["start"]
        frames, originals = {}, {}
        for i in range(20, 24):
            prefix = f"n{2*i}_"
            weights_i = {k.removeprefix(prefix):v for k, v in weights.items() if k.startswith(prefix)}
            frame = dict(x=data[f"a{i}_x"], receiver=data[f"a{i}_source"],
                mask=data["mask"], rope_cos=data["rope_cos"], rope_sin=data["rope_sin"])
            original = build(weights_i, frame, frame["x"])
            frames[i], originals[i] = (weights_i, frame), original
            for bits in POLICIES:
                checked(checks, "numpy_source_corner", corner(weights_i, frame, original, bits)[start:], data[f"a{i}_source_{bits}"], 1e-9)
                ncorner += 1
        for record in meta["derivatives"]:
            assert record["ad_error"] <= 1e-9 and record["fd_error"] <= 1e-4 and record["primal_error"] <= 1e-9
            prefix = f"w{record['layer_index']}_{record['witness_key']}"
            checked(checks, "primitive_FD", data[prefix+"_tangent"], (data[prefix+"_plus"]-data[prefix+"_minus"])/2e-5)
            nprimitive += 1
        for regime in ("forecast", "oracle"):
            for mode in MODES:
                arm = regime+"_"+mode
                initial = f["F_pred_mid"][19] if regime == "forecast" else o["true_mids"][19].astype(float)
                checked(checks, "incoming_regime", initial, data[arm+"_initial"], 0.)
                proposed = initial
                for i in range(19, 24):
                    if i > 19:
                        label = f"{arm}_a{i}"
                        checked(checks, "attention_input", proposed, data[label+"_input"], 1e-9)
                        weights_i, frame = frames[i]
                        if mode == "legacy":
                            direction = np.zeros_like(frame["x"])
                            direction[start:] = proposed-frame["x"][start:]
                            checked(checks, "legacy_direction", direction[start:], data[label+"_direction"], 1e-9)
                            tangent = derivative(weights_i, frame, frame["x"], direction, originals[i])[start:]
                            checked(checks, "numpy_legacy_derivative", tangent, data[label+"_tangent"], 1e-9)
                            checked(checks, "legacy_FD", tangent, (data[label+"_plus"]-data[label+"_minus"])/2e-5)
                            proposed = data[f"a{i}_source_mid"]+tangent
                            nlegacy += 1
                        elif mode == "public":
                            value = frame["receiver"].copy()
                            value[start:] = proposed.astype(np.float32).astype(float)
                            proposed = data[label+"_output"]
                            checked(checks, "numpy_public_attention", (value+build(weights_i, frame, value)["write"])[start:], proposed)
                        else:
                            frame = dict(frame, receiver=frame["receiver"].copy())
                            frame["receiver"][start:] = proposed
                            value = corner(weights_i, frame, originals[i], mode)[start:]
                            checked(checks, "numpy_receiving_corner", value, data[label+"_corner"], 1e-9)
                            proposed = data[f"a{i}_source_mid"]+value-data[f"a{i}_source_{mode}"]
                            ncorner += 1
                            if mode == "1001":
                                fixed = {mode:proposed}
                                for bits in POLICIES[1:]:
                                    name = f"{regime}_fixed{i}_{bits}"
                                    checked(checks, "fixed_incoming_state", data[label+"_input"], data[name+"_input"], 0.)
                                    value = corner(weights_i, frame, originals[i], bits)[start:]
                                    checked(checks, "numpy_fixed_corner", value, data[name+"_corner"], 1e-9)
                                    fixed[bits] = data[f"a{i}_source_mid"]+value-data[f"a{i}_source_{bits}"]
                                    checked(checks, "fixed_update", fixed[bits], data[name+"_output"], 1e-9)
                                    ncorner += 1
                                terms = effects(*(fixed[b] for b in POLICIES))
                                target = fixed["1111"]-fixed["1001"]
                                checked(checks, "fixed_factorial_closure", sum(terms.values()), target, 1e-9)
                                signed_rows(local_rows, key, meta["group"], regime, i+1, terms, target)
                        checked(checks, "attention_update", proposed, data[label+"_output"], 1e-9)
                        checked(checks, "mlp_input", proposed, data[f"{arm}_m{i}_input"], 1e-9)
                    label = f"{arm}_m{i}"
                    parent = proposed.astype(np.float32).astype(float)
                    checked(checks, "mlp_install", parent, data[label+"_parent"], 0.)
                    if i < 23:
                        prefix = f"n{2*i+1}_"
                        mlpweights = {k.removeprefix(prefix):v for k, v in mw.items() if k.startswith(prefix)}
                        checked(checks, "numpy_finite_mlp", finite(mlpweights, {}, parent, "mlp"), data[label+"_output"])
                        nfinite += 1
                    else:
                        checked(checks, "final_parent_install", parent, data[arm+"_parent"], 0.)
                    proposed = data[label+"_output"]
            terms = effects(*(data[regime+"_"+b+"_final"] for b in POLICIES))
            target = data[regime+"_1111_final"]-data[regime+"_1001_final"]
            checked(checks, "pipeline_factorial_closure", sum(terms.values()), target, 1e-9)
            signed_rows(pipeline_rows, key, meta["group"], regime, 0, terms, target)
            for suffix, bound in (("parent", 1e-4), ("final", 1e-4), ("scores", .005)):
                checked(checks, "legacy_236_replay", data[regime+"_legacy_"+suffix], p[regime+"_11_"+suffix], bound)
                checked(checks, "full_graph_public_bridge", data[regime+"_1111_"+suffix], data[regime+"_public_"+suffix], bound)
        checked(checks, "native_replay", data["oracle_public_final"], o["true_final"])
        checked(checks, "native_scores", data["oracle_public_scores"], o["true_scores"], .005)
        checked(checks, "false_final", data["false_final"], f["false_final"], 0.)
        metrics = {}
        for arm in ARMS:
            checked(checks, "finite_head", head(weights, data[arm+"_final"][-1]), data[arm+"_scores"], .005)
            row = dict(key=key, **meta["group"], arm=arm,
                vector_error=float(np.linalg.norm(data[arm+"_final"][-1]-data["native_final"][-1])),
                margin_error=float(abs(np.diff(data[arm+"_scores"])[0]-np.diff(data["native_scores"])[0])))
            for scope in ("all", "verdict"):
                pick = slice(None) if scope == "all" else -1
                pred = (data[arm+"_final"]-data["false_final"])[pick]
                actual = (data["native_final"]-data["false_final"])[pick]
                row.update({scope+"_"+k:v for k, v in geometry(pred, actual).items()})
            row["accuracy"] = bool(row["margin_error"] <= .10 and all(row[s+"_native_norm"] >= .001
                and row[s+"_relative"] <= .05 and row[s+"_cosine"] >= .99 for s in ("all", "verdict")))
            outcomes.append(row)
            metrics[arm] = row
        for regime in ("forecast", "oracle"):
            base = metrics[regime+"_1001"]
            eligible = base["vector_error"] >= .001 and base["margin_error"] >= .10
            for bits in POLICIES:
                m = metrics[regime+"_"+bits]
                decision_rows.append(dict(key=key, **meta["group"], regime=regime, policy=bits, eligible=eligible,
                    accuracy=m["accuracy"], halves=bool(eligible and all(m[n] <= .5*base[n] for n in ("vector_error", "margin_error")))))
    assert ncorner == 2304 and nlegacy == 256 and nfinite == 1536
    assert nprimitive == status["primitive_derivatives"]
    for name, records in (("outcomes", outcomes), ("local-effects", local_rows), ("pipeline-effects", pipeline_rows), ("decisions", decision_rows)):
        with (run/(name+".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    decisions = {}
    for regime in ("forecast", "oracle"):
        for bits in POLICIES:
            subset = [r for r in decision_rows if r["regime"] == regime and r["policy"] == bits]
            count = sum(r["accuracy"] for r in subset)
            strata = {f"{s}_{rev}":sum(r["accuracy"] for r in subset if r["schema"] == s and r["reverse"] == rev)
                for s in ("generic", "descriptive") for rev in (0, 1)}
            decisions[regime+"_"+bits] = dict(passes=count, strata=strata, whole=count >= 24 and all(n >= 6 for n in strata.values()),
                eligible=sum(r["eligible"] for r in subset), halves=sum(r["halves"] for r in subset))
    audit = dict(status="passed", experiment=237, retrospective=True, model_calls=0, checks=checks,
        counts=status["counts"], source_hashes=design["source_hashes"], decisions=decisions,
        primitive_derivatives=nprimitive, primitive_fd_endpoints=2*nprimitive, legacy_derivatives=nlegacy,
        finite_mlp_reconstructions=nfinite, numpy_corner_reconstructions=ncorner,
        raw_accuracy={arm:sum(r["accuracy"] for r in outcomes if r["arm"] == arm) for arm in ARMS},
        medians={arm:{name:float(np.median([r[name] for r in outcomes if r["arm"] == arm])) for name in
            ("vector_error", "margin_error", "all_relative", "all_cosine", "verdict_relative", "verdict_cosine")} for arm in ARMS})
    (run/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:v for k, v in audit.items() if k not in ("checks", "source_hashes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    target = parser.parse_args().run.resolve()
    try:
        main(target)
    except Exception as exc:
        if target.exists():
            (target/"failure-audit.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
