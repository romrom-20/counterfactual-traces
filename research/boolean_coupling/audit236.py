"""Independent finite-MLP, update-equation, FD-witness and finite-readout audit for236."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .audit235 import finite, head

MODES = ("00", "10", "01", "11", "R")
ARMS = tuple(f"{incoming}_{mode}" for incoming in ("forecast", "oracle") for mode in MODES)+("finite_oracle",)


def geometry(pred, actual):
    den = np.linalg.norm(actual)
    return dict(relative=float(np.linalg.norm(pred-actual)/den),
        cosine=float(np.vdot(pred, actual)/(np.linalg.norm(pred)*den)), native_norm=float(den))


def main(run):
    assert not (run/"audit.json").exists()
    root = Path(__file__).resolve().parents[2]
    design, status = (json.loads((run/n).read_text()) for n in ("design.json", "status.json"))
    assert status["status"] == "complete" and status["counts"] == design["expected"]
    for name, h in design["source_hashes"].items():
        assert sha(root/name) == sha(run/"sources"/Path(name).name) == h
    previous, old = Path(design["input235"]), Path(design["input228"])
    assert sha(previous/"audit.json") == design["audit235_sha256"]
    assert sha(previous/"weights.npz") == design["weights235_sha256"]
    for name, h in design["input_hashes"].items():
        assert sha(old/name) == h
    weights = {k:v.astype(float) for k, v in dict(np.load(previous/"weights.npz")).items()}
    checks, rows, decisions, layer_rows = [], [], [], []
    fd_count, finite_count = 0, 0
    for entry in status["cases"]:
        key = entry["key"]
        path = run/"cases"/(key+".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["arrays_sha256"] == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert all(np.isfinite(c["error"]) and c["error"] <= c["bound"] for c in meta["checks"])
        data = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
        for name, tangent in data.items():
            if name.endswith("_tangent"):
                prefix = name.removesuffix("_tangent")
                checked(checks, "independent_used_FD", tangent, (data[prefix+"_plus"]-data[prefix+"_minus"])/2e-5)
                fd_count += 1
        for incoming in ("forecast", "oracle"):
            for mode in MODES:
                arm = incoming+"_"+mode
                proposed = f["F_pred_mid"][19] if incoming == "forecast" else o["true_mids"][19].astype(float)
                checked(checks, "incoming_regime", proposed, data[arm+"_initial"], 0.)
                for i in range(19, 24):
                    source = f["false_mids"][i].astype(float)+f["source_mid"][i]
                    if i > 19:
                        label = f"{arm}_a{i}"
                        checked(checks, "attention_direction", proposed-f["false_boundaries"][i].astype(float), data[label+"_direction"], 1e-9)
                        proposed = source+data[label+"_tangent"]
                    label = f"{arm}_m{i}"
                    checked(checks, "mlp_input_update", proposed, data[label+"_input"], 1e-9)
                    actual = o["true_mids"][i].astype(float)-f["false_mids"][i].astype(float)
                    g = geometry(proposed-f["false_mids"][i].astype(float), actual)
                    layer_rows.append(dict(key=key, **meta["group"], arm=arm, layer=i+1, **g))
                    if i == 23:
                        checked(checks, "parent_install", proposed.astype(np.float32).astype(float), data[arm+"_parent"], 0.)
                        continue
                    use_finite = i >= 21 or mode == "R" or mode[i-19] == "1"
                    need_linear = i < 21 and (not use_finite or mode == "R")
                    linear, finite_out = None, None
                    if need_linear:
                        checked(checks, "mlp_direction", proposed-source, data[label+"_direction"], 1e-9)
                        linear = f["false_boundaries"][i+1].astype(float)+f["source_out"][i]+data[label+"_tangent"]
                        checked(checks, "linear_mlp_update", linear, data[label+"_linear"], 1e-9)
                    if use_finite:
                        parent = proposed.astype(np.float32).astype(float)
                        checked(checks, "finite_mlp_parent", parent, data[label+"_finite_parent"], 0.)
                        frame = {k.removeprefix(f"n{2*i+1}_"):v for k, v in weights.items() if k.startswith(f"n{2*i+1}_")}
                        finite_out = data[label+"_finite_output"]
                        checked(checks, "numpy_finite_mlp", finite(frame, {}, parent, "mlp"), finite_out)
                        finite_count += 1
                    if mode == "R" and i < 21:
                        correction = finite_out-linear
                        checked(checks, "local_correction", correction, data[label+"_correction"], 1e-9)
                        seed = int(hashlib.sha256((key+arm+str(i)+"236").encode()).hexdigest()[:16], 16)
                        random = np.random.default_rng(seed).standard_normal(correction.shape)
                        random *= np.linalg.norm(correction, axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
                        checked(checks, "seeded_row_control", random, data[label+"_random"], 1e-9)
                        proposed = linear+random
                    else:
                        proposed = finite_out if use_finite else linear
                    checked(checks, "mlp_output_update", proposed, data[label+"_output"], 1e-9)
        outcomes = {}
        for arm in ARMS:
            checked(checks, "finite_head", head(weights, data[arm+"_final"][-1]), data[arm+"_scores"], .005)
            row = dict(key=key, **meta["group"], arm=arm,
                vector_error=float(np.linalg.norm(data[arm+"_final"][-1]-data["native_final"][-1])),
                parent_error=float(np.linalg.norm(data[arm+"_parent"]-data["native_parent"])),
                margin_error=float(abs(np.diff(data[arm+"_scores"])[0]-np.diff(data["native_scores"])[0])))
            for scope in ("all", "verdict"):
                select = slice(None) if scope == "all" else -1
                pred = (data[arm+"_final"]-data["false_final"])[select]
                actual = (data["native_final"]-data["false_final"])[select]
                row.update({scope+"_"+k:v for k, v in geometry(pred, actual).items()})
            row["accuracy"] = bool(row["margin_error"] <= .10 and all(row[scope+"_native_norm"] >= .001
                and row[scope+"_relative"] <= .05 and row[scope+"_cosine"] >= .99 for scope in ("all", "verdict")))
            outcomes[arm] = row
            rows.append(row)
        for arm, final, scores in (("forecast_00", f["F_suffix"][-1], f["F_scores"]), ("finite_oracle", o["true_final"], o["true_scores"])):
            checked(checks, "final_replay_"+arm, data[arm+"_final"], final)
            checked(checks, "score_replay_"+arm, data[arm+"_scores"], scores, .005)
        decision = dict(key=key, **meta["group"])
        for incoming in ("forecast", "oracle"):
            joint = outcomes[incoming+"_11"]
            decision[incoming+"_accuracy"] = joint["accuracy"]
            decision[incoming+"_controlled"] = bool(joint["accuracy"] and all(joint[m] < outcomes[incoming+"_"+mode][m]
                for mode in ("00", "10", "01", "R") for m in ("vector_error", "margin_error")))
        interaction = data["forecast_11_final"]-data["forecast_10_final"]-data["forecast_01_final"]+data["forecast_00_final"]
        decision["factorial_vector_norm"] = float(np.linalg.norm(interaction))
        decision["factorial_margin"] = float(sum(sign*np.diff(data["forecast_"+mode+"_scores"])[0]
            for sign, mode in ((1,"11"),(-1,"10"),(-1,"01"),(1,"00"))))
        decisions.append(decision)
    assert fd_count == 1664 and finite_count == 1024
    for name, records in (("outcomes", rows), ("decisions", decisions), ("layerwise", layer_rows)):
        with (run/(name+".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    counts = {}
    for name in ("forecast_accuracy", "forecast_controlled", "oracle_accuracy", "oracle_controlled"):
        total = sum(r[name] for r in decisions)
        strata = {f"{s}_{rev}":sum(r[name] for r in decisions if r["schema"] == s and r["reverse"] == rev)
            for s in ("generic", "descriptive") for rev in (0, 1)}
        counts[name] = dict(passes=total, strata=strata, whole=total >= 24 and all(n >= 6 for n in strata.values()))
    summary = dict(status="passed", experiment=236, retrospective=True, checks=checks, counts=status["counts"],
        source_hashes=design["source_hashes"], decisions=counts, fd_count=fd_count, finite_mlp_reconstructions=finite_count,
        raw_accuracy={arm:sum(r["accuracy"] for r in rows if r["arm"] == arm) for arm in ARMS},
        medians={arm:{name:float(np.median([r[name] for r in rows if r["arm"] == arm])) for name in
            ("vector_error", "parent_error", "margin_error", "all_relative", "all_cosine", "verdict_relative", "verdict_cosine")} for arm in ARMS})
    (run/"audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k, v in summary.items() if k not in ("checks", "source_hashes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    target = parser.parse_args().run.resolve()
    try:
        main(target)
    except Exception as exc:
        (target/"failure-audit.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
