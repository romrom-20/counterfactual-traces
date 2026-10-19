"""Independent error recurrence, NumPy finite-operation reconstruction and outcome audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit229 import build, checked, sha

ARMS = ("base", "full", "top1", "top2", "top4", "gen4", "state4", "replay4",
        "random2_0", "random2_1", "random2_2", "random4_0", "random4_1", "random4_2")


def finite(weights, data, x, kind):
    if kind == "attention":
        return x+build(weights, data, x)["write"]
    z = x/np.sqrt(np.mean(x*x, -1, keepdims=True)+weights["eps"])*weights["norm"]
    g, u = (z@weights[name+"_weight"].T+weights.get(name+"_bias", 0) for name in ("gate", "up"))
    hidden = g/(1+np.exp(-g))*u
    return x+hidden@weights["down_weight"].T+weights.get("down_bias", 0)


def head(weights, value):
    x = value.astype(float)
    return (x/np.sqrt(np.mean(x*x)+weights["final_eps"])*weights["final_norm"])@weights["final_head"].T


def main(run):
    assert not (run/"audit.json").exists()
    root = Path(__file__).resolve().parents[2]
    design, status, selection = (json.loads((run/n).read_text()) for n in ("design.json", "status.json", "selection.json"))
    assert status["status"] == "complete" and len(status["cases"]) == 32
    for name, h in design["source_hashes"].items():
        assert sha(root/name) == sha(run/"sources"/Path(name).name) == h
    old = Path(design["input_root"])
    for name, h in design["inputs"].items():
        assert sha(old/name) == h
    assert sha(run/"weights.npz") == status["weights_sha256"]
    weights = {k:v.astype(float) for k, v in dict(np.load(run/"weights.npz")).items()}
    with (run/"ledger.csv").open() as f:
        ledger_rows = list(csv.DictReader(f))
    rank = sorted(range(47), key=lambda n:(-float(np.median([float(r["signed_share"]) for r in ledger_rows
        if int(r["node"]) == n and r["scope"] == "all"])), n))
    assert rank == [r["node"] for r in selection["ranking"]] and rank[:4] == selection["selected"]
    checks, rows, local_rows, case_rows = [], [], [], []
    for entry in status["cases"]:
        key = entry["key"]
        path = run/"cases"/(key+".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["arrays_sha256"] == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert meta["selected"] == selection["selected"] and meta["selection_sha256"] == sha(run/"selection.json")
        assert all(np.isfinite(c["error"]) and c["error"] <= c["bound"] for c in meta["checks"])
        d = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
        independently = []
        before = np.zeros_like(f["F_delta_in"][0])
        for i in range(24):
            mid_error = o["true_mids"][i].astype(float)-f["F_pred_mid"][i]
            independently.append(mid_error-before)
            if i < 23:
                after = o["true_states"][i].astype(float)-f["false_boundaries"][i+1].astype(float)-f["F_delta_out"][i]
                independently.append(after-mid_error)
                before = after
        independently.append(f["F_pred_mid"][-1]-f["F_target_mid"].astype(float))
        independently = np.stack(independently)
        saved = dict(np.load(run/"ledger"/(key+".npz")))
        checked(checks, "independent_write_error_recurrence", independently, saved["terms"], 1e-9)
        terminal = o["true_mids"][-1].astype(float)-f["F_target_mid"].astype(float)
        checked(checks, "complete_error_closure", independently.sum(0), terminal, 1e-9)
        for node in selection["selected"]:
            prefix = f"n{node}_"
            frame = {k.removeprefix(prefix):v for k, v in weights.items() if k.startswith(prefix)}
            kind = design["node_order"][node]["kind"]
            for side in ("native", "forecast"):
                x = d[prefix+side+"_input"]
                reconstructed = finite(frame, d, x, kind)[meta["start"]:]
                checked(checks, "numpy_"+kind+"_"+side, reconstructed, d[prefix+side+"_output"])
            total = saved["terms"][node]
            g, st, rp = (d[prefix+name] for name in ("generation", "state", "replay"))
            checked(checks, "independent_conditional_split", g+st+rp, total, 1e-9)
            for scope in ("all", "verdict"):
                select = slice(None) if scope == "all" else -1
                w, final_error = total[select], terminal[select]
                for name, vector in (("generation", g), ("state", st), ("replay", rp)):
                    v = vector[select]
                    den = float(np.vdot(w, w))
                    local_rows.append(dict(key=key, **meta["group"], node=node, kind=kind,
                        layer=design["node_order"][node]["layer"], scope=scope, component=name,
                        norm=float(np.linalg.norm(v)), write_norm=float(np.linalg.norm(w)),
                        signed_write_share=float(np.vdot(v, w)/den) if den else 0.,
                        signed_terminal_share=float(np.vdot(v, final_error)/np.vdot(final_error, final_error))))
        for total in (2, 4):
            for index in range(3):
                name = f"random{total}_{index}"
                seed = int(hashlib.sha256((key+name+"235").encode()).hexdigest()[:16], 16)
                random = np.random.default_rng(seed).standard_normal(terminal.shape)
                random *= np.linalg.norm(d[f"top{total}_correction"], axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
                checked(checks, "seeded_row_matched_control", random, d[name+"_correction"], 1e-9)
        outcomes = {}
        for arm in ARMS:
            installed = (d["parent_base"]+d[arm+"_correction"]).astype(np.float32).astype(float)
            checked(checks, "installed_parent", installed, d[arm+"_installed"], 0.)
            checked(checks, "numpy_finite_head", head(weights, d[arm+"_final"][-1]), d[arm+"_scores"], .005)
            metric = dict(parent_error=float(np.linalg.norm(installed-d["parent_native"])),
                vector_error=float(np.linalg.norm(d[arm+"_final"][-1]-d["native_final"][-1])),
                margin_error=float(abs(np.diff(d[arm+"_scores"])[0]-np.diff(d["native_scores"])[0])),
                signed_margin_error=float(np.diff(d[arm+"_scores"])[0]-np.diff(d["native_scores"])[0]))
            outcomes[arm] = metric
            rows.append(dict(key=key, **meta["group"], arm=arm, **metric))
        for arm, final, scores in (("base", d["reference_final"], d["reference_scores"]), ("full", d["native_final"], d["native_scores"])):
            checked(checks, "final_reconstruction_"+arm, d[arm+"_final"], final)
            checked(checks, "score_reconstruction_"+arm, d[arm+"_scores"], scores, .005)
        base = outcomes["base"]
        eligible = base["parent_error"] >= .001 and base["vector_error"] >= .001 and base["margin_error"] >= .10
        case_row = dict(key=key, **meta["group"], eligible=eligible)
        for arm in ("top1", "top2", "top4", "gen4", "state4"):
            controlled = arm in ("top2", "top4")
            passes = all(outcomes[arm][m] <= .5*base[m] for m in ("parent_error", "vector_error", "margin_error"))
            if controlled:
                passes = passes and all(outcomes[arm][m] < outcomes[f"random{arm[-1]}_{i}"][m]
                    for m in ("parent_error", "vector_error", "margin_error") for i in range(3))
            case_row[arm+"_pass"] = bool(eligible and passes)
        case_rows.append(case_row)
    for name, records in (("outcomes", rows), ("local-splits", local_rows), ("decisions", case_rows)):
        with (run/(name+".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    decisions = {}
    for name in ("top1", "top2", "top4", "gen4", "state4"):
        count = sum(r[name+"_pass"] for r in case_rows)
        strata = {f"{s}_{rev}":sum(r[name+"_pass"] for r in case_rows if r["schema"] == s and r["reverse"] == rev)
            for s in ("generic", "descriptive") for rev in (0, 1)}
        decisions[name] = dict(passes=count, eligible=sum(r["eligible"] for r in case_rows), strata=strata,
            whole=count >= 24 and all(n >= 6 for n in strata.values()))
    summary = dict(status="passed", experiment=235, retrospective=True, checks=checks, counts=status["counts"],
        source_hashes=design["source_hashes"], decisions=decisions, selection=selection,
        outcome_medians={arm:{m:float(np.median([r[m] for r in rows if r["arm"] == arm]))
            for m in ("parent_error", "vector_error", "margin_error", "signed_margin_error")} for arm in ARMS},
        local_medians={str(node):{component:{scope:{name:float(np.median([r[name] for r in local_rows if r["node"] == node
            and r["scope"] == scope and r["component"] == component])) for name in ("norm", "write_norm", "signed_write_share", "signed_terminal_share")}
            for scope in ("all", "verdict")} for component in ("generation", "state", "replay")} for node in selection["selected"]})
    (run/"audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k, v in summary.items() if k not in ("checks", "source_hashes")}, indent=2))


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
