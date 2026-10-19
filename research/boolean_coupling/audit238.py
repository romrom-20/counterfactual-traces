"""Separate NumPy checkpoint/derivative/remainder audit and frozen238 outcome scoring."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import build, checked, derivative, sha
from .audit235 import finite, head
from .audit236 import geometry
from .checkpoint238 import locate, mlp_derivative, read_head, weights
from .ledger235 import states

NODES = tuple(range(2, 39))
ARMS = ("base", "oracle4", "own4", "random4", "native")


def spec(n):
    return n//2, "attention" if n % 2 == 0 else "mlp"


def operation(w, context, full, start, kind):
    if kind == "attention":
        original = build(w, context, full)

        def apply(direction):
            d = np.zeros_like(full)
            d[start:] = direction
            return derivative(w, context, full, d, original)[start:]
    else:
        def apply(direction):
            return mlp_derivative(w, full, direction)
    return apply


def bootstrap(run, phase):
    assert not (run/("ledger-audit.json" if phase == "ledger" else "audit.json")).exists()
    root = Path(__file__).resolve().parents[2]
    design = json.loads((run/"design.json").read_text())
    for p, h in design["source_hashes"].items():
        assert sha(root/p) == sha(run/"sources"/Path(p).name) == h
    old = Path(design["input228"])
    for p, h in design["input_hashes"].items():
        assert sha(old/p) == h
    folder, provenance = locate(root)
    assert provenance == design["checkpoint"]
    return design, old, folder, json.loads((folder/"config.json").read_text())


def checked_case(run, phase, entry):
    path = run/phase/(entry["key"]+".npz")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert sha(path) == meta["arrays_sha256"] == entry["arrays_sha256"]
    assert sha(path.with_suffix(".json")) == entry["record_sha256"]
    assert all(np.isfinite(c["error"]) and c["error"] <= c["bound"] for c in meta["checks"])
    assert all(np.isfinite(w["error"]) and w["error"] <= w["bound"] for w in meta["fd_witnesses"])
    return meta, dict(np.load(path))


def audit_ledger(run):
    design, old, folder, config = bootstrap(run, "ledger")
    status = json.loads((run/"ledger-status.json").read_text())
    assert status["status"] == "complete" and status["selection_sha256"] == sha(run/"selection.json")
    checks, rows, nfd = [], [], 0
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for entry in status["cases"]:
            key = entry["key"]
            meta, data = checked_case(run, "ledger", entry)
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            start = meta["start"]
            context = {k:data[k].astype(float) for k in ("mask", "rope_cos", "rope_sin")}
            target = o["true_mids"][19].astype(float)-f["F_pred_mid"][19]
            checked(checks, "terminal_error", target, data["target_error"], 0.)
            adjoint, scores = target, []
            for n in reversed(NODES):
                i, kind = spec(n)
                w = weights(handle, config, i, kind)
                x = data[f"a{i}_false"].astype(float) if kind == "attention" else f["false_mids"][i].astype(float)
                apply = operation(w, context, x, start, kind)
                ni, no, pi, po = states(f, o, i, kind)
                e = ni-pi
                checked(checks, "error_direction", e, data[f"n{n}_direction"], 0.)
                tangent = apply(e)
                checked(checks, "numpy_error_derivative", tangent, data[f"n{n}_tangent"], 1e-9)
                checked(checks, "error_FD", tangent, data[f"n{n}_fd"])
                nfd += 1
                source_input = f["false_boundaries"][i].astype(float) if kind == "attention" else f["false_mids"][i].astype(float)+f["source_mid"][i]
                source_base = f["false_mids"][i].astype(float)+f["source_mid"][i] if kind == "attention" else f["false_boundaries"][i+1].astype(float)+f["source_out"][i]
                native_R = no-source_base-apply(ni-source_input)
                checked(checks, "numpy_native_input_remainder", native_R, data[f"n{n}_remainder"], 1e-9)
                checked(checks, "native_error_recurrence", no-po, tangent+data[f"n{n}_remainder"], 1e-9)
                checked(checks, "adjoint_chain", adjoint, data[f"n{n}_adjoint_out"], 0.)
                incoming = data[f"n{n}_adjoint_in"]
                d = data[f"n{n}_direction"]
                j = data[f"n{n}_tangent"]
                if not np.count_nonzero(d):
                    d, j = data[f"n{n}_dual_direction"], data[f"n{n}_dual_tangent"]
                    checked(checks, "numpy_dual_direction", apply(d), j, 1e-9)
                    checked(checks, "dual_FD", j, data[f"n{n}_dual_fd"])
                    nfd += 1
                checked(checks, "independent_reverse_bilinear", np.vdot(incoming, d), np.vdot(adjoint, j), 1e-9)
                score = float(np.vdot(adjoint, native_R)/np.vdot(target, target))
                scores.append(score)
                rows.append(dict(key=key, node=n, signed_share=score))
                adjoint = incoming
            checked(checks, "adjoint_complete_closure", sum(scores), 1., 1e-9)
            print("audited remainder ledger", key[:8], flush=True)
    assert nfd == status["counts"]["jvps"]
    rank = sorted(NODES, key=lambda n:(-float(np.median([abs(r["signed_share"]) for r in rows if r["node"] == n])), n))
    selection = json.loads((run/"selection.json").read_text())
    assert rank == [r["node"] for r in selection["ranking"]] and rank[:4] == selection["selected"]
    result = dict(status="passed", experiment=238, phase="ledger", retrospective=True, model_calls=0,
        selection_sha256=sha(run/"selection.json"), source_hashes=design["source_hashes"], counts=status["counts"], checks=checks, independent_derivative_FDs=nfd)
    (run/"ledger-audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Ledger audit passed", len(checks), flush=True)


def audit_interventions(run):
    design, old, folder, config = bootstrap(run, "intervene")
    assert json.loads((run/"ledger-audit.json").read_text())["status"] == "passed"
    status, selection = (json.loads((run/p).read_text()) for p in ("status.json", "selection.json"))
    assert status["status"] == "complete" and status["selection_sha256"] == sha(run/"selection.json")
    assert status["ledger_audit_sha256"] == sha(run/"ledger-audit.json")
    selected = selection["selected"]
    checks, raw, outcomes, decisions, nfd, finite_count = [], [], [], [], 0, 0
    h = read_head(folder)
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for entry in status["cases"]:
            key = entry["key"]
            meta, data = checked_case(run, "cases", entry)
            assert meta["selection_sha256"] == sha(run/"selection.json")
            assert meta["ledger_arrays_sha256"] == sha(run/"ledger"/(key+".npz"))
            ledger = dict(np.load(run/"ledger"/(key+".npz")))
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            context = {k:ledger[k].astype(float) for k in ("mask", "rope_cos", "rope_sin")}
            start, correction = meta["start"], np.zeros_like(ledger["target_error"])
            proposed = {arm:o["true_states"][0].astype(float) for arm in ("own4", "random4")}
            for n in NODES:
                i, kind = spec(n)
                w = weights(handle, config, i, kind)
                x = ledger[f"a{i}_false"].astype(float) if kind == "attention" else f["false_mids"][i].astype(float)
                apply = operation(w, context, x, start, kind)
                label = f"oracle_n{n}"
                checked(checks, "oracle_carry", correction, data[label+"_direction"], 1e-9)
                checked(checks, "oracle_numpy_JVP", apply(correction), data[label+"_tangent"], 1e-9)
                checked(checks, "oracle_FD", data[label+"_tangent"], data[label+"_fd"])
                nfd += 1
                correction = data[label+"_tangent"]+(ledger[f"n{n}_remainder"] if n in selected else 0)
                checked(checks, "oracle_update", correction, data[label+"_output"], 1e-9)
                source_input = f["false_boundaries"][i].astype(float) if kind == "attention" else f["false_mids"][i].astype(float)+f["source_mid"][i]
                base = f["false_mids"][i].astype(float)+f["source_mid"][i] if kind == "attention" else f["false_boundaries"][i+1].astype(float)+f["source_out"][i]
                for arm in ("own4", "random4"):
                    label = f"{arm}_n{n}"
                    checked(checks, "own_input", proposed[arm], data[label+"_input"], 1e-9)
                    direction = proposed[arm]-source_input
                    checked(checks, "own_direction", direction, data[label+"_direction"], 1e-9)
                    checked(checks, "own_numpy_JVP", apply(direction), data[label+"_tangent"], 1e-9)
                    checked(checks, "own_FD", data[label+"_tangent"], data[label+"_fd"])
                    nfd += 1
                    linear = base+data[label+"_tangent"]
                    checked(checks, "affine_update", linear, data[label+"_linear"], 1e-9)
                    value = linear
                    if n in selected:
                        parent = proposed[arm].astype(np.float32).astype(float)
                        checked(checks, "finite_input_install", parent, data[label+"_finite_input"], 0.)
                        actual = data[label+"_finite_output"]
                        if kind == "attention":
                            full = x.copy()
                            full[start-1] = f["source_rows"][i-1]
                            full[start:] = parent
                            checked(checks, "finite_attention", (full+build(w, context, full)["write"])[start:], actual)
                        else:
                            with np.errstate(over="ignore"):
                                single = finite({k:v.astype(np.float32) for k, v in w.items()}, {}, parent.astype(np.float32), "mlp")
                            checked(checks, "finite_mlp_float32", single, actual)
                            error = float(np.max(abs(finite(w, {}, parent, "mlp")-actual)))
                            raw.append(dict(case=key, arm=arm, node=n, error=error, bound=1e-4, passed=error <= 1e-4))
                        value = actual
                        finite_count += 1
                        if arm == "random4":
                            delta = actual-linear
                            seed = int(hashlib.sha256((key+str(n)+"238").encode()).hexdigest()[:16], 16)
                            random = np.random.default_rng(seed).standard_normal(delta.shape)
                            random *= np.linalg.norm(delta, axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
                            checked(checks, "seeded_row_matched_control", random, data[label+"_random"], 1e-9)
                            value = linear+random
                    checked(checks, "own_output", value, data[label+"_output"], 1e-9)
                    proposed[arm] = value
            checked(checks, "combined_correction", correction, data["oracle_correction"], 1e-9)
            sources = dict(base=f["F_pred_mid"][19], oracle4=f["F_pred_mid"][19]+correction,
                own4=proposed["own4"], random4=proposed["random4"], native=o["true_mids"][19])
            metrics = {}
            for arm in ARMS:
                checked(checks, "parent_install", sources[arm].astype(np.float32).astype(float), data[arm+"_parent"], 0.)
                checked(checks, "finite_head", head(h, data[arm+"_final"][-1]), data[arm+"_scores"], .005)
                row = dict(key=key, **meta["group"], arm=arm,
                    parent_error=float(np.linalg.norm(data[arm+"_parent"]-o["true_mids"][19].astype(float))),
                    vector_error=float(np.linalg.norm(data[arm+"_final"][-1]-o["true_final"].astype(float)[-1])),
                    margin_error=float(abs(np.diff(data[arm+"_scores"].astype(float))[0]-np.diff(o["true_scores"].astype(float))[0])))
                for scope in ("all", "verdict"):
                    pick = slice(None) if scope == "all" else -1
                    p = (data[arm+"_final"].astype(float)-f["false_final"].astype(float))[pick]
                    target = (o["true_final"].astype(float)-f["false_final"].astype(float))[pick]
                    row.update({scope+"_"+k:v for k, v in geometry(p, target).items()})
                row["accuracy"] = bool(row["margin_error"] <= .10 and all(row[s+"_native_norm"] >= .001 and
                    row[s+"_relative"] <= .05 and row[s+"_cosine"] >= .99 for s in ("all", "verdict")))
                metrics[arm] = row
                outcomes.append(row)
            checked(checks, "native_final_replay", data["native_final"], o["true_final"])
            checked(checks, "native_score_replay", data["native_scores"], o["true_scores"], .005)
            decision = dict(key=key, **meta["group"])
            for arm in ("oracle4", "own4"):
                decision[arm+"_halves"] = bool(all(metrics[arm][m] <= .5*metrics["base"][m] for m in ("parent_error", "vector_error", "margin_error")))
            decision["own4_controlled"] = bool(decision["own4_halves"] and all(metrics["own4"][m] < metrics["random4"][m]
                for m in ("parent_error", "vector_error", "margin_error")))
            decisions.append(decision)
            print("audited upstream intervention", key[:8], flush=True)
    assert nfd == status["counts"]["jvps"] and finite_count == 256
    for name, records in (("outcomes", outcomes), ("decisions", decisions), ("raw-float64-mlp-bridges", raw)):
        if records:
            with (run/(name+".csv")).open("w") as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    summary = dict(status="passed" if all(r["passed"] for r in raw) else "passed_declared_domain_with_raw_bridge_failures",
        experiment=238, retrospective=True, model_calls=0, source_hashes=design["source_hashes"], checks=checks,
        raw_float64_mlp_bridges=raw, counts=status["counts"], selected=selected, independent_derivative_FDs=nfd,
        finite_reconstructions=finite_count,
        raw_accuracy={arm:sum(r["accuracy"] for r in outcomes if r["arm"] == arm) for arm in ARMS},
        medians={arm:{n:float(np.median([r[n] for r in outcomes if r["arm"] == arm])) for n in
            ("parent_error", "vector_error", "margin_error", "all_relative", "verdict_relative")} for arm in ARMS},
        improvements={n:sum(r[n] for r in decisions) for n in ("oracle4_halves", "own4_halves", "own4_controlled")})
    summary["whole"] = {arm:dict(passes=summary["raw_accuracy"][arm],
        strata={f"{s}_{rev}":sum(r["accuracy"] for r in outcomes if r["arm"] == arm and r["schema"] == s and r["reverse"] == rev)
        for s in ("generic", "descriptive") for rev in (0, 1)}) for arm in ARMS}
    for value in summary["whole"].values():
        value["passes_rule"] = value["passes"] >= 24 and all(n >= 6 for n in value["strata"].values())
    (run/"audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k, v in summary.items() if k not in ("checks", "source_hashes", "raw_float64_mlp_bridges")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("ledger", "intervene"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        (audit_ledger if args.phase == "ledger" else audit_interventions)(args.run.resolve())
    except Exception as exc:
        (args.run/("failure-audit-"+args.phase+".json")).write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
