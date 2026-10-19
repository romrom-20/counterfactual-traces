"""243b reporting correction: absent unstable counter denotes zero, not failure."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .audit242b import decision, geometry, head
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("QQ", "FQ", "QF", "FF")


def read(path):
    return json.loads(path.read_text())


def rows_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(run, previous):
    assert not (run / "audit.json").exists()
    design, status, seal = (read(run / n) for n in ("design.json", "status.json", "intervention-seal.json"))
    assert design["experiment"] == 243 and design["retrospective"] and design["primary"] == "source_only"
    assert tuple(design["arms"]) == ARMS and status["status"] == "complete" and len(status["cases"]) == 32
    assert seal["status"] == "sealed" and seal["retrospective"] and not seal["native_arrays_opened_by_runner"]
    assert seal["ordinary_true_calls"] == status["ordinary_true_calls"] == 0
    assert seal["design_sha256"] == sha(run / "design.json") and seal["status_sha256"] == sha(run / "status.json")
    assert seal["cases"] == status["cases"] and seal["source_hashes"] == design["source_hashes"]
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    for tag, name in (("audit", "audit.json"), ("design", "design.json"), ("forecast_status", "forecast-status.json"),
        ("native_status", "status.json"), ("weights", "weights.npz")):
        assert sha(previous / name) == design["input_" + tag + "_sha256"]
    old_status, old_native = read(previous / "forecast-status.json"), read(previous / "status.json")
    assert read(previous / "audit.json")["status"] == "passed"
    cases = read(run / "sources/cases242.json")
    assert len(cases) == len(old_status["cases"]) == len(old_native["cases"]) == 32
    weights = dict(np.load(previous / "weights.npz"))
    checks, outcomes, accounting, recovery, qual = [], [], [], [], Counter()
    for case, ent, old_ent, native_ent in zip(cases, status["cases"], old_status["cases"], old_native["cases"], strict=True):
        key = case["key"]
        assert key == ent["key"] == old_ent["key"] == native_ent["key"]
        p = run / "cases" / (key + ".npz")
        meta = read(p.with_suffix(".json"))
        assert sha(p) == meta["arrays_sha256"] == ent["arrays_sha256"] and sha(p.with_suffix(".json")) == ent["record_sha256"]
        assert meta["key"] == key and meta["group"] == case["group"] and meta["start"] == case["position"]
        assert meta["ordinary_true_calls"] == 0 and set(meta["arms"]) == {"zero", *ARMS}
        old_path, native_path = (previous / n / p.name for n in ("forecasts", "observations"))
        assert sha(old_path) == meta["prior_forecast_sha256"] == old_ent["arrays_sha256"]
        assert sha(old_path.with_suffix(".json")) == meta["prior_record_sha256"] == old_ent["record_sha256"]
        assert sha(native_path) == native_ent["arrays_sha256"]
        assert sha(native_path.with_suffix(".json")) == native_ent["record_sha256"]
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        with np.load(p) as d, np.load(old_path) as old, np.load(native_path) as native:
            false, true = old["false_trajectory"].astype(float), native["true_trajectory"].astype(float)
            margin_true = float(np.diff(native["true_scores"].astype(float))[0])
            states, margins = {}, {}
            for arm in ("zero", *ARMS):
                record = meta["arms"][arm]
                for c in record["reference_checks"]:
                    assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
                qual.update(record["qualification_counts"])
                assert len(record["witnesses"]) == record["qualification_counts"]["distinct_directions"]
                for witness in record["witnesses"]:
                    assert np.isfinite(witness["scale"]) and witness["scale"] >= 1
                    assert witness["qualified_direction"] == "actual direction divided by max(1,infinity norm)"
                trajectory = d[arm + "_trajectory"]
                assert trajectory.shape[1:] == false.shape[1:] and len(trajectory) == record["completed_blocks"]
                row = dict(key=key, **case["group"], arm=arm, status=record["status"], passes=False)
                if record["status"] == "complete":
                    assert trajectory.shape == false.shape and np.isfinite(trajectory).all()
                    states[arm] = trajectory.astype(float)
                    scores = d[arm + "_scores"].astype(float)
                    checked(checks, "independent_finite_head", head(weights, trajectory[-1, -1]), scores, .005)
                    margins[arm] = float(scores[1] - scores[0])
                    if arm in ("zero", "QQ", "FF"):
                        stem = {"zero": "false", "QQ": "100", "FF": "101"}[arm]
                        checked(checks, "independent_old_trajectory_replay", trajectory, old[stem + "_trajectory"], 0.)
                        checked(checks, "independent_old_score_replay", scores, old[stem + "_scores"], 0.)
                    if arm != "zero":
                        valid = []
                        for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                            target = true[-1, pick] - false[-1, pick]
                            rel, cosine = geometry(trajectory[-1, pick].astype(float) - false[-1, pick], target)
                            row.update({scope + "_relative": rel, scope + "_cosine": cosine})
                            valid.append(np.linalg.norm(target) >= .001 and rel <= .05 and cosine >= .99)
                        row.update(margin_error=abs(margins[arm]-margin_true), native_margin=margin_true, margin=margins[arm])
                        row["passes"] = bool(all(valid) and row["margin_error"] <= .10)
                else:
                    assert record["status"] == "unstable" and arm in ("FQ", "QF")
                    assert arm + "_failed_values" in d and arm + "_scores" not in d
                if arm != "zero":
                    outcomes.append(row)
            for a, b in (("QQ", "QF"), ("FQ", "FF")):
                n = min(len(d[a+"_trajectory"]), len(d[b+"_trajectory"]))
                checked(checks, "independent_receiver_cannot_change_source", d[a+"_trajectory"][:n, 0], d[b+"_trajectory"][:n, 0], 0.)
            for a, b in (("QQ", "FQ"), ("QF", "FF")):
                if len(d[a+"_trajectory"]) and len(d[b+"_trajectory"]):
                    checked(checks, "independent_first_block_delay", d[a+"_trajectory"][0, 1:], d[b+"_trajectory"][0, 1:], 0.)
            B = states["FF"] - states["QQ"]
            targets = {"source_only": "FQ", "receiver_only": "QF"}
            terms = {name: states[arm]-states["QQ"] for name, arm in targets.items() if arm in states}
            if all(arm in states for arm in ARMS):
                terms["interaction"] = states["FF"]-states["FQ"]-states["QF"]+states["QQ"]
                checked(checks, "independent_full_trajectory_factorial_closure", sum(terms.values()), B, 1e-10)
            for name in (*targets, "interaction"):
                row = dict(key=key, **case["group"], account=name, passes=False)
                if name in terms:
                    term = terms[name]
                    dm = (margins[targets[name]]-margins["QQ"]) if name in targets else margins["FF"]-margins["FQ"]-margins["QF"]+margins["QQ"]
                    db = margins["FF"]-margins["QQ"]
                    row.update(margin_target=db, margin_term=dm, margin_error=abs(dm-db) if name in targets else abs(dm))
                    valid = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        b, v = B[-1, pick], term[-1, pick]
                        rel, cosine = geometry(v, b)
                        magnitude = float(np.linalg.norm(v)/max(np.linalg.norm(b), 1e-30))
                        row.update({scope+"_relative": rel, scope+"_cosine": cosine, scope+"_relative_norm": magnitude})
                        valid.append(np.linalg.norm(b) >= .001 and (rel <= .10 and cosine >= .99 if name in targets else magnitude <= .10))
                    row["passes"] = bool(all(valid) and row["margin_error"] <= .10)
                    for layer in range(28):
                        for scope, pick in (("all", slice(1, None)), ("verdict", -1), ("source", 0)):
                            b, v = B[layer, pick], term[layer, pick]
                            den = float(np.vdot(b, b))
                            rel, cosine = geometry(v, b)
                            accounting.append(dict(key=key, **case["group"], term=name, layer=layer+1, scope=scope,
                                target_norm=float(np.sqrt(den)), relative_error=rel, cosine=cosine,
                                relative_norm=float(np.linalg.norm(v)/max(np.sqrt(den), 1e-30)),
                                signed_share=float(np.vdot(v, b)/max(den, 1e-30))))
                recovery.append(row)
        print("independent regional audit", key[:8], flush=True)
    assert dict(qual) == status["qualifications"]
    assert status["counts"]["zero_edit_controls"] == 32 and status["counts"]["intervention_forwards"] == 128
    assert status["counts"]["complete_forwards"] + status["counts"].get("unstable_forwards", 0) == 160
    for name, rows in (("native-outcomes", outcomes), ("contrast-recovery", recovery), ("signed-effects", accounting)):
        rows_csv(run / (name + ".csv"), rows)
    decisions = {name: decision([r for r in recovery if r["account"] == name]) for name in ("source_only", "receiver_only", "interaction")}
    write_json(run / "audit.json", dict(status="passed", retrospective=True, ordinary_model_calls=0,
        checks=checks, scorer_sha256=sha(__file__), original_scorer_sha256=sha(ROOT / "research/boolean_coupling/audit243.py"), primary=decisions["source_only"], contrast_decisions=decisions,
        native_decisions={a: decision([r for r in outcomes if r["arm"] == a]) for a in ARMS},
        design_sha256=sha(run / "design.json"), seal_sha256=sha(run / "intervention-seal.json"),
        csv_hashes={name: sha(run/(name+".csv")) for name in ("native-outcomes", "contrast-recovery", "signed-effects")},
        interpretation="Regional operation-defined policy contrast, not a learned circuit or prospective prediction."))
    print(json.dumps(decisions, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input-242", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.input_242.resolve())
    except Exception as exc:
        write_json(args.run / "failure-audit.json", dict(error=repr(exc)))
        raise
