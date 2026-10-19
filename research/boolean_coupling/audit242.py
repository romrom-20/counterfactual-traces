"""Separate saved-array audit and frozen prospective scoring for experiment242."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from .audit229 import checked, sha, softmax
from .core225b import write_json
from .primitive242 import ARMS

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def head(w, state):
    x = state.astype(float)
    return (x / np.sqrt(np.mean(x * x) + w["final_eps"]) * w["final_norm"]) @ w["final_head"].T


def geometry(pred, target):
    den, mag = np.linalg.norm(target), np.linalg.norm(pred)
    return float(np.linalg.norm(pred - target) / max(den, 1e-30)), float(np.vdot(pred, target) / max(mag * den, 1e-30))


def decision(rows):
    strata = {f"{s}_{v}": sum(r["passes"] for r in rows if r["schema"] == s and r["reverse"] == v)
        for s in ("generic", "descriptive") for v in (0, 1)}
    n = sum(r["passes"] for r in rows)
    return dict(passes=n, total=32, strata=strata, meets_rule=n >= 24 and all(v >= 6 for v in strata.values()))


def main(run, phase):
    dest = run / ("pre-measurement-audit.json" if phase == "pre" else "audit.json")
    assert not dest.exists(), "Never overwrite an audit."
    design, manifest, status = (read(run / p) for p in ("design.json", "forecast-manifest.json", "forecast-status.json"))
    assert design["experiment"] == 242 and design["primary"] == "100" and tuple(design["arms"]) == ARMS
    assert manifest["design_sha256"] == sha(run / "design.json")
    assert manifest["status_sha256"] == sha(run / "forecast-status.json")
    assert status["status"] == "complete" and status["ordinary_true"] == manifest["ordinary_true"] == 0
    assert manifest["cases"] == status["cases"] and len(status["cases"]) == 32
    assert manifest["source_hashes"] == design["source_hashes"]
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    assert sha(run / "weights.npz") == design["weights_sha256"]
    cases = read(run / "sources/cases242.json")
    catalog = set(read(run / "sources/historical-input-hashes-242.json")["hashes"])
    seen, groups = set(), Counter()
    for case in cases:
        a, b, start = case["false_ids"], case["true_ids"], case["position"]
        assert len(a) == len(b) and [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y] == [start]
        assert a[start] == 895 and b[start] == 830 and len(a) > start + 1
        hashes = [hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() for ids in (a, b)]
        assert hashes == case["input_hashes"] and not set(hashes) & (seen | catalog)
        seen.update(hashes)
        groups[(case["group"]["schema"], case["group"]["reverse"])] += 1
    assert len(cases) == 32 and len(seen) == 64 and set(groups.values()) == {8}
    assert [x["key"] for x in cases] == [x["key"] for x in status["cases"]]
    counts = status["counts"]
    assert counts["ordinary_false"] == counts["zero_edit_controls"] == 32 and counts["forecast_arms"] == 256
    assert counts.get("completed_forecasts", 0) + counts.get("unstable_forecasts", 0) == 256
    if phase == "pre":
        assert not list((run / "observations").glob("*")) and not (run / "status.json").exists()
    else:
        sealed, measured, proof = (read(run / p) for p in ("forecast-seal.json", "status.json", "measurement-proof.json"))
        assert sealed["manifest_sha256"] == sha(run / "forecast-manifest.json")
        assert sealed["design_sha256"] == sha(run / "design.json")
        assert sealed["pre_audit_sha256"] == sha(run / "pre-measurement-audit.json")
        assert measured["status"] == "complete" and measured["ordinary_true"] == 32 and len(measured["cases"]) == 32
        assert measured["forecast_seal_sha256"] == proof["seal_sha256"] == sha(run / "forecast-seal.json")
        assert measured["proof_sha256"] == sha(run / "measurement-proof.json")
        assert measured["replication"] == sealed["replication"] == design["replication"]
        assert proof["mode"] == ("local_replication" if design["replication"] else "private_remote")
        assert datetime.fromisoformat(proof["accepted_utc"]) > datetime.fromisoformat(sealed["created_utc"])
    w = dict(np.load(run / "weights.npz"))
    checks, rows, quals, completions = [], [], Counter(), Counter()
    for case, ent in zip(cases, status["cases"], strict=True):
        key = case["key"]
        path = run / "forecasts" / (key + ".npz")
        meta = read(path.with_suffix(".json"))
        assert sha(path) == meta["arrays_sha256"] == ent["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == ent["record_sha256"]
        assert meta["key"] == key and meta["group"] == case["group"] and meta["start"] == case["position"]
        assert meta["ordinary_true"] == 0 and len(meta["false_prefix_hashes"]) == 28
        assert set(meta["arms"]) == set(ARMS) and len(meta["anchor_types"]) == 112
        for stat in meta["primitive_statistics"]:
            assert stat["arm"] in (*ARMS, "zero_control")
            if stat["mode"] == "tilt":
                assert stat["minimum"] >= 0 and stat["negative_entries"] == 0 and stat["public_mass_error"] <= 1e-5
        for c in meta["checks"] + meta["reference_checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"], (key, c)
        quals.update(meta["qualification_counts"])
        assert len(meta["qualification_witnesses"]) == meta["qualification_counts"]["distinct_directions"]
        for witness in meta["qualification_witnesses"]:
            assert np.isfinite(witness["scale"]) and witness["scale"] >= 1
            assert witness["label"] in meta["anchor_types"] and witness["mode"] in ("tilt", "taylor")
            assert witness["qualified_direction"] == "actual direction divided by max(1,infinity norm)"
            if not witness["zero_direction"]:
                assert len(witness["first_plus_sha256"]) == len(witness["first_minus_sha256"]) == 64
        with np.load(path) as d:
            f = d["false_trajectory"]
            assert f.shape == (28, len(case["false_ids"]) - case["position"], 1536) and np.isfinite(f).all()
            checked(checks, "zero_edit_trajectory", d["zero_trajectory"], f, 0.)
            checked(checks, "zero_edit_scores", d["zero_scores"], d["false_scores"], 0.)
            checked(checks, "False_finite_head", head(w, f[-1, -1]), d["false_scores"], .005)
            for label, kind in meta["anchor_types"].items():
                x, y = (d["anchor_" + label + "_" + n].astype(float) for n in ("x", "y"))
                assert np.isfinite(x).all() and np.isfinite(y).all()
                if kind == "norm":
                    value = x / np.sqrt(np.mean(x*x, -1, keepdims=True) + d["anchor_" + label + "_eps"]) * d["anchor_" + label + "_weight"]
                elif kind == "silu":
                    value = x / (1 + np.exp(-x))
                else:
                    assert kind == "softmax"
                    value = softmax(x)
                    assert np.min(y) >= 0
                checked(checks, "independent_False_" + kind, value, y, 1e-4)
            if phase == "final":
                p = run / "observations" / (key + ".npz")
                m = read(p.with_suffix(".json"))
                observation_ent = measured["cases"][len(rows) // len(ARMS)]
                assert observation_ent["key"] == key and sha(p) == m["arrays_sha256"] == observation_ent["arrays_sha256"]
                assert sha(p.with_suffix(".json")) == observation_ent["record_sha256"]
                assert m["forecast_record_sha256"] == ent["record_sha256"] and m["seal_sha256"] == sha(run / "forecast-seal.json")
                assert datetime.fromisoformat(m["native_started_utc"]) > datetime.fromisoformat(proof["accepted_utc"])
                with np.load(p) as o:
                    true, true_scores = o["true_trajectory"], o["true_scores"]
                assert true.shape == f.shape and np.isfinite(true).all() and np.isfinite(true_scores).all()
                checked(checks, "True_finite_head", head(w, true[-1, -1]), true_scores, .005)
                target = true[-1, 1:].astype(float) - f[-1, 1:].astype(float)
                native_margin = float(true_scores[1] - true_scores[0])
            for arm in ARMS:
                record, trajectory = meta["arms"][arm], d[arm + "_trajectory"]
                assert trajectory.shape[1:] == f.shape[1:] and len(trajectory) == record["completed_blocks"]
                complete = record["status"] == "complete"
                completions[record["status"]] += 1
                if complete:
                    assert trajectory.shape == f.shape and np.isfinite(trajectory).all()
                    scores = d[arm + "_scores"]
                    assert np.isfinite(scores).all()
                    checked(checks, "forecast_finite_head", head(w, trajectory[-1, -1]), scores, .005)
                else:
                    assert record["status"] == "unstable" and "failure" in record
                    assert arm + "_scores" not in d and arm + "_failed_values" in d
                if phase == "final":
                    row = dict(key=key, **case["group"], arm=arm, status=record["status"], passes=False)
                    if complete:
                        pred = trajectory[-1, 1:].astype(float) - f[-1, 1:].astype(float)
                        passes = []
                        for scope, pick in (("all", slice(None)), ("verdict", -1)):
                            rel, cosine = geometry(pred[pick], target[pick])
                            row.update({scope + "_relative": rel, scope + "_cosine": cosine})
                            passes.append(np.linalg.norm(target[pick]) >= .001 and rel <= .05 and cosine >= .99)
                        margin = float(scores[1] - scores[0])
                        row.update(native_margin=native_margin, forecast_margin=margin, margin_error=abs(margin-native_margin))
                        row["passes"] = bool(all(passes) and row["margin_error"] <= .10)
                    rows.append(row)
        print("saved-array audit", phase, key[:8], flush=True)
    assert dict(quals) == status["qualifications"]
    assert completions["complete"] == counts.get("completed_forecasts", 0) and completions["unstable"] == counts.get("unstable_forecasts", 0)
    result = dict(status="passed", phase=phase, ordinary_true=0 if phase == "pre" else 32,
        replication=design["replication"], checks=checks, counts=counts, qualifications=dict(quals),
        design_sha256=sha(run / "design.json"), manifest_sha256=sha(run / "forecast-manifest.json"))
    if phase == "final":
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (run / "outcomes.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        decisions = {arm: decision([r for r in rows if r["arm"] == arm]) for arm in ARMS}
        result.update(primary=decisions["100"], decisions=decisions, outcomes_sha256=sha(run / "outcomes.csv"),
            interpretation="Uniform computational sufficiency test; no learned circuit or full answer to Question B.")
    write_json(dest, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "qualifications")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.phase)
    except Exception as exc:
        write_json(args.run / ("failure-audit-" + args.phase + ".json"), dict(error=repr(exc)))
        raise
