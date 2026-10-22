"""Stdlib verification of268's failed interaction hypothesis and sealed fresh269 test."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/amplitude-direction-268-269"
FRESH = ROOT/"results/signed-amplitude-forecast-269"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), (a, b)


def vector_pass(row, relative, cosine):
    return all(float(row[s+"_norm"]) >= .001 and float(row[s+"_relative"]) <= relative
               and float(row[s+"_cosine"]) >= cosine for s in ("all", "verdict"))


def checks(audit):
    assert audit["status"] == "passed"
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in audit["checks"])


def raw_checks(path, recorded, bound, total):
    data = rows(path)
    assert len(data) == total
    for row in data:
        assert float(row["original_bound"]) == bound and math.isfinite(float(row["error"]))
        assert (row["passes"] == "True") == (float(row["error"]) <= bound)
    assert recorded == dict(total=total, failed=sum(r["passes"] == "False" for r in data), original_bound=bound)


def main():
    manifest = read(EVIDENCE/"manifest.json")
    for group in ("files", "scientific_sources"):
        for name, value in manifest[group].items():
            assert "drafts" not in Path(name).parts and sha(ROOT/name) == value, name
    notes = read(EVIDENCE/"release-notes.json")
    assert notes["old_contexts"] == 4 and notes["fresh_literal_pairs"] == 32
    assert notes["complete_forwards"] == 400 and notes["geometry_model_calls"] == 0
    assert notes["claims_separate"] and not notes["primary_268_pass"]
    assert not notes["general_question_B_solved"] and not notes["learned_circuit_identified"]
    assert not notes["private_articles_included"]
    inventory = read(EVIDENCE/"raw-array-inventory.json")
    arrays = {r["path"]: r for r in inventory["arrays"]}
    assert inventory["omitted"] and len(arrays) == len(inventory["arrays"]) == 72
    assert all(r["bytes"] > 0 and len(r["sha256"]) == 64 for r in arrays.values())

    old = EVIDENCE/"amplitude-direction-268"
    design, status, audit = (read(old/name) for name in ("design.json", "status.json", "audit.json"))
    assert design["experiment"] == 268 and design["retrospective"]
    assert design["source_commit"].startswith("0421b73")
    assert sha(old/"design.json") == audit["design_sha256"] and sha(old/"status.json") == audit["status_sha256"]
    checks(audit)
    assert len(audit["checks"]) == 67616 and audit["counts"]["exact_field_points"] == 2240
    assert audit["counts"]["normalization_AD"] == 4480 and audit["counts"]["normalization_FD"] == 8960
    assert audit["counts"]["public_MLP_node_seals"] == 4928
    assert status["counts"]["full_forwards_attempted"] == 80
    for name, value in audit["csv_sha256"].items():
        assert sha(old/name) == value
    raw_checks(old/"raw-projections.csv", audit["raw_projection_checks"], .0001, 14784)
    raw_checks(old/"raw-head.csv", audit["raw_head_checks"], .005, 80)
    assert audit["raw_projection_checks"]["failed"] == 396
    factorial, native = rows(old/"factorial.csv"), rows(old/"native.csv")
    assert len(factorial) == 20 and len(native) == 68
    for row in factorial:
        passes = vector_pass(row, .25, .95) and float(row["margin_error"]) <= float(row["margin_bound"])
        assert (row["passes"] == "True") == passes
        near(row["margin_error"], abs(float(row["candidate_endpoint_margin"])-float(row["target_endpoint_margin"])))
        near(row["margin_bound"], .005+.25*abs(float(row["anchor_margin"])-float(row["target_endpoint_margin"])))
    for row in native:
        passes = vector_pass(row, .05, .99) and float(row["margin_error"]) <= .1
        assert (row["passes"] == "True") == passes
    for family, expected in (("11", 4), ("10", 2), ("01", 0), ("00", 0), ("r", 0)):
        selected = [r for r in factorial if r["family"] == family]
        assert len(selected) == 4 and sum(r["passes"] == "True" for r in selected) == expected
        assert audit["decisions"][family] == dict(total=4, passes=expected, complete=4)
    assert not audit["primary_amplitudes_sufficient"]
    assert audit["native_calibration"] == dict(H=4, A11=4)
    assert sum(r["passes"] == "True" for r in native if r["arm"] == "A10") == 4
    for entry in status["cases"]:
        path = old/"cases"/(entry["key"]+".json")
        assert sha(path) == entry["metadata_sha256"]
        meta = read(path)
        assert meta["arrays_sha256"] == entry["arrays_sha256"] == arrays[old.name+"/cases/"+entry["key"]+".npz"]["sha256"]
        assert len(meta["records"]) == 560 and all(s["status"] == "complete" for s in meta["statuses"].values())
        assert all(c["error"] <= c["bound"] for c in meta["checks"])
    for name, value in audit["vectors_sha256"].items():
        assert arrays[old.name+"/"+name]["sha256"] == value
    geom = EVIDENCE/"directional-geometry-268"
    geometry = read(geom/"summary.json")
    assert geometry["posthoc"] and geometry["model_calls"] == 0
    assert geometry["source_sha256"] == sha(ROOT/"research/boolean_coupling/geometry268.py")
    assert all(c["error"] <= c["bound"] for c in geometry["checks"])
    for name, value in geometry["input_hashes"].items():
        assert sha(old/name) == value
    points = rows(geom/"geometry.csv")
    assert len(points) == 1512
    for summary in geometry["summaries"]:
        selected = [r for r in points if r["arm"] == summary["arm"] and r["schema"] == summary["schema"]
                    and r["eligible"] == "True" and (summary["scope"] == "all" or r["verdict"] == "True")]
        assert len(selected) == summary["eligible"]
        energy = sum(float(r["angular_energy"]) for r in selected)
        for mode in ("line", "false_plane", "random_plane"):
            near(summary["weighted_"+mode], sum(float(r["angular_energy"])*float(r[mode+"_fraction"]) for r in selected)/energy)

    pre = FRESH/"pre-native"
    d, f, a, seal = (read(pre/name) for name in ("design.json", "forecast-status.json", "forecast-audit.json", "forecast-seal.json"))
    assert d["experiment"] == 269 and not d["retrospective"] and d["source_commit"].startswith("09f83f9")
    assert d["motivation_audit_sha256"] == sha(old/"audit.json") and not d["motivation_primary_passed"]
    assert seal["globally_pre_native"] and seal["ordinary_true"] == a["ordinary_true"] == 0
    assert seal["design_sha256"] == a["design_sha256"] == sha(pre/"design.json")
    assert seal["status_sha256"] == a["status_sha256"] == sha(pre/"forecast-status.json")
    assert seal["audit_sha256"] == sha(pre/"forecast-audit.json")
    assert f["status"] == "complete" and f["counts"]["full_forwards_attempted"] == 288
    assert f["raw_bytes"] <= 600*1024**2
    checks(a)
    assert a["counts"]["exact_field_points"] == 8064
    assert a["counts"]["normalization_AD"] == 16128 and a["counts"]["normalization_FD"] == 32256
    assert a["counts"]["public_MLP_node_seals"] == 15232 and a["counts"]["exact_public_head_replays"] == 288
    raw_checks(pre/"forecast-raw-projections.csv", a["raw_projection_checks"], .0001, 45696)
    raw_checks(pre/"forecast-raw-head.csv", a["raw_head_checks"], .005, 288)
    scores = {(r["key"], r["arm"]): r for r in rows(pre/"forecast-scores.csv")}
    assert len(scores) == 288
    cases = read(ROOT/"research/boolean_coupling/cases269.json")
    catalog = read(ROOT/"research/boolean_coupling/historical-input-hashes-269.json")
    assert d["cases"] == cases and len(cases) == 32
    hashes = []
    for case in cases:
        actual = [hashlib.sha256(json.dumps(case[b+"_ids"], separators=(",", ":")).encode()).hexdigest() for b in ("false", "true")]
        assert actual == case["input_hashes"]
        hashes.extend(actual)
        edits = [(i, x, y) for i, (x, y) in enumerate(zip(case["false_ids"], case["true_ids"], strict=True)) if x != y]
        assert edits == [(case["position"], 895, 830)]
    assert len(set(hashes)) == 64 and not set(hashes).intersection(catalog["hashes"])
    assert set(Counter((c["group"]["schema"], c["group"]["reverse"]) for c in cases).values()) == {8}
    for entry in f["cases"]:
        path = FRESH/"forecasts"/(entry["key"]+".json")
        assert sha(path) == entry["metadata_sha256"]
        meta = read(path)
        assert meta["arrays_sha256"] == entry["arrays_sha256"] == arrays["signed-amplitude-forecast-269/forecasts/"+entry["key"]+".npz"]["sha256"]
        assert len(meta["records"]) == 252 and all(s["status"] == "complete" for s in meta["statuses"].values())
        assert set(meta["statuses"]) == set(d["arms"]) and "T" not in meta["statuses"]
    started, ns, final = (read(FRESH/name) for name in ("native-started.json", "native-status.json", "audit.json"))
    assert started["ordinary_true_before"] == 0 and started["started"] > seal["created"]
    assert started["forecast_seal_sha256"] == ns["forecast_seal_sha256"] == sha(pre/"forecast-seal.json")
    assert ns["native_started_sha256"] == sha(FRESH/"native-started.json") and ns["ordinary_true"] == 32
    checks(final)
    assert final["seal_sha256"] == sha(pre/"forecast-seal.json")
    assert final["forecast_audit_sha256"] == sha(pre/"forecast-audit.json")
    assert final["native_status_sha256"] == sha(FRESH/"native-status.json")
    assert final["counts"]["native_head_replays"] == 32 and final["counts"]["source_prefix_points"] == 8064
    raw_checks(FRESH/"native-raw-head.csv", final["raw_native_head"], .005, 32)
    outcomes = rows(FRESH/"native-outcomes.csv")
    assert len(outcomes) == 224 and len({(r["key"], r["arm"]) for r in outcomes}) == 224
    for r in outcomes:
        assert r["status"] == "complete"
        passed = vector_pass(r, .05, .99) and float(r["margin_error"]) <= .1
        assert (r["passes"] == "True") == passed
        near(r["predicted_margin"], scores[r["key"], r["arm"]]["margin"])
        near(r["margin_error"], abs(float(r["predicted_margin"])-float(r["native_margin"])))
    for arm in final["decisions"]:
        selected = [r for r in outcomes if r["arm"] == arm]
        count = sum(r["passes"] == "True" for r in selected)
        assert len(selected) == 32 and final["decisions"][arm] == dict(total=32, passes=count, guard_failed=0)
    assert final["primary_pass"] == (final["decisions"]["A10"]["passes"] == 32) == notes["primary_269_pass"]
    assert final["reference_calibrated"] == (final["decisions"]["H"]["passes"] == 32)
    for entry in ns["cases"]:
        path = FRESH/"observations"/(entry["key"]+".json")
        assert sha(path) == entry["metadata_sha256"]
        assert read(path)["arrays_sha256"] == entry["arrays_sha256"] == arrays["signed-amplitude-forecast-269/observations/"+entry["key"]+".npz"]["sha256"]
    fig = ROOT/"docs/figures/amplitude"
    provenance = read(fig/"provenance.json")
    assert provenance["posthoc"] and provenance["all_cases"] and provenance["model_calls"] == 0
    assert provenance["source_sha256"] == sha(ROOT/"research/boolean_coupling/describe_amplitude.py")
    for name, value in provenance["input_sha256"].items():
        assert sha(ROOT/name) == value
    for name, value in provenance["figure_sha256"].items():
        assert sha(fig/name) == value
    print("Verified268/269: failed old interaction hypothesis retained; all32 fresh forecasts globally sealed before natives; every outcome, original raw failure, source seal and figure input checked. Integrity is not independent model replication or a complete Question B answer.")


if __name__ == "__main__":
    main()
