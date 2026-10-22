"""Stdlib267 source/record verification, including failures and frozen decisions."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/correction-field-feedback-267"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), (a, b)


def decision(row, relative, cosine):
    return all(float(row[s+"_norm"]) >= .001
               and float(row[s+"_relative"]) <= relative
               and float(row[s+"_cosine"]) >= cosine for s in ("all", "verdict"))


def main():
    manifest = read(EVIDENCE/"manifest.json")
    for group in ("files", "scientific_sources"):
        for name, value in manifest[group].items():
            assert "drafts" not in Path(name).parts and sha(ROOT/name) == value, name
    assert manifest["scientific_freeze_commit"] == "d2e3e408d1ebcd682e18289c27b8f5022813d5ff"
    notes = read(EVIDENCE/"release-notes.json")
    assert notes["retrospective"] and notes["fresh_examples"] == 0
    assert notes["planned_retained_forwards"] == 80 and notes["additional_unretained_forwards"] == 20
    assert not notes["general_question_B_solved"] and not notes["learned_circuit_identified"]
    assert not notes["private_articles_included"]
    failed, run = (EVIDENCE/name for name in ("field-state-267", "field-state-267-retry"))
    assert read(failed/"failure.json")["error"] == "AssertionError()"
    assert not (failed/"status.json").exists() and not list(failed.glob("cases/*.json"))
    design, status, audit = (read(run/n) for n in ("design.json", "status.json", "audit.json"))
    assert design["source_hashes"] == read(failed/"design.json")["source_hashes"] == manifest["scientific_sources"]
    assert design["source_commit"] == read(failed/"design.json")["source_commit"] == manifest["scientific_freeze_commit"]
    assert design["cases"] == read(failed/"design.json")["cases"]
    assert sha(run/"design.json") == audit["design_sha256"]
    assert sha(run/"status.json") == audit["status_sha256"]
    assert audit["source_sha256"] == sha(ROOT/"research/boolean_coupling/audit267.py")
    assert audit["status"] == "passed" and audit["retrospective"]
    assert len(design["cases"]) == 4 and status["counts"]["full_forwards_attempted"] == 80
    assert status["counts"]["finite_policy_points"] == 2240
    assert status["counts"]["extra_public_MLP_evaluations"] == 4816
    for name, value in (("normalization_AD", 4480), ("normalization_FD", 8960),
                        ("independent_polynomial_jets", 4480), ("exact_field_points", 2240),
                        ("public_norm_replays", 4480), ("public_MLP_node_seals", 4480),
                        ("exact_public_head_replays", 80)):
        assert audit["counts"][name] == value, name
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in audit["checks"])
    for name, value in audit["csv_sha256"].items():
        assert sha(run/name) == value
    inventory = read(EVIDENCE/"raw-array-inventory.json")
    assert inventory["omitted"] and len(inventory["arrays"]) == 8
    arrays = {e["path"]: e for e in inventory["arrays"]}
    assert len(arrays) == 8 and all(e["bytes"] > 0 and len(e["sha256"]) == 64 for e in arrays.values())
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        assert case["key"] == entry["key"]
        path = run/"cases"/(case["key"]+".json")
        meta = read(path)
        assert sha(path) == entry["metadata_sha256"] and meta["arrays_sha256"] == entry["arrays_sha256"]
        assert arrays[run.name+"/cases/"+case["key"]+".npz"]["sha256"] == entry["arrays_sha256"]
        assert all(s["status"] == "complete" for s in meta["statuses"].values())
        assert set(meta["statuses"]) == set(design["arms"])
        assert len(meta["records"]) == 560 and all(c["error"] <= c["bound"] for c in meta["checks"])
        record = {(r["arm"], r["layer"]): r for r in meta["records"]}
        assert len(record) == 560
        for (arm, layer), r in record.items():
            assert r["prefix_digest"] == record["F", layer]["prefix_digest"]
            source = "F" if arm in ("F", "Z") else "T"
            assert r["source_digest"] == record[source, layer]["source_digest"]
            assert math.isfinite(r["tail_ratio"]) and r["tail_ratio"] <= 100
            if arm == "Z":
                assert r["block_digest"] == record["F", layer]["block_digest"]
    assert status["raw_bytes"] == sum(e["bytes"] for e in status["cases"]) <= 260*1024**2
    for name, value in audit["vectors_sha256"].items():
        assert arrays[run.name+"/"+name]["sha256"] == value
    old = ROOT/"results/mlp-normalization-origin-261-264/normalization-branches-energy-263"
    for name in ("design.json", "status.json", "audit.json"):
        assert design["reference_hashes"][name] == sha(old/name)
    raw = rows(run/"raw-projections.csv")
    head = rows(run/"raw-head.csv")
    for data, field, bound, count in ((raw, "raw_projection_checks", .0001, 13440),
                                     (head, "raw_head_checks", .005, 80)):
        assert len(data) == count
        assert all(float(r["original_bound"]) == bound for r in data)
        assert all((r["passes"] == "True") == (float(r["error"]) <= bound) for r in data)
        assert audit[field] == dict(total=count, failed=sum(r["passes"] == "False" for r in data), original_bound=bound)
    factorial, native = (rows(run/name) for name in ("factorial.csv", "native.csv"))
    assert len(factorial) == 20 and len(native) == 68
    assert len({(r["key"], r["family"]) for r in factorial}) == 20
    assert len({(r["key"], r["arm"]) for r in native}) == 68
    for r in factorial:
        assert r["complete"] == "True"
        passes = decision(r, .25, .95) and float(r["margin_error"]) <= float(r["margin_bound"])
        assert (r["passes"] == "True") == passes
        near(r["margin_error"], abs(float(r["candidate_endpoint_margin"])-float(r["target_endpoint_margin"])))
        near(r["margin_bound"], .005+.25*abs(float(r["anchor_margin"])-float(r["target_endpoint_margin"])))
    for r in native:
        assert r["status"] == "complete"
        passes = decision(r, .05, .99) and float(r["margin_error"]) <= .1
        assert (r["passes"] == "True") == passes
    for s in ("11", "10", "01", "00", "r"):
        selected = [r for r in factorial if r["family"] == s]
        count = sum(r["passes"] == "True" for r in selected)
        assert len(selected) == 4 and audit["decisions"][s] == dict(total=4, passes=count, complete=4)
        assert audit["preserved"][s] == (count >= 3)
    calibration = {a: sum(r["passes"] == "True" for r in native if r["arm"] == a) for a in ("H", "A11")}
    assert audit["native_calibration"] == calibration
    assert audit["native_connection_qualified"] == all(v == 4 for v in calibration.values())
    assert audit["random_direction_control_qualified"]
    assert len(rows(run/"contrasts.csv")) == 24
    fig = ROOT/"docs/figures/fields"
    provenance = read(fig/"provenance.json")
    assert provenance["posthoc"] and provenance["all_cases"] and provenance["model_calls"] == 0
    assert provenance["source_sha256"] == sha(ROOT/"research/boolean_coupling/describe_fields.py")
    for name, value in provenance["input_sha256"].items():
        assert sha(ROOT/name) == value
    for name, value in provenance["figure_sha256"].items():
        assert sha(fig/name) == value
    print("Verified267: unchanged-source storage retry, all80 retained forwards, all factorial/native decisions, finite readout criteria, raw failures, source seals and figure provenance. Integrity is not independent model replication or a complete Question B answer.")


if __name__ == "__main__":
    main()
