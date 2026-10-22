"""Stdlib record/source/decision verification for the two backbone diagnostics."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/backbone-feedback-270-271"
RUNS = {270: "backbone-curvature-feedback-270", 271: "amplitude-query-mediation-271"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def vector_pass(row, relative, cosine):
    return all(float(row[s + "_norm"]) >= .001 and float(row[s + "_relative"]) <= relative
               and float(row[s + "_cosine"]) >= cosine for s in ("all", "verdict"))


def main():
    manifest = read(EVIDENCE / "manifest.json")
    for group in ("files", "scientific_sources"):
        for name, value in manifest[group].items():
            assert "drafts" not in Path(name).parts and sha(ROOT / name) == value, name
    notes = read(EVIDENCE / "release-notes.json")
    assert notes["retrospective"] and notes["old_contexts"] == 4 and notes["fresh_examples"] == 0
    assert notes["complete_forwards"] == 76
    assert not any(notes[k] for k in ("general_question_B_solved", "learned_circuit_identified",
                                     "private_articles_included", "speedup_claimed"))
    inventory = read(EVIDENCE / "raw-array-inventory.json")
    arrays = {r["path"]: r for r in inventory["arrays"]}
    assert inventory["omitted"] and len(arrays) == len(inventory["arrays"]) == 16
    assert all(r["bytes"] > 0 and len(r["sha256"]) == 64 for r in arrays.values())
    for number, name in RUNS.items():
        run = EVIDENCE / name
        design, status, audit = (read(run / n) for n in ("design.json", "status.json", "audit.json"))
        assert design["experiment"] == number and design["retrospective"]
        assert design["source_commit"] == manifest["scientific_freezes"][str(number)]
        assert design["source_hashes"] == {p: manifest["scientific_sources"][p]
                                           for p in design["source_hashes"]}
        assert audit["status"] == "passed" and audit["native_calibration"]
        assert sha(run / "design.json") == audit["design_sha256"]
        assert sha(run / "status.json") == audit["status_sha256"]
        assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in audit["checks"])
        expected = 40 if number == 270 else 36
        assert status["counts"]["full_forwards_attempted"] == expected
        assert audit["counts"]["policy_points"] == expected * 28
        assert audit["counts"]["exact_public_head_replays"] == expected
        assert audit["counts"]["SiLU_AD"] == 560 and audit["counts"]["SiLU_FD"] == 1120
        assert len(status["cases"]) == len(design["cases"]) == 4
        for entry in status["cases"]:
            key = entry["key"]
            path = run / "cases" / (key + ".json")
            assert sha(path) == entry["metadata_sha256"]
            meta = read(path)
            assert meta["arrays_sha256"] == entry["arrays_sha256"]
            assert arrays[name + "/cases/" + key + ".npz"]["sha256"] == entry["arrays_sha256"]
            assert len(meta["records"]) == 28 * (10 if number == 270 else 9)
            assert all(s["status"] == "complete" for s in meta["statuses"].values())
            assert all(c["error"] <= c["bound"] for c in meta["checks"])
        for filename, value in audit["csv_sha256"].items():
            assert sha(run / filename) == value
        for filename, value in audit["vectors_sha256"].items():
            assert arrays[name + "/" + filename]["sha256"] == value
        for filename, field, bound in (("raw-projections.csv", "raw_projection_checks", .0001),
                                        ("raw-head.csv", "raw_head_checks", .005)):
            data = rows(run / filename)
            assert len(data) == audit[field]["total"]
            assert audit[field]["original_bound"] == bound
            for row in data:
                assert math.isfinite(float(row["error"])) and float(row["original_bound"]) == bound
                assert (row["passes"] == "True") == (float(row["error"]) <= bound)
            assert sum(r["passes"] == "False" for r in data) == audit[field]["failed"]
        native, restoration = rows(run / "native.csv"), rows(run / "restoration.csv")
        for row in native:
            passed = vector_pass(row, .05, .99) and float(row["margin_error"]) <= .1
            assert (row["passes"] == "True") == passed
        for row in restoration:
            passed = vector_pass(row, .25, .95) and float(row["margin_error"]) <= float(row["margin_bound"])
            assert (row["passes"] == "True") == passed
        for arm, count in audit["native_counts"].items():
            selected = [r for r in native if r["arm"] == arm]
            assert len(selected) == 4 and sum(r["passes"] == "True" for r in selected) == count
        for arm, count in audit["restoration_counts"].items():
            selected = [r for r in restoration if r["arm"] == arm]
            assert len(selected) == 4 and sum(r["passes"] == "True" for r in selected) == count
        if number == 270:
            assert not audit["primary_backbone_tangent_sufficient"]
            assert audit["native_counts"] == dict(L=0, H=4, B=4, J=2, G=2, X=2, R=2)
            assert audit["restoration_counts"] == dict(G=0, X=0)
            assert audit["raw_projection_checks"] == dict(total=6720, failed=169, original_bound=.0001)
            local = rows(run / "local-effects.csv")
            for row in local:
                norm, discrepancy = float(row["gate_norm"]), float(row["gate_smooth_bridge_norm"])
                assert (row["gate_smooth_qualified"] == "True") == (norm >= .001 and discrepancy <= .01 * norm)
        else:
            assert not audit["primary_joint_controller_sufficient"]
            assert audit["restoration_counts"] == dict(S=0, Q=0, C=0)
            assert audit["native_counts"] == dict(L=0, B=4, J=2, S=2, Q=2, C=2)
            assert audit["raw_projection_checks"] == dict(total=6384, failed=161, original_bound=.0001)
            assert audit["counts"]["clamped_query_maps"] == audit["counts"]["clamped_amplitude_points"] == 224
            controller = EVIDENCE / RUNS[270]
            for filename, value in design["controller_hashes"].items():
                assert sha(controller / filename) == value
    clamp = EVIDENCE / "controller-clamp-checks-271"
    summary = read(clamp / "summary.json")
    assert summary["posthoc"] and summary["new_model_calls"] == 0
    assert summary["source_sha256"] == sha(ROOT / "research/boolean_coupling/clamps271.py")
    assert summary["counts"] == dict(query_equalities=112, correction_node_equalities=224,
        correction_field_equalities=1344, source_equalities=112, prefix_equalities=112,
        state_increment_ledgers=112)
    assert len(summary["checks"]) == 112 and all(c["error"] == 0 for c in summary["checks"])
    assert sha(clamp / "increments.csv") == summary["csv_sha256"]
    assert len(rows(clamp / "increments.csv")) == 224
    run = EVIDENCE / RUNS[271]
    assert sha(run / "status.json") == summary["status_sha256"]
    assert sha(run / "audit.json") == summary["audit_sha256"]
    for key, witness in summary["inputs"].items():
        path = run / "cases" / (key + ".json")
        assert sha(path) == witness["metadata_sha256"]
        assert witness["arrays_sha256"] == arrays[RUNS[271] + "/cases/" + key + ".npz"]["sha256"]
        records = {(r["arm"], r["layer"]): r for r in read(path)["records"]}
        for layer in range(1, 29):
            b, c = records["B", layer], records["C", layer]
            assert b["qinstalled_digest"] == c["qinstalled_digest"]
            for tag in ("p", "d"):
                assert b["nodes"][tag] == c["nodes"][tag]
            for name in ("source_digest", "prefix_digest"):
                assert b[name] == c[name]
    log_provenance = read(EVIDENCE / "log-provenance.json")
    assert len(log_provenance["logs"]) == 4
    for entry in log_provenance["logs"]:
        path = EVIDENCE / RUNS[entry["experiment"]] / (entry["stage"] + ".log")
        assert sha(path) == entry["published_sha256"] and "/Users/" not in path.read_text()
        assert len(entry["raw_sha256"]) == 64 and entry["raw_bytes"] > 0
    figure = read(ROOT / "docs/figures/backbone/provenance.json")
    assert figure["posthoc"] and not figure["new_model_calls"]
    for name, value in figure["inputs"].items():
        assert sha(ROOT / name) == value
    print("270/271: source seals, all cases, retained raw failures and frozen decisions verified.")


if __name__ == "__main__":
    main()
