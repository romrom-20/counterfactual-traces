"""Stdlib evidence verification for261–264; no model or activation replication."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/mlp-normalization-origin-261-264"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(folder, name):
    with (folder/name).open() as stream:
        return list(csv.DictReader(stream))


def near(value, expected):
    assert math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9), (value, expected)


def checks(audit, total):
    assert audit["status"] == "passed" and audit["retrospective"]
    assert len(audit["checks"]) == total
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in audit["checks"])


def outcomes(folder, audit, expected):
    data = rows(folder, "outcomes.csv")
    assert len(data) == 4*len(expected)
    for arm, count in expected.items():
        selected = [r for r in data if r["arm"] == arm]
        assert len(selected) == len({r["key"] for r in selected}) == 4
        for r in selected:
            success = r["status"] == "complete" and all(float(r[s+"_norm"]) >= .001 and
                float(r[s+"_relative"]) <= .05 and float(r[s+"_cosine"]) >= .99 for s in ("all", "verdict")) and float(r["margin_error"]) <= .1
            assert (r["passes"] == "True") == success
        assert sum(r["passes"] == "True" for r in selected) == count
    for name, digest in audit["csv_sha256"].items():
        assert sha(folder/name) == digest
    return data


def main():
    manifest = read(EVIDENCE/"manifest.json")
    for name in ("files", "scientific_sources"):
        for path, digest in manifest[name].items():
            assert "drafts" not in Path(path).parts and sha(ROOT/path) == digest, path
    inventory = read(EVIDENCE/"raw-array-inventory.json")
    assert inventory["omitted"] and len({r["path"] for r in inventory["arrays"]}) == len(inventory["arrays"])
    assert all(r["bytes"] > 0 and len(r["sha256"]) == 64 and r["path"].endswith(".npz") for r in inventory["arrays"])
    indexed = {r["path"]:r for r in inventory["arrays"]}
    notes = read(EVIDENCE/"release-notes.json")
    assert notes["fresh_examples"] == 0 and not notes["general_question_B_solved"] and not notes["private_files_included"]
    original = EVIDENCE/"mlp-input-origin-261"
    failure = read(original/"failure-bookkeeping.json")
    assert failure["status"] == "invalid_metadata_paths"
    for r in failure["restoration"]:
        path = ROOT/"results/receiver-mlp-curvature-258/observations"/(r["key"]+".json")
        assert sha(path) == r["restored_sha256"] == r["expected_sha256"]
        assert read(path)["arrays_sha256"] == r["array_unchanged_sha256"]
    for folder_name in ("mlp-input-origin-261b", "normalization-branches-energy-263"):
        folder = EVIDENCE/folder_name
        for entry in read(folder/"status.json")["cases"]:
            record = folder/"cases"/(entry["key"]+".json")
            assert sha(record) == entry["metadata_sha256"]
            assert read(record)["arrays_sha256"] == entry["arrays_sha256"] == indexed[folder_name+"/cases/"+entry["key"]+".npz"]["sha256"]
            assert all(c["error"] <= c["bound"] for c in read(record)["checks"])
    bounded = EVIDENCE/"mlp-input-bounded-261d"
    a261 = read(bounded/"audit.json")
    checks(a261, 20792)
    assert a261["source_sha256"] == sha(ROOT/"research/boolean_coupling/audit261d.py")
    for name, digest in a261["vectors_sha256"].items():
        assert indexed["mlp-input-bounded-261d/"+name]["sha256"] == digest
    assert sha(EVIDENCE/"mlp-input-origin-261b/design.json") == a261["design_sha256"]
    assert sha(EVIDENCE/"mlp-input-origin-261b/status.json") == a261["status_sha256"]
    for name, field, failed, total in (("raw-projection-checks.csv", "raw_projection", 27, 1568), ("raw-displacement-checks.csv", "raw_displacement", 8, 448)):
        data = rows(bounded, name)
        assert len(data) == total and sum(r["passes"] == "False" for r in data) == failed
        assert sha(bounded/name) == a261[field+"_csv_sha256"]
        assert a261[field+"_checks"] == dict(total=total, failed=failed, original_bound=.0001)
    outcomes(bounded, a261, dict(B=4, L=0, C=2, R=0))
    investigation = read(EVIDENCE/"mlp-input-replay-261c/investigation.json")
    assert investigation["comparisons"] == investigation["exact_replays"] == 1568 and investigation["CPU64_failures"] == 27
    assert investigation["maximum_full_shape_error"] == 0
    assert read(EVIDENCE/"mlp-input-origin-261b/failure-audit.json")
    assert read(EVIDENCE/"mlp-input-qualified-261c/failure-audit.json")
    factors = EVIDENCE/"gate-curvature-factors-262"
    a262 = read(factors/"audit.json")
    checks(a262, 8848)
    assert a262["source_sha256"] == sha(ROOT/"research/boolean_coupling/factors262.py")
    assert a262["protocol_sha256"] == sha(ROOT/"docs/experiments/262-gate-curvature-factors.md")
    assert a262["model_calls"] == 0 and a262["input_qualified_audit_sha256"] == sha(bounded/"audit.json")
    data = rows(factors, "factors.csv")
    assert len(data) == 4*28*2*2*3*3
    for arm in ("B", "T"):
        for scope in ("all", "selected"):
            for term in ("base", "direction", "energy"):
                group = [r for r in data if r["arm"] == arm and r["scope"] == scope and r["term"] == term and r["branch"] == "gate" and r["eligible"] == "True"]
                summary = a262["summary"][arm+"_"+scope+"_"+term]
                assert len(group) == summary["eligible"] == 112
                count = sum(r["passes"] == "True" for r in group)
                assert summary["passes"] == count and summary["meets_90"] == (count >= .9*len(group))
                near(summary["median_relative"], median(float(r["relative_error"]) for r in group))
                assert summary["robust_public_passes"] == sum(r["robust_public_pass"] == "True" for r in group)
        assert not a262["summary"][arm+"_all_base"]["meets_90"] and not a262["summary"][arm+"_selected_base"]["meets_90"]
    branches = EVIDENCE/"normalization-branches-energy-263"
    a263 = read(branches/"audit.json")
    checks(a263, 35480)
    assert a263["source_sha256"] == sha(ROOT/"research/boolean_coupling/audit263.py")
    assert a263["design_sha256"] == sha(branches/"design.json") and a263["status_sha256"] == sha(branches/"status.json")
    assert a263["primary"] == "E" and a263["rounding_control_passes"] == 4
    data = outcomes(branches, a263, dict(B=4, L=0, G=0, U=0, P=1, D=0, E=2, H=4))
    assert all(float(r["all_relative"]) <= .05 and float(r["verdict_relative"]) <= .05 for r in data if r["arm"] == "E")
    raw = rows(branches, "raw-projections.csv")
    assert len(raw) == 2464 and sum(r["passes"] == "False" for r in raw) == 41
    assert a263["raw_projection_checks"] == dict(total=2464, failed=41, original_bound=.0001)
    public = [c for c in a263["checks"] if c["name"].startswith("exact_full_shape_public_")]
    assert len(public) == 2464 and all(c["error"] == c["bound"] == 0 for c in public)
    assert all(r["passes"] == "True" for r in rows(branches, "rounding-control.csv"))
    interaction = EVIDENCE/"normalization-component-interaction-264b"
    a264 = read(interaction/"audit.json")
    checks(a264, 452)
    assert a264["source_sha256"] == sha(ROOT/"research/boolean_coupling/interaction264b.py")
    assert a264["protocol_sha256"] == sha(ROOT/"docs/experiments/264b-block-index-correction.md")
    assert a264["model_calls"] == a264["new_native_measurements"] == 0
    assert a264["input_audit_sha256"] == sha(branches/"audit.json")
    data = rows(interaction, "layerwise.csv")
    assert len(data) == 4*28*2*6 and {int(r["layer"]) for r in data} == set(range(1, 29))
    assert len(rows(interaction, "error-changes.csv")) == 4*28*2*2
    lookup = {(r["key"], r["layer"], r["scope"], r["contrast"]):r for r in data}
    assert len(lookup) == len(data)
    for (key, layer, scope, contrast), row in lookup.items():
        if contrast != "direction_alone":
            continue
        coupled = lookup[key, layer, scope, "interaction"]
        near(float(coupled["signed_target_share"]), 1-float(row["signed_target_share"]))
        near(float(coupled["relative_target_norm"])**2, 1+float(row["relative_target_norm"])**2-2*float(row["signed_target_share"]))
    margins = rows(interaction, "margins.csv")
    assert len(margins) == 4
    for r in margins:
        near(float(r["direction_after_pattern"]), float(r["direction_alone"])+float(r["interaction"]))
        near(float(r["direction_after_pattern"]), float(r["H_error"])-float(r["P_error"]))
        near(float(r["complete_after_energy"]), float(r["H_error"])-float(r["E_error"]))
    assert rows(EVIDENCE/"normalization-component-interaction-264", "margins.csv") == margins
    for name, digest in a264["csv_sha256"].items():
        assert sha(interaction/name) == digest
    figures = ROOT/"docs/figures/normalization"
    provenance = read(figures/"provenance.json")
    assert provenance["all_cases"] and provenance["posthoc"] and provenance["model_calls"] == 0
    assert provenance["source_sha256"] == sha(ROOT/"research/boolean_coupling/normalization_figures.py")
    for path, digest in provenance["input_sha256"].items():
        assert sha(ROOT/path) == digest
    for name, digest in provenance["figure_sha256"].items():
        assert sha(figures/name) == digest
    print("Verified261–264: source seals, all old contexts, retained raw failures, original failed hypotheses, corrected28-block coverage and figure provenance. Not model replication or a general Question B solution.")


if __name__ == "__main__":
    main()
