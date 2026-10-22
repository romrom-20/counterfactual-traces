"""Stdlib verification of265/266 records, failed hypotheses and figure inputs."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/local-mlp-mixing-265-266"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(folder, name):
    with (folder/name).open() as stream:
        return list(csv.DictReader(stream))


def near(value, expected):
    assert math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9), (value, expected)


def checked_audit(folder, total, raw_total, raw_failed):
    a = read(folder/"audit.json")
    assert a["status"] == "passed" and a["retrospective"] and len(a["checks"]) == total
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in a["checks"])
    assert sha(folder/"design.json") == a["design_sha256"]
    for name, digest in a["csv_sha256"].items():
        assert sha(folder/name) == digest
    raw = rows(folder, "raw-projections.csv")
    assert len(raw) == raw_total and sum(r["passes"] == "False" for r in raw) == raw_failed
    assert all((r["passes"] == "True") == (float(r["error"]) <= float(r["original_bound"])) for r in raw)
    assert a["raw_projection_checks"] == dict(total=raw_total, failed=raw_failed, original_bound=.0001)
    return a


def main():
    manifest = read(EVIDENCE/"manifest.json")
    for name in ("files", "scientific_sources"):
        for path, digest in manifest[name].items():
            assert "drafts" not in Path(path).parts and sha(ROOT/path) == digest, path
    notes = read(EVIDENCE/"release-notes.json")
    assert notes["fresh_examples"] == 0 and notes["full_forwards"] == 28
    assert not notes["general_question_B_solved"] and not notes["private_articles_included"]
    inventory = read(EVIDENCE/"raw-array-inventory.json")
    assert inventory["omitted"]
    arrays = {r["path"]:r for r in inventory["arrays"]}
    assert len(arrays) == len(inventory["arrays"]) and all(r["bytes"] > 0 and len(r["sha256"]) == 64 for r in arrays.values())
    local = EVIDENCE/"local-mixing-265"
    rollout = EVIDENCE/"local-mixing-continuation-265"
    finite = EVIDENCE/"finite-silu-interaction-266"
    a265 = checked_audit(local, 3824, 3584, 120)
    arun = checked_audit(rollout, 16952, 5712, 115)
    a266 = checked_audit(finite, 1120, 2688, 141)
    for folder, audit, source in ((local, a265, "local265.py"), (rollout, arun, "audit265.py"), (finite, a266, "finite266.py")):
        assert audit["source_sha256"] == sha(ROOT/"research/boolean_coupling"/source)
        design = read(folder/"design.json")
        assert len(design["cases"]) == len({c["key"] for c in design["cases"]}) == 4
        for name, digest in design["source_hashes"].items():
            assert digest == manifest["scientific_sources"][name]
        if folder != rollout:
            assert audit["model_calls"] == 0
            for r in audit["records"]:
                assert r["arrays_sha256"] == arrays[folder.name+"/"+r["key"]+".npz"]["sha256"]
    previous = ROOT/"results/mlp-normalization-origin-261-264/normalization-branches-energy-263"
    assert read(local/"design.json")["reference_audit_sha256"] == sha(previous/"audit.json")
    assert read(local/"design.json")["reference_design_sha256"] == sha(previous/"design.json")
    for folder in (local, rollout, finite):
        assert read(folder/"design.json")["reference_status_sha256"] == sha(previous/"status.json")
    assert read(rollout/"design.json")["local_audit_sha256"] == sha(local/"audit.json")
    assert read(finite/"design.json")["input_local_audit_sha256"] == sha(local/"audit.json")
    assert a265["counts"]["exact_old_full_shape_MLP_replays"] == 224
    data = rows(local, "components.csv")
    assert len(data) == 4*2*28*8*2 and len({(r["key"],r["background"],r["layer"],r["component"],r["scope"]) for r in data}) == len(data)
    for r in data:
        eligible = float(r["norm"]) >= .001
        assert (r["eligible"] == "True") == eligible
        assert (r["passes"] == "True") == (eligible and float(r["relative"]) <= .25 and float(r["cosine"]) >= .95)
    for background, eligible, passed in (("H", 88, 3), ("L", 89, 2)):
        group = [r for r in data if r["background"] == background and r["scope"] == "all" and r["component"] == "gate_up_cross" and r["eligible"] == "True"]
        assert len(group) == eligible and sum(r["passes"] == "True" for r in group) == passed
        assert a265["decisions"][background] == dict(eligible=eligible, passes=passed, primary_pass=False)
    status = read(rollout/"status.json")
    assert status["status"] == "complete" and status["counts"]["full_forwards_attempted"] == 28
    assert status["combined_raw_bytes"] <= 300*1024**2
    assert status["combined_raw_bytes"] == sum(r["bytes"] for r in a265["records"])+sum(r["bytes"] for r in status["cases"])
    assert arun["status_sha256"] == sha(rollout/"status.json") and not arun["primary_global_sufficiency"]
    for r in status["cases"]:
        path = rollout/"cases"/(r["key"]+".json")
        assert sha(path) == r["metadata_sha256"] and read(path)["arrays_sha256"] == r["arrays_sha256"]
        assert arrays[rollout.name+"/cases/"+r["key"]+".npz"]["sha256"] == r["arrays_sha256"]
        assert all(c["error"] <= c["bound"] for c in read(path)["checks"])
        assert all(s["status"] == "complete" for s in read(path)["statuses"].values())
    data = rows(rollout, "outcomes.csv")
    margin = {(r["key"],r["arm"]):r for r in rows(rollout, "margins.csv")}
    assert len(data) == len(margin) == 16
    for r in data:
        assert r["status"] == "complete"
        interaction = all(float(r[s+"_interaction_norm"]) >= .001 and float(r[s+"_interaction_relative"]) <= .25 and float(r[s+"_interaction_cosine"]) >= .95 for s in ("all", "verdict"))
        interaction = interaction and float(r["interaction_margin_error"]) <= float(r["interaction_margin_bound"])
        native = all(float(r[s+"_native_norm"]) >= .001 and float(r[s+"_native_relative"]) <= .05 and float(r[s+"_native_cosine"]) >= .99 for s in ("all", "verdict")) and float(r["native_margin_error"]) <= .1
        assert (r["interaction_pass"] == "True") == interaction
        assert (r["native_pass"] == "True") == native
        m = margin[r["key"],r["arm"]]
        near(float(r["interaction_margin_bound"]), .005+.25*abs(float(m["H_margin"])-float(m["additive_margin"])))
        near(float(r["interaction_margin_error"]), abs(float(m["arm_margin"])-float(m["additive_margin"])))
    for arm in ("H", "A", "K", "R"):
        group = [r for r in data if r["arm"] == arm]
        assert len(group) == 4 and all(r["native_pass"] == "True" and r["interaction_pass"] == "False" for r in group)
        assert arun["decisions"][arm] == dict(total=4, interaction_passes=0, native_passes=4, guard_failed=0)
    data = rows(finite, "comparison.csv")
    assert len(data) == len(rows(finite, "alignment.csv")) == 4*2*28*3*2
    assert len(rows(finite, "margins.csv")) == 96
    for r in data:
        eligible = float(r["norm"]) >= .001
        assert (r["eligible"] == "True") == eligible
        assert (r["passes"] == "True") == (eligible and float(r["relative"]) <= .25 and float(r["cosine"]) >= .95)
    assert not a266["primary_sparse_interaction_fidelity"]
    for background, eligible, fixed in (("H", 88, 86), ("L", 89, 89)):
        for mode, expected in (("adaptive", 0), ("fixed", fixed)):
            group = [r for r in data if r["background"] == background and r["mode"] == mode and r["scope"] == "all" and r["eligible"] == "True"]
            assert len(group) == eligible and sum(r["passes"] == "True" for r in group) == expected
            assert a266["decisions"][background+"_"+mode] == dict(eligible=eligible, passes=expected, meets_90=expected >= .9*eligible)
    figures = ROOT/"docs/figures/transmission"
    provenance = read(figures/"provenance.json")
    assert provenance["posthoc"] and provenance["all_cases"] and provenance["model_calls"] == 0
    assert provenance["source_sha256"] == sha(ROOT/"research/boolean_coupling/describe_transmission.py")
    for path, digest in provenance["input_sha256"].items():
        assert sha(ROOT/path) == digest
    for name, digest in provenance["figure_sha256"].items():
        assert sha(figures/name) == digest
    print("Verified265/266: all contexts, primitive controls, retained raw failures, failed sufficiency/fidelity hypotheses, finite-readout criteria and figure provenance. Integrity is not model replication or a general Question B solution.")


if __name__ == "__main__":
    main()
