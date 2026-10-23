"""Stdlib verification of272 prediction, measured work, latency and review records."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/causal-prefix-budget-272"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12), (a, b)


def main():
    manifest = read(DATA / "manifest.json")
    for group in ("files", "scientific_sources"):
        for name, value in manifest[group].items():
            assert "drafts" not in Path(name).parts and sha(ROOT / name) == value, name
    design, status, audit, notes = (read(DATA / n) for n in
                                  ("design.json", "status.json", "audit.json", "release-notes.json"))
    assert design["experiment"] == 272 and design["retrospective"]
    assert design["source_commit"] == manifest["scientific_freeze"]
    assert design["source_hashes"] == manifest["scientific_sources"]
    assert audit["design_sha256"] == sha(DATA / "design.json")
    assert audit["status_sha256"] == sha(DATA / "status.json")
    assert status["status"] == "complete" and audit["status"] == "passed"
    assert audit["primary_prediction_budget_passes"]
    assert audit["native_counts"] == dict(P=32, Z=0, S=0)
    assert not any(notes[k] for k in ("general_question_B_solved", "learned_circuit_identified",
                                     "novelty_priority_claimed", "private_articles_included", "fresh_confirmation"))
    assert notes["old_contexts"] == 32 and notes["retrospective"]
    expected = dict(profiled_full_forwards=64, profiled_suffix_forwards=96,
                    benchmark_full_forwards=96, benchmark_suffix_forwards=96)
    assert all(status[k] == notes[k] == v for k, v in expected.items())
    arrays = read(DATA / "raw-array-inventory.json")
    assert arrays["omitted"] and not arrays["KV_tensors_saved"] and len(arrays["arrays"]) == 32
    inventory = {r["path"]: r for r in arrays["arrays"]}
    native, raw, heads, budget, timing = (rows(DATA / name) for name in
        ("native.csv", "raw-coordinates.csv", "raw-heads.csv", "budget.csv", "timing.csv"))
    assert len(native) == 96 and len(raw) == 2688 and len(heads) == 352 and len(budget) == 32 and len(timing) == 96
    for name, value in audit["csv_sha256"].items():
        assert sha(DATA / name) == value
    for row in native:
        passed = all(float(row[s + "_norm"]) >= .001 and float(row[s + "_relative"]) <= .05
                     and float(row[s + "_cosine"]) >= .99 for s in ("all", "verdict"))
        passed = passed and float(row["margin_error"]) <= .1
        assert (row["passes"] == "True") == passed
    for row in raw:
        assert float(row["original_bound"]) == .0001
        assert (row["passes"] == "True") == (float(row["error"]) <= .0001)
    assert audit["raw_coordinates"] == dict(total=2688, failed=299, original_bound=.0001)
    assert sum(r["passes"] == "False" for r in raw) == 299
    assert {a: sum(r["passes"] == "False" for r in raw if r["arm"] == a) for a in ("P", "Z", "S")} == dict(P=108, Z=98, S=93)
    for row in heads:
        assert (row["passes"] == "True") == (float(row["error"]) <= float(row["original_bound"]))
        assert (row["bridge_passes"] == "True") == (float(row["error"]) <= .005)
    assert audit["raw_heads"] == dict(total=352, failed=0, bridge_failed=0)
    assert all(float(r["error"]) == 0 for r in heads)
    ratios, speeds = [], []
    for index, (case, entry) in enumerate(zip(design["cases"], status["cases"], strict=True)):
        key = case["key"]
        path = DATA / "cases" / (key + ".json")
        assert sha(path) == entry["metadata_sha256"]
        meta = read(path)
        assert meta["arrays_sha256"] == entry["arrays_sha256"] == inventory["cases/" + key + ".npz"]["sha256"]
        assert all(v == "complete" for v in meta["statuses"].values())
        n, k = len(case["false_ids"]), case["position"]
        assert len(case["true_ids"]) == n and n - k in (5, 6)
        assert [(i, a, b) for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"], strict=True)) if a != b] == [(k, 895, 830)]
        assert meta["records"]["F"]["prefix_states"] == meta["records"]["T"]["prefix_states"]
        assert len(meta["shared_prefix"]) == 28
        for arm in ("F", "T", "P", "Z"):
            for point, prefix in zip(meta["records"][arm]["attention"], meta["shared_prefix"], strict=True):
                assert point["prefix_key"] == prefix["key"] and point["prefix_value"] == prefix["value"]
        # Independent config formula was qualified in audit272. Check all saved
        # counts/shapes against that pinned architecture, not just a summary ratio.
        per_row = 28 * (2 * 1536**2 + 2 * 1536 * 256 + 3 * 1536 * 8960)
        head = 1536 * 151936
        for arm, profile in meta["records"].items():
            m = n if arm in ("F", "T") else n - k - (arm == "S")
            b = meta["budgets"][arm]
            assert b["dense_macs"] == per_row * m + head
            assert b["dense_calls"] == 197 and b["non_head_rows"] == 196 * m
            assert b["head_macs"] == head
            assert b["attention_dense_pairs"] == 28 * 12 * m * n
            assert b["attention_macs"] == 2 * 128 * b["attention_dense_pairs"]
            assert len(profile["dense"]) == 197
            assert all(p["query_shape"][2] == m and p["key_shape"][2] == n for p in profile["attention"])
        b = next(r for r in budget if r["key"] == key)
        ratio = (per_row * (n - k) + head) / (per_row * n + head)
        near(b["dense_ratio"], ratio)
        assert b["dense_budget_passes"] == "True" and ratio <= .05
        near(b["token_rows_ratio"], (n - k) / n)
        ratios.append(ratio)
        assert int(b["prefix_bytes"]) == meta["prefix_bytes"] == 2 * 28 * 2 * k * 128 * 4
        t = [r for r in timing if r["key"] == key]
        assert len(t) == 3
        for repeat in range(3):
            r = next(p for p in t if int(p["repetition"]) == repeat)
            near(r["speedup"], float(r["full_seconds"]) / float(r["suffix_seconds"]))
            assert r["order"] == ("P/T" if (index + repeat) % 2 == 0 else "T/P")
        speeds.append(median(float(r["full_seconds"]) for r in t) / median(float(r["suffix_seconds"]) for r in t))
    near(audit["max_dense_ratio"], max(ratios))
    near(audit["latency"]["median_case_speedup"], median(speeds))
    assert audit["latency"]["cases_faster"] == sum(v > 1 for v in speeds) == 32
    assert audit["latency"]["secondary_material_speedup_passes"] == (median(speeds) >= 1.25 and sum(v > 1 for v in speeds) >= 24)
    assert audit["counts"] == dict(prefix_KV_equalities=7168, prefix_residual_equalities=896,
                                   independent_profile_formula_checks=960)
    review = read(DATA / "research-review.json")
    assert review["posthoc"] and review["model_calls"] == 0 and review["current_269_271_batches_complete"]
    assert review["source_sha256"] == sha(ROOT / "research/boolean_coupling/review272.py")
    assert len(review["design_artifacts"]) == 57 and len(review["current_batch"]) == 4
    for name, value in review["inputs"].items():
        assert sha(ROOT / name) == value
    plot = read(ROOT / "docs/figures/cache/provenance.json")
    assert plot["posthoc"] and plot["new_model_calls"] == 0
    for name, value in plot["inputs"].items():
        assert sha(ROOT / name) == value
    print("272: all32 old cases, causal reuse, measured work/timing, retained299 raw failures and completion/target audit verified.")


if __name__ == "__main__":
    main()
