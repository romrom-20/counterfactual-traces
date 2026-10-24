"""Standard-library278 evidence verification; normalization-only account stays failed."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-mlp-primitives-278"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def boolean(row, key="passes"):
    assert row[key] in ("True", "False")
    return row[key] == "True"


def vec(row, prefix, bound, cosine):
    return float(row[prefix + "norm"]) >= .001 and float(row[prefix + "relative"]) <= bound and float(row[prefix + "cosine"]) >= cosine


def main():
    manifest, design, audit = (read(DEST / n) for n in ("manifest.json", "design.json", "audit.json"))
    for relative, h in manifest["files"].items():
        p = Path(relative)
        assert not p.is_absolute() and not any(x in p.parts for x in ("..", ".context", "drafts")) and p.suffix != ".npz"
        assert sha(ROOT / p) == h, relative
    assert manifest["scientific_freeze"] == design["source_commit"] == "828706083a0c8e0eaecf088379a902a4ec41c9b8"
    for relative, h in design["source_hashes"].items():
        assert sha(ROOT / relative) == h
    assert audit["status"] == "passed" and audit["retrospective"] and audit["experiment"] == 278
    assert audit["design_sha256"] == sha(DEST / "design.json") and audit["status_sha256"] == sha(DEST / "status.json")
    for n, h in audit["csv_sha256"].items():
        assert sha(DEST / n) == h
    assert len(design["cases"]) == len(design["partners"]) == 32
    for k, v in design["partners"].items():
        assert k != v and design["partners"][v] == k
    for label in ("native", "effect", "local", "paired", "combined", "restoration"):
        data = rows(DEST / (label + ".csv"))
        assert len(data) == 672
        counts = {a: sum(boolean(r) for r in data if r["arm"] == a) for a in audit[label + "_counts"]}
        assert counts == audit[label + "_counts"]
        if label == "native":
            for r in data:
                expected = vec(r, "all_", .05, .99) and vec(r, "verdict_", .05, .99) and float(r["margin_error"]) <= .1
                assert boolean(r) == expected
        elif label == "local":
            for r in data:
                assert boolean(r) == vec(r, "", .1, .995)
        elif label == "effect":
            for r in data:
                expected = vec(r, "all_", .25, .95) and vec(r, "verdict_", .25, .95) and abs(float(r["target_margin"])) >= .02 and float(r["margin_error"]) <= .005 + .25 * abs(float(r["target_margin"]))
                assert boolean(r) == expected
        elif label == "paired":
            for r in data:
                expected = vec(r, "source_", .25, .95) and vec(r, "verdict_", .25, .95) and abs(float(r["target_margin"])) >= .02 and float(r["margin_error"]) <= .005 + .25 * abs(float(r["target_margin"]))
                assert boolean(r) == expected
        elif label == "combined":
            for r in data:
                assert boolean(r) == all(boolean(r, n) for n in ("native_pass", "effect_pass", "local_pass", "paired_pass"))
        else:
            for r in data:
                assert boolean(r) == (boolean(r, "local_pass") and boolean(r, "downstream_pass") and int(r["random_failures"]) >= 2)
    assert not audit["primary_passes"] and not audit["normalization_restoration_passes"] and not audit["normalization_mechanistic_account_passes"]
    assert audit["prespecified_secondary_candidates"] == []
    assert audit["combined_counts"]["N111"] == audit["combined_counts"]["T"] == 32
    assert all(n == 0 for a, n in audit["combined_counts"].items() if a not in ("N111", "T"))
    assert audit["native_counts"]["N101"] == 11 and audit["paired_counts"]["N101"] == 32 and audit["local_counts"]["N101"] == 0
    assert audit["native_counts"]["C111"] == 32 and audit["paired_counts"]["C111"] == 0
    assert audit["numerical_checks"] == 21600 and audit["primitive_derivative_directions"] == 448
    derivatives = rows(DEST / "derivatives.csv")
    assert len(derivatives) == 448
    for r in derivatives:
        assert float(r["bound"]) == .0001
        assert all(float(v) <= .0001 for k, v in r.items() if k not in ("key", "primitive", "bound"))
    raw = rows(DEST / "raw.csv")
    assert len(raw) == 4608 and all(float(r["original_bound"]) == .0001 for r in raw)
    assert sum(not boolean(r) for r in raw) == audit["raw_checks"]["failed"] == 0
    assert len(rows(DEST / "heads.csv")) == 736 and all(float(r["public_error"]) == 0 and float(r["cpu_error"]) <= .005 for r in rows(DEST / "heads.csv"))
    assert len(rows(DEST / "ledger.csv")) == 1344
    ledgers = rows(DEST / "paired-source-ledger.csv")
    assert len(ledgers) == 640 and all(float(r["closure_error"]) <= 1e-10 for r in ledgers)
    assert len(rows(DEST / "margins.csv")) == 736 and len(rows(DEST / "margin-ledger.csv")) == 448
    comparisons = rows(DEST / "ablation-comparisons.csv")
    assert len(comparisons) == 192
    for a, b, count in (("N100", "N000", 0), ("N110", "N010", 0), ("N101", "N001", 32), ("N111", "N011", 32)):
        selected = [r for r in comparisons if r["after"] == a and r["before"] == b]
        assert len(selected) == 32
        assert sum(boolean(r, "all_relative_improved") for r in selected) == count
        assert sum(boolean(r, "verdict_relative_improved") for r in selected) == count
    posthoc = read(DEST / "posthoc-provenance.json")
    assert posthoc["posthoc"] and posthoc["model_calls"] == 0 and not posthoc["forecasts_modified"]
    assert sha(ROOT / "research/boolean_coupling/describe_source_primitives.py") == posthoc["script_sha256"]
    assert posthoc["frozen_audit_sha256"] == sha(DEST / "audit.json")
    for n, h in posthoc["csv_sha256"].items():
        assert sha(DEST / n) == h
    inventory = read(DEST / "raw-array-inventory.json")
    assert inventory["omitted"] and len(inventory["arrays"]) == 32 and sum(r["bytes"] for r in inventory["arrays"]) == 375665311
    assert read(DEST / "status.json")["counts"]["full_forwards"] == 736
    notes = read(DEST / "release-notes.json")
    assert not any(notes[k] for k in ("fresh_confirmation", "private_articles_included", "general_question_B_solved", "learned_circuit_identified", "novelty_priority_claimed", "efficiency_claimed"))
    fig = ROOT / "docs/figures/source-primitives"
    prov = read(fig / "provenance.json")
    assert prov["posthoc"] and prov["model_calls"] == 0
    assert prov["script_sha256"] == sha(ROOT / "research/boolean_coupling/describe_source_primitives.py")
    for n, h in prov["source_csv_sha256"].items():
        assert sha(DEST / n) == h
    for n, h in prov["figure_sha256"].items():
        assert sha(fig / n) == h
    print("278 source primitive evidence PASS:all32 old cases;21600 checks/448 derivative directions/4608 raw comparisons;normalization-only primary remains FAILED0/32.")


if __name__ == "__main__":
    main()
