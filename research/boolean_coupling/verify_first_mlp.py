"""Standard-library integrity checks for276/277, preserving the failed gain account."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {276: "first-source-mlp-gain-276", 277: "first-source-mlp-forecast-277"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def passes(row):
    assert row["passes"] in ("True", "False")
    return row["passes"] == "True"


def main():
    total = 0
    for number, name in FOLDERS.items():
        dest = ROOT / "results" / name
        manifest = read(dest / "manifest.json")
        for relative, value in manifest["files"].items():
            p = Path(relative)
            assert not p.is_absolute() and ".." not in p.parts
            assert "drafts" not in p.parts and ".context" not in p.parts and p.suffix != ".npz"
            assert sha(ROOT / p) == value, relative
            total += 1
        design, audit = read(dest / "design.json"), read(dest / "audit.json")
        assert design["experiment"] == audit["experiment"] == number and audit["status"] == "passed"
        assert manifest["scientific_freeze"] == design["source_commit"]
        for relative, value in design["source_hashes"].items():
            assert sha(ROOT / relative) == value
        assert audit["design_sha256"] == sha(dest / "design.json")
        assert len(design["cases"]) == len(design["partners"]) == 32
        for key, partner in design["partners"].items():
            assert key != partner and design["partners"][partner] == key
        for relative, value in audit["csv_sha256"].items():
            assert sha(dest / relative) == value
        for label in ("native", "combined"):
            data = rows(dest / (label + ".csv"))
            counts = {arm: sum(passes(r) for r in data if r["arm"] == arm) for arm in audit[label + "_counts"]}
            assert counts == audit[label + "_counts"]
            assert len(data) == 32 * len(counts)
        combined = rows(dest / "combined.csv")
        native = rows(dest / "native.csv")
        paired = rows(dest / "paired.csv")
        for row in combined:
            n = next(r for r in native if (r["key"], r["arm"]) == (row["key"], row["arm"]))
            p = next(r for r in paired if (r["key"], r["arm"]) == (row["key"], row["arm"]))
            assert passes(row) == (passes(n) and passes(p))
        assert audit["combined_counts"]["S"] == 32 and set(audit["S_strata"].values()) == {8}
        assert audit["combined_counts"]["L"] == audit["combined_counts"]["C"] == 0
        for arm in ("G0", "G1", "G2", "Z"):
            assert audit["combined_counts"][arm] == 0
        assert audit["primary_passes"] and not audit["gain_explanation_passes"] and not audit["generative_account_passes"]
        explanations = rows(dest / "explanation.csv")
        assert len(explanations) == 32 and sum(passes(r) for r in explanations) == audit["explanation_count"] == (8 if number == 276 else 6)
        derivative_name = "derivatives.csv" if number == 276 else "forecast-derivatives.csv"
        derivative = rows(dest / derivative_name)
        assert len(derivative) == 64
        for row in derivative:
            assert float(row["bound"]) == .0001
            for k, v in row.items():
                if k not in ("key", "anchor", "bound"):
                    assert float(v) <= .0001
        qualification = audit if number == 276 else read(dest / "forecast-audit.json")
        assert qualification["numerical_checks"] == (6112 if number == 276 else 5376)
        assert qualification["derivative_anchors"] == 64 and qualification["raw_checks"]["total"] == 768
        raw = rows(dest / ("raw-primitives.csv" if number == 276 else "forecast-raw-primitives.csv"))
        assert len(raw) == 768 and all(float(r["original_bound"]) == .0001 for r in raw)
        assert sum(not passes(r) for r in raw) == qualification["raw_checks"]["failed"] == 0
        inventory = read(dest / "raw-array-inventory.json")
        assert inventory["omitted"] and len(inventory["arrays"]) == (32 if number == 276 else 64)
        assert all(a["bytes"] > 0 and len(a["sha256"]) == 64 and a["path"].endswith(".npz") for a in inventory["arrays"])
        release = read(dest / "release-notes.json")
        assert not any(release[k] for k in ("general_question_B_solved", "learned_circuit_identified", "novelty_priority_claimed", "efficiency_claimed", "private_articles_included"))
        if number == 276:
            assert audit["combined_counts"]["A"] == 8
            assert read(dest / "status.json")["counts"]["full_forwards"] == 448
        else:
            assert audit["combined_counts"]["A"] == 0
            seal, started = read(dest / "forecast-seal.json"), read(dest / "native-started.json")
            assert seal["ordinary_true"] == qualification["ordinary_true"] == started["ordinary_true_before"] == 0
            assert seal["globally_pre_native"] and seal["created"] < started["started"]
            assert started["public_commit"] == audit["public_receipt_commit"] == "7a398a580bbe879ef051e76e70af9769d51008d6"
            assert started["forecast_seal_sha256"] == audit["forecast_seal_sha256"] == sha(dest / "forecast-seal.json")
            assert seal["audit_sha256"] == sha(dest / "forecast-audit.json")
            assert seal["status_sha256"] == sha(dest / "forecast-status.json")
            assert seal["design_sha256"] == sha(dest / "design.json")
            for p in (dest / "pre-native").iterdir():
                if p.name != "README.md":
                    assert sha(p) == sha(dest / p.name)
            assert read(dest / "forecast-status.json")["counts"]["native_True_source_MLP_primals"] == 0
            assert read(dest / "native-status.json")["ordinary_true"] == 32
            history = set(read(ROOT / "research/boolean_coupling/historical-input-hashes-277.json")["hashes"])
            seen = set()
            for case in design["cases"]:
                hs = [hashlib.sha256(json.dumps(case[n], separators=(",", ":")).encode()).hexdigest() for n in ("false_ids", "true_ids")]
                assert hs == case["input_hashes"] and not set(hs) & (seen | history)
                seen.update(hs)
            assert len(seen) == 64 and len(history) == 1920
    figure = ROOT / "docs/figures/first-mlp/provenance.json"
    provenance = read(figure)
    assert provenance["posthoc"] and provenance["model_calls"] == 0 and not provenance["predictions_modified"]
    for relative, value in provenance["source_csv_sha256"].items():
        assert sha(ROOT / relative) == value
    assert sha(ROOT / "research/boolean_coupling/describe_first_mlp.py") == provenance["script_sha256"]
    for name, value in provenance["figure_sha256"].items():
        assert sha(figure.parent / name) == value
    print(f"276/277 first-source MLP evidence PASS: {total} sealed paths; fresh receipt,128 derivative anchors,1536 raw checks; both stronger gain accounts remain FAILED.")


if __name__ == "__main__":
    main()
