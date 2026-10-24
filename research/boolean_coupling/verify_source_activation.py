"""Standard-library279 receipt/endpoint verification, retaining materiality failures."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-activation-forecast-279"


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
    manifest, design, audit, pre = (read(DEST / n) for n in ("manifest.json", "design.json", "audit.json", "forecast-audit.json"))
    for relative, h in manifest["files"].items():
        p = Path(relative)
        assert not p.is_absolute() and not any(x in p.parts for x in ("..", ".context", "drafts")) and p.suffix != ".npz"
        assert sha(ROOT / p) == h, relative
    assert manifest["scientific_freeze"] == design["source_commit"] == "f8120436877c213441d189bd399190949cdd4308"
    for relative, h in design["source_hashes"].items():
        assert sha(ROOT / relative) == h
    assert audit["status"] == pre["status"] == "passed" and audit["experiment"] == pre["experiment"] == 279
    assert pre["ordinary_true"] == 0 and pre["design_sha256"] == audit["design_sha256"] == sha(DEST / "design.json")
    for n, h in audit["csv_sha256"].items():
        assert sha(DEST / n) == h
    for label in ("native", "effect", "local", "paired", "combined", "restoration"):
        data = rows(DEST / (label + ".csv"))
        assert len(data) == 288
        counts = {a: sum(boolean(r) for r in data if r["arm"] == a) for a in audit[label + "_counts"]}
        assert counts == audit[label + "_counts"]
        for r in data:
            if label == "native":
                expected = vec(r, "all_", .05, .99) and vec(r, "verdict_", .05, .99) and float(r["margin_error"]) <= .1
            elif label == "local":
                expected = vec(r, "", .1, .995)
            elif label in ("effect", "paired"):
                prefixes = ("all_", "verdict_") if label == "effect" else ("source_", "verdict_")
                expected = all(vec(r, p, .25, .95) for p in prefixes) and abs(float(r["target_margin"])) >= .02 and float(r["margin_error"]) <= .005 + .25 * abs(float(r["target_margin"]))
            elif label == "combined":
                expected = all(boolean(r, n) for n in ("native_pass", "effect_pass", "local_pass", "paired_pass"))
            else:
                expected = boolean(r, "local_pass") and boolean(r, "downstream_pass") and int(r["random_failures"]) >= 2
            assert boolean(r) == expected
    assert audit["primary_passes"] and audit["sparse_prediction_passes"] and not audit["sparse_correction_account_passes"]
    for arm in ("Q", "C", "S", "R0", "R1", "R2", "T"):
        assert audit["native_counts"][arm] == audit["local_counts"][arm] == audit["paired_counts"][arm] == 32
        assert audit["combined_counts"][arm] == audit["effect_counts"][arm] == 29
    assert audit["native_counts"]["L"] == 13 and audit["local_counts"]["L"] == audit["combined_counts"]["L"] == 0
    assert audit["native_counts"]["Z"] == 0 and set(audit["restoration_counts"].values()) == {0}
    assert sorted(audit["C_strata"].values()) == sorted(audit["S_strata"].values()) == [6, 7, 8, 8]
    exact = [r for r in rows(DEST / "effect.csv") if r["arm"] == "T"]
    failures = [r for r in exact if not boolean(r)]
    assert len(failures) == 3 and all(abs(float(r["target_margin"])) < .02 for r in failures)
    eligibility = rows(DEST / "eligibility.csv")
    assert len(eligibility) == 288
    for arm in ("C", "Q", "S", "T"):
        selected = [r for r in eligibility if r["arm"] == arm]
        assert len(selected) == 32 and sum(not boolean(r, "material_margin_pass") for r in selected) == 3
        assert all(boolean(r, n) for r in selected for n in ("native_pass", "local_pass", "paired_pass", "effect_vectors_pass"))
        margin_failures = {r["key"] for r in selected if not boolean(r, "effect_margin_bound_pass")}
        materiality_failures = {r["key"] for r in selected if not boolean(r, "material_margin_pass")}
        assert margin_failures == (materiality_failures if arm == "Q" else set())
    seal, started = read(DEST / "forecast-seal.json"), read(DEST / "native-started.json")
    assert seal["ordinary_true"] == started["ordinary_true_before"] == 0 and seal["globally_pre_native"]
    assert seal["created"] < started["started"] and seal["audit_sha256"] == sha(DEST / "forecast-audit.json")
    assert seal["status_sha256"] == sha(DEST / "forecast-status.json") and seal["design_sha256"] == sha(DEST / "design.json")
    assert audit["forecast_seal_sha256"] == started["forecast_seal_sha256"] == sha(DEST / "forecast-seal.json")
    assert audit["public_receipt_commit"] == started["public_commit"] == "519f4e857c1448662e0d914bccbedb495df4d4a1"
    for p in (DEST / "pre-native").iterdir():
        if p.name != "README.md":
            assert sha(p) == sha(DEST / p.name)
    assert pre["numerical_checks"] == 4928 and pre["False_gate_anchors"] == 32 and pre["coefficient_vectors_checked"] == 96
    coefficient = rows(DEST / "forecast-coefficients.csv")
    assert len(coefficient) == 32
    for r in coefficient:
        assert float(r["bound"]) == .0001 and all(float(v) <= .0001 for k, v in r.items() if k not in ("key", "bound"))
    raw = rows(DEST / "forecast-raw.csv")
    assert len(raw) == 672 and sum(not boolean(r) for r in raw) == pre["raw_checks"]["failed"] == 0
    assert all(float(r["original_bound"]) == .0001 for r in raw)
    assert audit["exact_native_operand_checks"] == 832 and audit["ordinary_true"] == 32
    history = set(read(ROOT / "research/boolean_coupling/historical-input-hashes-279.json")["hashes"])
    assert len(history) == 1984
    seen = set()
    for c in design["cases"]:
        hs = [hashlib.sha256(json.dumps(c[n], separators=(",", ":")).encode()).hexdigest() for n in ("false_ids", "true_ids")]
        assert hs == c["input_hashes"] and not set(hs) & (history | seen)
        seen.update(hs)
    assert len(seen) == 64 and len(design["partners"]) == 32
    counts = read(DEST / "forecast-status.json")["counts"]
    assert counts["full_True_source_activation_primal_count"] == 0 and counts["current_source_SiLU_coordinates_per_sparse_arm"] == 90
    assert counts["full_false"] == 32 and counts["full_forecasts"] == 288
    inv = read(DEST / "raw-array-inventory.json")
    assert inv["omitted"] and len(inv["arrays"]) == 64 and sum(x["bytes"] for x in inv["arrays"]) == 97679037
    repetition = read(DEST / "repetition-check.json")
    assert repetition["forecasts_reused"] and not repetition["fresh_confirmation"] and not repetition["full_forecasts_regenerated"]
    assert repetition["all_original_native_and_joint_counts_reproduced"] and repetition["native_references_recomputed"] == 32
    notes = read(DEST / "release-notes.json")
    assert notes["fresh_literals"] and not any(notes[k] for k in ("independent_templates", "private_articles_included", "general_question_B_solved", "learned_circuit_identified", "novelty_priority_claimed", "efficiency_claimed"))
    fig = ROOT / "docs/figures/source-activation"
    prov = read(fig / "provenance.json")
    assert prov["posthoc"] and prov["model_calls"] == 0 and not prov["predictions_modified"]
    assert prov["script_sha256"] == sha(ROOT / "research/boolean_coupling/describe_source_activation.py")
    for n, h in prov["source_csv_sha256"].items():
        assert sha(DEST / n) == h
    for n, h in prov["figure_sha256"].items():
        assert sha(fig / n) == h
    print("279 source activation evidence PASS:29/32joint,32/32original/local/paired;3materiality failures retained;public pre-native receipt;strong sparse correction FAILED0/32.")


if __name__ == "__main__":
    main()
