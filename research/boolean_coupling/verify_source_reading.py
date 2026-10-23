"""Stdlib source integrity, signed-ledger and independent decision checks for275."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/source-attention-reading-275"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(name):
    with (DATA / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), (a, b)


def main():
    manifest = read(DATA / "manifest.json")
    for group in ("files", "scientific_sources"):
        for name, value in manifest[group].items():
            assert "drafts" not in Path(name).parts and sha(ROOT / name) == value, name
    design, status, audit, notes = (read(DATA / n) for n in
        ("design.json", "status.json", "audit.json", "release-notes.json"))
    assert design["experiment"] == audit["experiment"] == 275
    assert design["retrospective"] and audit["retrospective"] and notes["retrospective"]
    assert design["source_commit"] == manifest["scientific_freeze"]
    assert design["source_hashes"] == manifest["scientific_sources"]
    assert audit["design_sha256"] == sha(DATA / "design.json")
    assert audit["status_sha256"] == sha(DATA / "status.json")
    assert status["status"] == "complete" and audit["status"] == "passed"
    assert notes["old_contexts"] == 32
    assert not any(notes[k] for k in ("fresh_confirmation", "private_articles_included",
        "learned_circuit_identified", "general_question_B_solved", "novelty_priority_claimed", "speedup_claimed"))
    assert notes["counts"] == status["counts"] == dict(full_forwards=416, source_intervention_PV=3584,
        native_corner_PV=2688, native_corner_output_projections=2688, value_intervention_PV=5376)
    previous = ROOT / "results/native-value-interchange-273"
    for name, value in design["reference_hashes"].items():
        assert sha(previous / name) == value
    assert design["cases"] == read(previous / "design.json")["cases"]
    old_entries = {r["key"]: r for r in read(previous / "status.json")["cases"]}
    inventory = read(DATA / "raw-array-inventory.json")
    assert inventory["omitted"] and len(inventory["arrays"]) == 32
    arrays = {r["path"]: r for r in inventory["arrays"]}
    assert sum(r["bytes"] for r in arrays.values()) == status["raw_bytes"] <= 1000 * 1024**2
    for name, value in audit["csv_sha256"].items():
        assert sha(DATA / name) == value
    names = ("native", "effects", "emission", "emission-layers", "raw-primitives", "heads", "combined",
             "source-ledger", "local-attention", "paired-source-ledger")
    tables = {n: rows(n) for n in names}
    expected = dict(zip(names, (288, 160, 128, 3584, 26880, 416, 128, 3584, 2688, 2688), strict=True))
    for name, records in tables.items():
        assert len(records) == expected[name], name
        for r in records:
            for field in ("error", "all_relative", "verdict_relative", "cosine", "relative", "signed_projection"):
                if field in r:
                    assert math.isfinite(float(r[field]))
    for r in tables["native"]:
        ok = all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .05
                 and float(r[s + "_cosine"]) >= .99 for s in ("all", "verdict")) and float(r["margin_error"]) <= .1
        assert (r["passes"] == "True") == ok
    for r in tables["effects"]:
        bound = .005 + .25 * abs(float(r["target_margin"]))
        close(r["margin_bound"], bound)
        ok = all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .25
                 and float(r[s + "_cosine"]) >= .95 for s in ("all", "verdict"))
        ok = ok and abs(float(r["target_margin"])) >= .02 and float(r["margin_error"]) <= bound
        assert (r["passes"] == "True") == ok
    good = {}
    for r in tables["emission-layers"]:
        ok = float(r["norm"]) >= .001 and float(r["relative"]) <= .1 and float(r["cosine"]) >= .995
        assert (r["passes"] == "True") == ok
        k = r["key"], r["arm"]
        good[k] = good.get(k, 0) + int(ok)
    for r in tables["emission"]:
        assert int(r["good_layers"]) == good[r["key"], r["arm"]]
        ok = float(r["norm"]) >= .001 and float(r["relative"]) <= .1 and float(r["cosine"]) >= .995 and int(r["good_layers"]) >= 24
        assert (r["passes"] == "True") == ok
    emission = {(r["key"], r["arm"]): r for r in tables["emission"]}
    effects = {(r["key"], r["arm"]): r for r in tables["effects"]}
    for r in tables["combined"]:
        arm = "U" if r["arm"] == "N" else r["arm"]
        e = emission[r["key"], r["arm"]]
        f = effects[r["key"], arm]
        assert r["emitter_pass"] == e["passes"] and r["effect_pass"] == f["passes"]
        assert (r["passes"] == "True") == (e["passes"] == f["passes"] == "True")
    cc = {a: sum(r["passes"] == "True" for r in tables["combined"] if r["arm"] == a) for a in ("N", "W", "V", "B")}
    nc = {a: sum(r["passes"] == "True" for r in tables["native"] if r["arm"] == a)
          for a in ("N", "W", "V", "B", "value_U", "value_W", "value_V", "value_B", "value_Z")}
    ec = {a: sum(r["passes"] == "True" for r in tables["effects"] if r["arm"] == a) for a in ("U", "W", "V", "B", "Z")}
    strata = {f"{s}/{rev}": sum(r["passes"] == "True" for r in tables["combined"] if r["arm"] == "V" and r["schema"] == s and int(r["reverse"]) == rev)
              for s in ("generic", "descriptive") for rev in (0, 1)}
    assert audit["native_counts"] == nc and audit["effect_counts"] == ec
    assert audit["combined_counts"] == cc and audit["V_strata"] == strata
    assert audit["primary_passes"] == (cc["V"] >= 24 and min(strata.values()) >= 6 and cc["W"] <= 8)
    assert nc["N"] == nc["value_U"] == cc["N"] == ec["U"] == 32
    for r in tables["raw-primitives"]:
        assert float(r["original_bound"]) == .0001
        assert (r["passes"] == "True") == (float(r["error"]) <= .0001)
    assert audit["raw_checks"] == dict(total=26880, failed=sum(r["passes"] == "False" for r in tables["raw-primitives"]), original_bound=.0001)
    assert all(float(r["public_error"]) == 0 and float(r["cpu_error"]) <= .005 for r in tables["heads"])
    for name, number in (("source-ledger", 4), ("local-attention", 3), ("paired-source-ledger", 3)):
        grouped = {}
        for r in tables[name]:
            grouped.setdefault((r["key"], r["layer"]), []).append(r)
        assert len(grouped) == 896
        for records in grouped.values():
            assert len(records) == number
            if float(records[0]["target_norm"]) >= 1e-9:
                close(sum(float(r["signed_projection"]) for r in records), 1.)
    for entry in status["cases"]:
        key = entry["key"]
        path = DATA / "cases" / (key + ".json")
        meta = read(path)
        assert sha(path) == entry["metadata_sha256"]
        assert meta["input_sha256"] == old_entries[key]["arrays_sha256"]
        assert meta["arrays_sha256"] == entry["arrays_sha256"] == arrays["cases/" + key + ".npz"]["sha256"]
        assert all(meta[k] for k in ("false_and_native_exact", "unchanged_prefix_all_layers", "source_query_only_install",
            "native_corner_B_matches_F", "native_value_KV_exact", "self_zero_archived_exact", "full_factor_zero_edit_exact"))
    first = read(DATA / "first-source-provenance.json")
    assert first["posthoc"] and first["model_calls"] == 0
    assert first["source_sha256"] == sha(ROOT / "research/boolean_coupling/describe_first_source.py")
    assert first["input_status_sha256"] == sha(DATA / "status.json")
    assert first["input_audit_sha256"] == sha(DATA / "audit.json")
    assert first["arrays_sha256"] == {r["key"]: r["arrays_sha256"] for r in status["cases"]}
    for name in ("first-source", "first-source-pairs"):
        records = rows(name)
        assert len(records) == 128
        for r in records:
            close(r["norm_ratio"], float(r["candidate_norm"]) / float(r["target_norm"]))
            close(r["signed_projection"], float(r["cosine"]) * float(r["norm_ratio"]))
            close(float(r["relative"]) ** 2, 1 + float(r["norm_ratio"]) ** 2 - 2 * float(r["signed_projection"]))
    provenance = read(ROOT / "docs/figures/source-reading/provenance.json")
    assert provenance["posthoc"] and provenance["model_calls"] == 0
    assert provenance["source_sha256"] == sha(ROOT / "research/boolean_coupling/describe_source_reading.py")
    for name, value in provenance["inputs"].items():
        assert sha(ROOT / name) == value
    print(f"275: source/probability/self-value factorial, all32 old cases and signed ledgers verified; combined V={cc['V']}, W={cc['W']}; primary={audit['primary_passes']}. No circuit or fresh forecast claimed.")


if __name__ == "__main__":
    main()
