"""Stdlib source integrity and separate CSV decision evaluation for274."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/source-residual-reanchoring-274"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = read(DATA / "manifest.json")
    for group in ("files", "scientific_sources"):
        for name, value in manifest[group].items():
            assert "drafts" not in Path(name).parts and sha(ROOT / name) == value, name
    design, status, audit, notes = (read(DATA / n) for n in
        ("design.json", "status.json", "audit.json", "release-notes.json"))
    assert design["experiment"] == audit["experiment"] == 274
    assert design["retrospective"] and audit["retrospective"] and notes["retrospective"]
    assert design["source_commit"] == manifest["scientific_freeze"]
    assert design["source_hashes"] == manifest["scientific_sources"]
    assert audit["design_sha256"] == sha(DATA / "design.json")
    assert audit["status_sha256"] == sha(DATA / "status.json")
    assert status["status"] == "complete" and audit["status"] == "passed"
    assert notes["old_contexts"] == 32
    assert not any(notes[k] for k in ("fresh_confirmation", "private_articles_included",
        "learned_circuit_identified", "general_question_B_solved", "novelty_priority_claimed", "speedup_claimed"))
    assert notes["counts"] == status["counts"] == dict(full_forwards=384, intervention_PV=8960,
        source_norm_calls=10752, source_V_projection_calls=10752)
    arrays = read(DATA / "raw-array-inventory.json")
    assert arrays["omitted"] and len(arrays["arrays"]) == 32
    inventory = {r["path"]: r for r in arrays["arrays"]}
    assert sum(r["bytes"] for r in inventory.values()) == status["raw_bytes"] <= 600 * 1024**2
    for name, value in audit["csv_sha256"].items():
        assert sha(DATA / name) == value
    native, effects, emission, layers, raw, heads, combined = (rows(n) for n in (
        "native.csv", "effects.csv", "emission.csv", "emission-layers.csv", "raw-primitives.csv", "heads.csv", "combined.csv"))
    assert len(native) == len(effects) == 288
    assert len(emission) == len(combined) == 256 and len(layers) == 7168
    assert len(raw) == 21504 and len(heads) == 384
    for records in (native, effects, emission, layers, raw, heads):
        for r in records:
            for name in ("error", "all_relative", "verdict_relative", "relative", "cosine"):
                if name in r:
                    assert math.isfinite(float(r[name]))
    for r in native:
        ok = all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .05
                 and float(r[s + "_cosine"]) >= .99 for s in ("all", "verdict")) and float(r["margin_error"]) <= .1
        assert (r["passes"] == "True") == ok
    for r in effects:
        bound = .005 + .25 * abs(float(r["target_margin"]))
        assert math.isclose(float(r["margin_bound"]), bound, rel_tol=1e-12, abs_tol=1e-12)
        ok = all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .25
                 and float(r[s + "_cosine"]) >= .95 for s in ("all", "verdict"))
        ok = ok and abs(float(r["target_margin"])) >= .02 and float(r["margin_error"]) <= bound
        assert (r["passes"] == "True") == ok
    good = {}
    for r in layers:
        ok = float(r["norm"]) >= .001 and float(r["relative"]) <= .10 and float(r["cosine"]) >= .995
        assert (r["passes"] == "True") == ok
        key = r["key"], r["arm"]
        good[key] = good.get(key, 0) + int(ok)
    for r in emission:
        assert int(r["good_layers"]) == good[r["key"], r["arm"]]
        ok = float(r["norm"]) >= .001 and float(r["relative"]) <= .10 and float(r["cosine"]) >= .995 and int(r["good_layers"]) >= 24
        assert (r["passes"] == "True") == ok
    em = {(r["key"], r["arm"]): r for r in emission}
    ef = {(r["key"], r["arm"]): r for r in effects}
    for r in combined:
        k = r["key"], r["arm"]
        assert r["emitter_pass"] == em[k]["passes"] and r["effect_pass"] == ef[k]["passes"]
        assert (r["passes"] == "True") == (em[k]["passes"] == ef[k]["passes"] == "True")
    arms = ("U", "P", "C", "D", "E", "R", "J", "X", "Z")
    counts = {a: sum(r["passes"] == "True" for r in native if r["arm"] == a) for a in arms}
    ec = {a: sum(r["passes"] == "True" for r in effects if r["arm"] == a) for a in arms}
    cc = {a: sum(r["passes"] == "True" for r in combined if r["arm"] == a) for a in arms if a != "Z"}
    strata = {f"{s}/{rev}": sum(r["passes"] == "True" for r in combined if r["arm"] == "P" and r["schema"] == s and int(r["reverse"]) == rev)
        for s in ("generic", "descriptive") for rev in (0, 1)}
    assert audit["native_counts"] == counts and audit["effect_counts"] == ec
    assert audit["combined_counts"] == cc and audit["P_strata"] == strata
    assert audit["primary_passes"] == (cc["P"] >= 24 and min(strata.values()) >= 6)
    assert counts["U"] == cc["U"] == ec["U"] == 32
    for r in raw:
        assert float(r["original_bound"]) == .0001
        assert (r["passes"] == "True") == (float(r["error"]) <= .0001)
    assert audit["raw_checks"] == dict(total=21504, failed=sum(r["passes"] == "False" for r in raw), original_bound=.0001)
    assert all(float(r["public_error"]) == 0 and float(r["cpu_error"]) <= .005 for r in heads)
    cases = {c["key"]: c for c in design["cases"]}
    assert len(cases) == len(status["cases"]) == 32 and design["fixed_reference"] == design["cases"][0]["key"]
    for entry in status["cases"]:
        key = entry["key"]
        donor = design["partners"][key]
        a, b = cases[key]["group"], cases[donor]["group"]
        assert design["partners"][donor] == key and a["schema"] != b["schema"]
        assert {k: v for k, v in a.items() if k != "schema"} == {k: v for k, v in b.items() if k != "schema"}
        meta_path = DATA / "cases" / (key + ".json")
        meta = read(meta_path)
        assert sha(meta_path) == entry["metadata_sha256"]
        assert meta["arrays_sha256"] == entry["arrays_sha256"] == inventory["cases/" + key + ".npz"]["sha256"]
        assert meta["donor"] == donor and meta["fixed_reference"] == design["fixed_reference"]
        assert all(meta[k] for k in ("source_primal_replay_exact", "source_KV_unchanged", "prefix_and_source_exact", "self_and_zero_exact", "F_T_X_Z_archived_replay_exact"))
    provenance = read(ROOT / "docs/figures/reanchoring/provenance.json")
    assert provenance["posthoc"] and provenance["model_calls"] == 0
    assert provenance["source_sha256"] == sha(ROOT / "research/boolean_coupling/describe_reanchoring.py")
    for name, value in provenance["inputs"].items():
        assert sha(ROOT / name) == value
    print(f"274: all32 old cases and source seals verified; {cc['P']}/32 combined primary passes, {ec['P']}/32 effect passes. Private article excluded.")


if __name__ == "__main__":
    main()
