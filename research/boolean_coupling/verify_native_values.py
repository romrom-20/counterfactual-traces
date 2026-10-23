"""Stdlib integrity and independent CSV rule evaluation for native study273."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/native-value-interchange-273"


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
    assert design["experiment"] == 273 and design["retrospective"] and notes["retrospective"]
    assert design["source_commit"] == manifest["scientific_freeze"]
    assert design["source_hashes"] == manifest["scientific_sources"]
    assert audit["design_sha256"] == sha(DATA / "design.json")
    assert audit["status_sha256"] == sha(DATA / "status.json")
    assert status["status"] == "complete" and audit["status"] == "passed"
    assert not any(notes[k] for k in ("fresh_confirmation", "private_articles_included",
        "learned_circuit_identified", "general_question_B_solved", "novelty_priority_claimed", "speedup_claimed"))
    assert notes["old_contexts"] == 32
    assert status["counts"] == notes["counts"] == dict(full_forwards=288, source_blocks=896,
        intervention_PV=6272, diagnostic_PV=1792, diagnostic_output_projections=2688)
    inventory = read(DATA / "raw-array-inventory.json")
    assert inventory["omitted"] and len(inventory["arrays"]) == 64
    arrays = {r["path"]: r for r in inventory["arrays"]}
    assert sum(r["bytes"] for r in arrays.values()) == status["raw_bytes"]
    assert status["raw_bytes"] <= 600 * 1024**2
    for name, value in audit["csv_sha256"].items():
        assert sha(DATA / name) == value
    native, effects, source, local, raw, heads = (rows(DATA / n) for n in (
        "native.csv", "effects.csv", "source-geometry.csv", "local-vectors.csv", "raw-source-projections.csv", "heads.csv"))
    assert len(native) == len(effects) == 192
    assert len(source) == len(local) == len(raw) == 1792 and len(heads) == 288
    for table in (native, effects, source, local, raw, heads):
        for r in table:
            for name in ("error", "all_relative", "verdict_relative", "cosine", "relative"):
                if name in r:
                    assert math.isfinite(float(r[name]))
    for r in native:
        ok = all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .05
                 and float(r[s + "_cosine"]) >= .99 for s in ("all", "verdict")) and float(r["margin_error"]) <= .1
        assert (r["passes"] == "True") == ok
        near(r["margin_ratio"], float(r["margin_response"]) / float(r["native_margin_response"]))
    for r in effects:
        bound = .005 + .25 * abs(float(r["target_margin_effect"]))
        near(r["margin_bound"], bound)
        ok = all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .25
                 and float(r[s + "_cosine"]) >= .95 for s in ("all", "verdict"))
        ok = ok and abs(float(r["target_margin_effect"])) >= .02 and float(r["margin_effect_error"]) <= bound
        assert (r["passes"] == "True") == ok
    names = ("U", "X", "D", "A", "Z", "R")
    counts = {a: sum(r["passes"] == "True" for r in native if r["arm"] == a) for a in names}
    ec = {a: sum(r["passes"] == "True" for r in effects if r["arm"] == a) for a in names}
    strata = {f"{s}/{rev}": sum(r["passes"] == "True" for r in effects if r["arm"] == "X" and r["schema"] == s and int(r["reverse"]) == rev)
        for s in ("generic", "descriptive") for rev in (0, 1)}
    assert audit["native_counts"] == counts and audit["effect_counts"] == ec and audit["X_strata"] == strata
    assert audit["primary_passes"] == (ec["X"] >= 24 and min(strata.values()) >= 6)
    assert counts["U"] == ec["U"] == 32
    assert audit["counts"] == dict(archived_endpoint_and_score_equalities=128, complete_attention_witnesses=96)
    for r in raw:
        assert float(r["original_bound"]) == .0001
        assert (r["passes"] == "True") == (float(r["error"]) <= .0001)
    assert audit["raw_source_checks"] == dict(total=1792, failed=sum(r["passes"] == "False" for r in raw), original_bound=.0001)
    assert all(float(r["public_error"]) == 0 and float(r["cpu_error"]) <= .005 for r in heads)
    cases = {c["key"]: c for c in design["cases"]}
    pairs = {a: b for pair in design["pairs"] for a, b in (pair, pair[::-1])}
    assert len(cases) == len(pairs) == len(status["cases"]) == 32
    geometry = {(r["key"], r["layer"], r["head"]): r for r in source}
    for entry in status["cases"]:
        key, donor = entry["key"], entry["donor"]
        assert pairs[key] == donor and pairs[donor] == key
        a, b = cases[key]["group"], cases[donor]["group"]
        assert a["schema"] != b["schema"] and {k: v for k, v in a.items() if k != "schema"} == {k: v for k, v in b.items() if k != "schema"}
        path = DATA / "cases" / (key + ".json")
        meta = read(path)
        assert sha(path) == entry["metadata_sha256"]
        assert meta["donor"] == donor and meta["arrays_sha256"] == entry["arrays_sha256"]
        assert arrays["cases/" + key + ".npz"]["sha256"] == entry["arrays_sha256"]
        assert arrays["witnesses/" + key + ".npz"]["sha256"] == meta["witness_sha256"] == entry["witness_sha256"]
        assert len(meta["controls"]) == 8
        assert all(meta[k] for k in ("source_generator_replay_exact", "self_install_replay_exact", "zero_edit_replay_exact"))
        assert meta["historical"]["F_T_replay_exact"]
        for i in range(1, 29):
            for h in range(2):
                x, y = geometry[key, str(i), str(h)], geometry[donor, str(i), str(h)]
                near(x["cosine"], y["cosine"])
                near(float(x["norm_ratio"]) * float(y["norm_ratio"]), 1.)
    plot = read(ROOT / "docs/figures/native-values/provenance.json")
    assert plot["posthoc"] and plot["model_calls"] == 0
    assert plot["source_sha256"] == sha(ROOT / "research/boolean_coupling/describe_native_values.py")
    for name, value in plot["inputs"].items():
        assert sha(ROOT / name) == value
    print(f"273: native value interchange, all32 old contexts, {ec['X']}/32 primary effect passes; numerical/source seals and all controls verified.")


if __name__ == "__main__":
    main()
