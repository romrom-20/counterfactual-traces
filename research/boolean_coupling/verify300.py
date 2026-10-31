"""Stdlib validation of source seals, controls, all decisions and signed statistics."""

import argparse
from pathlib import Path

try:
    from .verify285 import boolean, close, number, read, relative, require, sha
    from .verify286 import source_seal
    from .verify288 import rows
    from .verify293 import vstats
except ImportError:
    from verify285 import boolean, close, number, read, relative, require, sha
    from verify286 import source_seal
    from verify288 import rows
    from verify293 import vstats

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/query-interaction-300")


def verify(root):
    dest = root / DEST
    manifest = read(dest / "release-manifest.json")
    require(manifest["experiment"] == 300 and manifest["retrospective"], "Release identity")
    for n, h in manifest["files"].items():
        require(sha(relative(root, n)) == h, "Release bytes:" + n)
        require(
            not (set(Path(n).parts) & {".context", "drafts"})
            and Path(n).suffix not in {".npz", ".npy", ".pt", ".safetensors"},
            "Public scope",
        )
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    require(
        actual == {n for n in manifest["files"] if n.startswith(str(DEST) + "/")},
        "Exact release inventory",
    )
    d, status, audit = [read(dest / n) for n in ("design.json", "status.json", "audit.json")]
    source_seal(root, d, "interaction300")
    require(len(d["source_hashes"]) == 154, "154 immutable sources")
    require(
        d["retrospective"] and d["native_parents_supplied"] and not d["false_only_forecast"],
        "Native parent information budget",
    )
    require(
        d["criteria"]
        == dict(
            norm=0.001,
            relative=0.25,
            cosine=0.95,
            material=0.02,
            margin_abs=0.1,
            margin_fraction=0.25,
            population=6,
            per_schema=3,
            per_layout=1,
            control_max=2,
        ),
        "Frozen criteria",
    )
    require(
        d["bounds"]
        == dict(
            composed=1e-4,
            projection_abs=1e-3,
            projection_relative=1e-5,
            contrast_relative=1e-3,
            head=1e-4,
            raw_diagnostic=1e-4,
            random_norm_relative=1e-4,
        ),
        "Frozen numerical bounds",
    )
    require(
        audit["status"] == "passed"
        and not audit["scientific_outcomes_withheld"]
        and audit["design_sha256"] == sha(dest / "design.json")
        and audit["status_sha256"] == sha(dest / "status.json"),
        "Qualification gate",
    )
    expected_counts = dict(
        full_forwards=48,
        extra_full_QK=8,
        extra_full_softmax=8,
        extra_full_PV=8,
        extra_full_output_projection=32,
    )
    require(
        status["counts"] == expected_counts
        and status["ordinary_native_calibration_repetitions"] == 8
        and status["conditional_intervention_forwards"] == 40,
        "Complete work",
    )
    parents = d["parents"]
    for path, field in (
        ("results/query-generators-299/pre-native/design.json", "parent_public_design_sha256"),
        (
            "results/query-generators-299/native-phase/native-status.json",
            "parent_native_status_sha256",
        ),
        (
            "results/query-generators-299/native-phase/release-manifest.json",
            "parent_release_sha256",
        ),
    ):
        require(sha(root / path) == parents[field], "299 parent release")
    require(
        parents["retrospective"]
        and parents["parents_already_native"]
        and parents["selected_block"] == 15
        and len(parents["cases"]) == 8,
        "Preselected old parent scope",
    )
    parent_entries = {
        e["key"]: e
        for e in read(root / "results/query-generators-299/native-phase/native-status.json")[
            "cases"
        ]
    }
    for c in parents["cases"]:
        e = parent_entries[c["case"]["key"]]
        require(
            c["native_arrays_sha256"] == e["arrays_sha256"]
            and c["native_metadata_sha256"] == e["metadata_sha256"]
            and c["witness_sha256"] == e["witness_sha256"],
            "Exact parent bindings",
        )
    inv = read(dest / "omitted-array-inventory.json")["arrays"]
    require(
        sum(x["bytes"] for x in inv) == status["raw_bytes"] <= 64 * 1024**2,
        "Raw quota and inventory",
    )
    require(
        [e["key"] for e in status["cases"]] == [c["case"]["key"] for c in parents["cases"]],
        "Ordered cohort",
    )
    for e, item in zip(status["cases"], inv, strict=True):
        p = dest / "cases" / (e["key"] + ".json")
        m = read(p)
        require(
            sha(p) == e["metadata_sha256"]
            and m["arrays_sha256"] == item["sha256"] == e["arrays_sha256"]
            and e["bytes"] == item["bytes"],
            "Case and omitted array binding",
        )
        require(
            m["retrospective"] and m["native_parents_supplied"] and m["block"] == 15, "Case scope"
        )
        require(
            all(r["passes"] and r["error"] <= r["bound"] for r in m["checks"]),
            "Exact native, protected and installation checks",
        )
    rr = rows(dest / "checks.csv")
    require(
        len(rr) == audit["required_checks"] == 864
        and all(boolean(r, "passes") and number(r, "error") <= number(r, "bound") for r in rr),
        "Required checks",
    )
    raw = rows(dest / "raw-coordinates.csv")
    require(len(raw) == audit["raw_checks"] == 40, "Raw denominator")
    for r in raw:
        require(boolean(r, "passes") == (number(r, "error") <= 1e-4), "Original raw diagnostics")
    require(
        sum(not boolean(r, "passes") for r in raw) == audit["raw_failed"],
        "Failed diagnostics retained",
    )
    for n, h in audit["csv_sha256"].items():
        require(sha(dest / n) == h, "Audit table binding")
    stats = read(dest / "vector-records.json")["statistics"]
    ix = {(r["key"], r["arm"], r["scope"]): r for r in stats}
    require(len(stats) == len(ix) == 80, "All final effect and suffix-interaction vectors")
    for r in stats:
        vstats(r)
    local = read(dest / "local-records.json")["statistics"]
    require(
        len(local)
        == sum(len(c["case"]["true_ids"]) - c["case"]["position"] - 1 for c in parents["cases"])
        * 3,
        "All signed receiving writes",
    )
    for r in local:
        vstats(r)
    scores = {r["key"]: r["scores"] for r in read(dest / "scores.json")["cases"]}
    effects = rows(dest / "effects.csv")
    require(
        len(effects) == 32 and len({(r["key"], r["arm"]) for r in effects}) == 32, "All decisions"
    )
    for r in effects:
        key, arm = r["key"], r["arm"]

        def margin(a):
            s = scores[key][a + "_scores"]
            return s[1] - s[0]

        effect = margin("T0") - margin("L")
        error = abs(margin(arm) - margin("T0"))
        bound = min(0.1, 0.25 * abs(effect))
        close(number(r, "native_margin_effect"), effect, "Native signed margin effect")
        close(
            number(r, "arm_margin_effect"),
            margin(arm) - margin("L"),
            "Installed signed margin effect",
        )
        close(number(r, "margin_error"), error, "Actual finite margin error")
        close(number(r, "margin_bound"), bound, "Relative margin bound")
        material = abs(effect) >= 0.02
        passed = material and error <= bound
        for scope in ("receiving", "verdict"):
            n, re, co = vstats(ix[key, arm, scope])
            close(number(r, scope + "_relative"), re, "Vector metric")
            passed = passed and n >= 0.001 and re <= 0.25 and co >= 0.95
        require(
            boolean(r, "passes") == passed and boolean(r, "material") == material,
            "Complete frozen decision including materiality",
        )
    pop = {}
    for arm in ("A", "R", "J", "O"):
        rr = [r for r in effects if r["arm"] == arm]
        schemas = {
            s: sum(boolean(r, "passes") for r in rr if r["schema"] == s)
            for s in {r["schema"] for r in rr}
        }
        layouts = {
            s: sum(boolean(r, "passes") for r in rr if r["layout"] == s)
            for s in {r["layout"] for r in rr}
        }
        n = sum(schemas.values())
        pop[arm] = dict(
            passes=n,
            total=8,
            schemas=schemas,
            layouts=layouts,
            sufficient=n >= 6 and min(schemas.values()) >= 3 and min(layouts.values()) >= 1,
        )
    require(pop == audit["populations"], "Population arithmetic")
    require(
        audit["H1_attention"]
        == (pop["A"]["sufficient"] and pop["R"]["passes"] <= 2 and pop["J"]["passes"] <= 2)
        and audit["H2_query_generator"]
        == (pop["R"]["sufficient"] and pop["A"]["passes"] <= 2 and pop["J"]["passes"] <= 2),
        "Competing sufficient claims",
    )
    cpu = read(dest / "cpu-qualification.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["checks"] == 432
        and cpu["counts"]
        == dict(parent_tiny_forwards=36, diagnostic_tiny_forwards=24, failed_control_forwards=4)
        and len(cpu["rejected"]) == 12
        and cpu["source_hashes"] == d["source_hashes"]
        and sha(dest / "cpu-qualification.json") == d["cpu_qualification_sha256"],
        "Tiny qualification and stop-control evidence",
    )
    cost = read(dest / "cost.json")
    require(
        cost["checkpoint_full_forwards"] == 48
        and cost["counts"] == expected_counts
        and not cost["speedup_claimed"],
        "Computational cost",
    )
    provenance = read(root / "docs/figures/query-interaction-300/provenance.json")
    require(
        provenance["retrospective"]
        and provenance["all_eight_cases"]
        and provenance["script_sha256"] == sha(root / "research/boolean_coupling/describe300.py"),
        "Figure scope",
    )
    for n, h in provenance["inputs"].items():
        require(sha(root / n) == h, "Figure input")
    for n, h in provenance["outputs"].items():
        require(sha(root / "docs/figures/query-interaction-300" / n) == h, "Figure output")
    require(not audit["general_question_B_solved"], "General scope")
    print(
        dict(
            passed=True,
            experiment=300,
            retrospective=True,
            required_checks=audit["required_checks"],
            raw_failed=audit["raw_failed"],
            populations=pop,
            general_question_B_solved=False,
            omitted_arrays_replicated=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify(ROOT)
