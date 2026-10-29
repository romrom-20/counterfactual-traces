"""Stdlib296 receipts/outcome arithmetic; no independent model replication."""

import argparse
import hashlib
import json
import subprocess
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
DEST = Path("results/query-geometry-296")
ARMS = ("A", "D", "B", "R")


def manifest(root, dest, phase):
    m = read(dest / "release-manifest.json")
    require(m["experiment"] == 296 and m["phase"] == phase, "Release identity")
    for n, h in m["files"].items():
        require(sha(relative(root, n)) == h, "Release bytes: " + n)
        require(
            not ({".context", "drafts"} & set(Path(n).parts))
            and Path(n).suffix not in (".npz", ".npy", ".pt", ".safetensors"),
            "Public scope",
        )
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    require(
        actual == {n for n in m["files"] if n.startswith(str(dest.relative_to(root)) + "/")},
        "Exact result inventory",
    )


def checks(dest, name, a):
    rr = rows(dest / name)
    require(
        len(rr) == a["required_checks"] and a["failed_checks"] == 0, "Required numeric denominator"
    )
    for r in rr:
        require(
            boolean(r, "passes") and 0 <= number(r, "error") <= number(r, "bound"),
            "Required numeric decision",
        )
    for n, h in a["csv_sha256"].items():
        p = dest / n
        if not p.exists():
            p = dest.parent / "pre-native" / n
        require(sha(p) == h, "Audit table bytes")
    return rr


def case_records(root, dest, st, expected):
    require(
        [e["key"] for e in st["cases"]] == [c["key"] for c in expected], "Ordered case coverage"
    )
    inv = read(dest / "omitted-array-inventory.json")
    require(
        sum(x["bytes"] for x in inv["arrays"]) == st["raw_bytes"] == inv["raw_bytes"],
        "Actual storage",
    )
    bypath = {x["path"]: x for x in inv["arrays"]}
    folder = "forecasts" if dest.name == "pre-native" else "observations"
    for e in st["cases"]:
        p = dest / "cases" / (e["key"] + ".json")
        m = read(p)
        require(
            sha(p) == e["metadata_sha256"] and m["arrays_sha256"] == e["arrays_sha256"],
            "Case metadata",
        )
        v = bypath[folder + "/" + e["key"] + ".npz"]
        require(
            v["sha256"] == e["arrays_sha256"] and v["bytes"] == e["bytes"],
            "Omitted archive binding",
        )
        if e["witness_sha256"]:
            wp = (
                ("witnesses-forecast" if folder == "forecasts" else "witnesses-native")
                + "/"
                + e["key"]
                + ".npz"
            )
            require(
                bypath[wp]["sha256"] == e["witness_sha256"]
                and bypath[wp]["bytes"] == e["witness_bytes"],
                "Witness binding",
            )
        require(
            all(r["passes"] and r["error"] <= r["bound"] for r in m["checks"]),
            "Producer installation/zero/prefix checks",
        )
    return {e["key"]: read(dest / "cases" / (e["key"] + ".json")) for e in st["cases"]}


def verify(root, phase):
    pre = root / DEST / "pre-native"
    manifest(root, pre, "pre-native")
    d, st, a, seal = (
        read(pre / n)
        for n in ["design.json", "forecast-status.json", "pre-audit.json", "forecast-seal.json"]
    )
    source_seal(root, d, "query296")
    require(len(d["source_hashes"]) == 125, "125 immutable scientific sources")
    require(
        d["primary"]
        == dict(
            norm=0.001,
            relative=0.05,
            cosine=0.99,
            margin=0.1,
            population=24,
            per_layout=6,
            control_max=8,
        ),
        "Frozen criteria",
    )
    require(
        d["bounds"] == dict(map=1e-5, composed=1e-4, head=0.005, raw_diagnostic=1e-4)
        and d["guards_MiB"] == [1024, 512, 256, 128],
        "Numerical/resource boundary",
    )
    require(
        len(d["cases"]) == 32
        and st["ordinary_true"] == seal["ordinary_true"] == d["ordinary_true"] == 0
        and st["counts"]
        == dict(full_forwards=256, extra_full_QK_softmax_PV_each=6272, witness_extra_QK=120),
        "Forecast budget",
    )
    require(
        a["status"] == "passed"
        and not a["scientific_outcomes_withheld"]
        and seal["status"] == "sealed"
        and seal["design_sha256"] == sha(pre / "design.json")
        and seal["status_sha256"] == sha(pre / "forecast-status.json")
        and seal["pre_audit_sha256"] == sha(pre / "pre-audit.json")
        and a["design_sha256"] == sha(pre / "design.json")
        and a["forecast_status_sha256"] == sha(pre / "forecast-status.json"),
        "Global qualified seal",
    )
    checks(pre, "pre-checks.csv", a)
    raw = rows(pre / "pre-raw-logits.csv")
    require(len(raw) == a["raw_logit_checks"] == a["witness_kernels"] == 120, "Witness denominator")
    for r in raw:
        require(
            boolean(r, "passes") == (number(r, "error") <= 1e-4) and number(r, "bound") == 1e-4,
            "Retained raw diagnostic",
        )
    require(
        sum(not boolean(r, "passes") for r in raw) == a["raw_logit_failed"],
        "Raw failures remain failures",
    )
    catalog = set(read(root / "research/boolean_coupling/history296.json")["hashes"])
    seen = set()
    for c in d["cases"]:
        f, t = c["false_ids"], c["true_ids"]
        j = c["position"]
        hs = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in [f, t]
        ]
        require(
            hs == c["input_hashes"]
            and not set(hs) & (seen | catalog)
            and len(f) == len(t)
            and [k for k, (x, y) in enumerate(zip(f, t, strict=True)) if x != y] == [j]
            and f[j] == 895
            and t[j] == 830,
            "Exact unseen single-token edits",
        )
        seen.update(hs)
    require(len(seen) == 64 and len(catalog) == d["historical_input_hashes"], "Freshness scope")
    fm = case_records(root, pre, st, d["cases"])
    require(st["raw_bytes"] <= 256 * 1024**2, "Forecast storage guard")
    q = read(pre / "execution-qualification.json")
    require(
        q["method_unchanged"]
        and q["progress_paths_only"]
        and q["original_completed_forecasts"] == 16,
        "Progress-only correction",
    )
    for n, h in q["source_hashes"].items():
        require(
            sha(root / n) == h
            and hashlib.sha256(
                subprocess.check_output(["git", "show", q["source_commit"] + ":" + n], cwd=root)
            ).hexdigest()
            == h,
            "Frozen execution qualifier",
        )
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", q["source_commit"], "HEAD"],
            cwd=root,
            capture_output=True,
        ).returncode
        == 0,
        "Qualifier ancestry",
    )
    replay = read(pre / "exact-original-replay.json")
    require(
        replay["passed"]
        and replay["arrays_witness_metadata_exact"]
        and replay["before_third_case"]
        and replay["keys"] == [c["key"] for c in d["cases"][:2]],
        "Exact original replay before new cases",
    )
    old = read(pre / "original-stop/summary.json")
    require(
        old["saved_cases"] == 2
        and old["ordinary_true"] == 0
        and old["completed_full_forwards"] == 16
        and old["raw_failure_sha256"] == q["original_failure_sha256"]
        and sha(pre / "original-stop/design.json") == q["original_design_sha256"],
        "Preserved original stop",
    )
    for e, ne in zip(old["cases"], st["cases"][:2], strict=True):
        for field in ["key", "arrays_sha256", "metadata_sha256", "witness_sha256"]:
            require(e[field] == ne[field], "Original array/metadata/witness replay")
    cpu = read(pre / "cpu-qualification.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["required_checks"] == 1132
        and cpu["failed_checks"] == 0
        and len(cpu["rejected"]) == 8
        and cpu["source_hashes"] == d["source_hashes"]
        and cpu["zero_query_bypass"]
        and cpu["scientific_outcomes_not_required"],
        "Tiny qualification",
    )
    result = dict(
        passed=True,
        experiment=296,
        phase=phase,
        pre_required_checks=a["required_checks"],
        raw_pre_failures=a["raw_logit_failed"],
        native_observed=False,
        omitted_arrays_replicated=False,
        general_question_B_solved=False,
    )
    if phase == "native":
        nd = root / DEST / "native-phase"
        manifest(root, nd, "native-phase")
        na = read(nd / "audit.json")
        ns = read(nd / "native-status.json")
        proof = read(nd / "measurement-proof.json")
        require(
            na["status"] == "passed"
            and not na["scientific_outcomes_withheld"]
            and na["native_status_sha256"] == sha(nd / "native-status.json"),
            "Qualified native gate",
        )
        require(
            ns["forecast_seal_sha256"] == proof["seal_sha256"] == sha(pre / "forecast-seal.json")
            and ns["proof_sha256"] == sha(nd / "measurement-proof.json")
            and proof["accepted_utc"] > seal["created_utc"]
            and proof["commit"] == proof["remote_head"],
            "Pre-native public receipt",
        )
        require(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    proof["commit"] + ":" + str(DEST / "pre-native/forecast-seal.json"),
                ],
                cwd=root,
            )
            == (pre / "forecast-seal.json").read_bytes(),
            "Public seal bytes",
        )
        require(
            ns["ordinary_true"] == 32
            and ns["counts"]
            == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=48)
            and ns["raw_bytes"] <= 128 * 1024**2,
            "Native budget",
        )
        nm = case_records(root, nd, ns, d["cases"])
        checks(nd, "checks.csv", na)
        nr = rows(nd / "raw-logits.csv")
        require(
            len(nr) == na["raw_logit_checks"] == 168
            and sum(not boolean(r, "passes") for r in nr) == na["raw_logit_failed"],
            "Combined raw denominator",
        )
        for r in nr:
            require(
                boolean(r, "passes") == (number(r, "error") <= 1e-4),
                "Raw native qualification remains scoped",
            )
        for key, m in nm.items():
            require(
                m["native_started_utc"] > proof["accepted_utc"]
                and m["forecast_arrays_sha256"] == fm[key]["arrays_sha256"],
                "Native ordering/input binding",
            )
            for arm in ARMS:
                require(
                    m["source_hashes"]["T"] == fm[key]["source_hashes"][arm], "Native source exact"
                )
        stats = read(nd / "vector-records.json")["statistics"]
        ix = {(r["key"], r["arm"], r["scope"]): r for r in stats}
        require(len(stats) == len(ix) == 256, "All vector records")
        for r in stats:
            vstats(r)
        fs = {r["key"]: r["scores"] for r in read(pre / "scores.json")["cases"]}
        ts = {r["key"]: r["scores"] for r in read(nd / "scores.json")["cases"]}
        native = rows(nd / "native.csv")
        require(len(native) == 128, "All native rows")
        for r in native:
            key, arm = r["key"], r["arm"]
            error = abs(
                (fs[key][arm + "_scores"][1] - fs[key][arm + "_scores"][0])
                - (ts[key]["T_scores"][1] - ts[key]["T_scores"][0])
            )
            close(number(r, "margin_error"), error, "Actual finite margin")
            ok = error <= 0.1
            for scope in ["receiving", "verdict"]:
                n, re, co = vstats(ix[key, arm, scope])
                ok = ok and n >= 0.001 and re <= 0.05 and co >= 0.99
                close(number(r, scope + "_relative"), re, "Native vector metric")
            require(boolean(r, "passes") == ok, "Frozen native decision")
        pop = {}
        for arm in ARMS:
            rr = [r for r in native if r["arm"] == arm]
            layouts = {
                s: sum(boolean(r, "passes") for r in rr if r["layout"] == s)
                for s in sorted({r["layout"] for r in rr})
            }
            n = sum(layouts.values())
            pop[arm] = dict(
                passes=n,
                total=32,
                layouts=layouts,
                sufficient=n >= 24 and min(layouts.values()) >= 6,
            )
        require(
            na["populations"] == pop
            and na["H1_magnitude"]
            == (pop["A"]["sufficient"] and pop["D"]["passes"] <= 8 and pop["R"]["passes"] <= 8)
            and na["H2_direction"]
            == (pop["D"]["sufficient"] and pop["A"]["passes"] <= 8 and pop["R"]["passes"] <= 8),
            "Preregistered competing population claims",
        )
        cost = read(nd / "cost.json")
        require(
            cost["total_checkpoint_full_forwards"] == 336
            and cost["original_stopped_full_forwards"] == 16
            and cost["completed_forecast_counts"] == st["counts"]
            and cost["native_counts"] == ns["counts"]
            and cost["ordinary_native_true_calls"] == 32
            and not cost["wallclock_speedup_claimed"],
            "Actual work accounting",
        )
        require(
            cost["original_stopped_counts"]
            == dict(full_forwards=16, extra_full_QK_softmax_PV_each=392, witness_extra_QK=15)
            and cost["total_extra_QK_softmax_PV_each"] == 7560
            and cost["total_witness_extra_QK"] == 183,
            "All stopped/replayed attention work",
        )
        prov = read(root / "docs/figures/query-geometry-296/provenance.json")
        require(
            prov["all_32_cases"]
            and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe296.py"),
            "Figure provenance",
        )
        for n, h in prov["inputs"].items():
            require(sha(root / n) == h, "Figure input")
        for n, h in prov["outputs"].items():
            require(sha(root / "docs/figures/query-geometry-296" / n) == h, "Figure output")
        result.update(
            native_observed=True,
            native_required_checks=na["required_checks"],
            raw_combined_failures=na["raw_logit_failed"],
            populations=pop,
            H1_magnitude=na["H1_magnitude"],
            H2_direction=na["H2_direction"],
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["forecast", "native"], default="forecast")
    verify(ROOT, p.parse_args().phase)
