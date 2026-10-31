"""Stdlib source/seal/decision arithmetic, without model or omitted-array replay."""

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
    from .verify296 import case_records, checks
except ImportError:
    from verify285 import boolean, close, number, read, relative, require, sha
    from verify286 import source_seal
    from verify288 import rows
    from verify293 import vstats
    from verify296 import case_records, checks

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/query-routes-297")
ARMS = ("S", "C", "B")


def manifest(root, dest, phase):
    m = read(dest / "release-manifest.json")
    require(m["experiment"] == 297 and m["phase"] == phase, "Release identity")
    for name, digest in m["files"].items():
        require(sha(relative(root, name)) == digest, "Release bytes " + name)
        require(
            not ({".context", "drafts"} & set(Path(name).parts))
            and Path(name).suffix not in (".npz", ".npy", ".pt", ".safetensors"),
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


def diagnostics(dest, name, audit):
    rr = rows(dest / name)
    require(
        len(rr) == audit["raw_logit_checks"] == 2 * audit["witness_kernels"],
        "Raw diagnostic denominator",
    )
    for r in rr:
        require(
            boolean(r, "passes") == (number(r, "error") <= 1e-4) and number(r, "bound") == 1e-4,
            "Original raw diagnostic decision",
        )
    require(
        sum(not boolean(r, "passes") for r in rr) == audit["raw_logit_failed"],
        "Raw failures remain failures",
    )


def population(native, field):
    out = {}
    for arm in ARMS:
        rr = [r for r in native if r["arm"] == arm]
        layouts = {
            s: sum(boolean(r, field) for r in rr if r["layout"] == s)
            for s in sorted({r["layout"] for r in rr})
        }
        n = sum(layouts.values())
        out[arm] = dict(
            passes=n,
            total=len(rr),
            layouts=layouts,
            sufficient=n >= 24 and min(layouts.values()) >= 6,
        )
    return out


def verify(root, phase):
    pre = root / DEST / "pre-native"
    manifest(root, pre, "pre-native")
    d, st, audit, seal = [
        read(pre / n)
        for n in ["design.json", "forecast-status.json", "pre-audit.json", "forecast-seal.json"]
    ]
    source_seal(root, d, "routes297")
    require(len(d["source_hashes"]) == 133, "133 immutable scientific sources")
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
        "Frozen native criterion",
    )
    require(
        d["effect"] == dict(norm=0.001, relative=0.25, cosine=0.95, margin=0.1, material=0.02),
        "Separate conditional target",
    )
    require(
        d["bounds"]
        == dict(map=2e-7, probability_sum=1e-6, composed=1e-4, head=0.005, raw_diagnostic=1e-4),
        "Numerical boundary",
    )
    require(
        d["guards_MiB"] == dict(free=8192, forecast=512, native=256, maps=64), "Storage boundary"
    )
    require(
        len(d["cases"]) == 32
        and st["ordinary_true"] == seal["ordinary_true"] == d["ordinary_true"] == 0,
        "Pre-native budget",
    )
    require(
        st["counts"]
        == dict(
            full_forwards=224,
            extra_full_QK=10800,
            extra_full_softmax=16200,
            extra_full_PV=5376,
            witness_extra_full_PV=288,
        ),
        "All forecast work",
    )
    require(
        audit["status"] == "passed"
        and not audit["scientific_outcomes_withheld"]
        and audit["design_sha256"] == sha(pre / "design.json")
        and audit["forecast_status_sha256"] == sha(pre / "forecast-status.json"),
        "Forecast numerical gate",
    )
    require(
        seal["status"] == "sealed"
        and seal["design_sha256"] == sha(pre / "design.json")
        and seal["status_sha256"] == sha(pre / "forecast-status.json")
        and seal["pre_audit_sha256"] == sha(pre / "pre-audit.json"),
        "Qualified global seal",
    )
    checks(pre, "pre-checks.csv", audit)
    require(audit["witness_kernels"] == 96, "Forecast witness coverage")
    diagnostics(pre, "pre-raw-logits.csv", audit)
    cpu = read(pre / "cpu-qualification.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["model_forwards"] == 36
        and cpu["required_checks"] == 2184
        and cpu["failed_checks"] == 0
        and len(cpu["rejected"]) == 9
        and cpu["gauge_invariant"]
        and cpu["source_hashes"] == d["source_hashes"],
        "Tiny qualification and corruption rejection",
    )
    require(
        sha(pre / "cpu-qualification.json") == d["cpu_qualification_sha256"],
        "CPU qualification binding",
    )
    catalog = set(read(root / "research/boolean_coupling/history297.json")["hashes"])
    seen = set()
    for c in d["cases"]:
        f, t, pos = c["false_ids"], c["true_ids"], c["position"]
        hs = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in [f, t]
        ]
        require(
            hs == c["input_hashes"]
            and not set(hs) & (catalog | seen)
            and len(f) == len(t)
            and [j for j, (x, y) in enumerate(zip(f, t, strict=True)) if x != y] == [pos]
            and f[pos] == 895
            and t[pos] == 830,
            "Fresh single-token input",
        )
        seen.update(hs)
    require(
        len(seen) == d["fresh_input_hashes"] == 64 and len(catalog) == d["historical_input_hashes"],
        "Available-catalog freshness",
    )
    fm = case_records(root, pre, st, d["cases"])
    require(st["raw_bytes"] <= 512 * 1024**2, "Forecast storage cap")
    result = dict(
        passed=True,
        experiment=297,
        phase=phase,
        pre_required_checks=audit["required_checks"],
        raw_pre_failures=audit["raw_logit_failed"],
        native_observed=False,
        omitted_arrays_replicated=False,
        general_question_B_solved=False,
    )
    if phase == "native":
        dest = root / DEST / "native-phase"
        manifest(root, dest, "native-phase")
        na, ns, proof = [
            read(dest / n) for n in ["audit.json", "native-status.json", "measurement-proof.json"]
        ]
        require(
            na["status"] == "passed"
            and not na["scientific_outcomes_withheld"]
            and na["native_status_sha256"] == sha(dest / "native-status.json"),
            "Qualified native gate",
        )
        require(
            ns["forecast_seal_sha256"] == proof["seal_sha256"] == sha(pre / "forecast-seal.json")
            and ns["proof_sha256"] == sha(dest / "measurement-proof.json")
            and proof["accepted_utc"] > seal["created_utc"]
            and proof["remote_head"] == proof["commit"],
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
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", proof["commit"], "HEAD"],
                cwd=root,
                capture_output=True,
            ).returncode
            == 0,
            "Pre-native seal ancestry",
        )
        require(
            ns["ordinary_true"] == 32
            and ns["counts"]
            == dict(
                full_forwards=64,
                extra_full_QK=1840,
                extra_full_softmax=2760,
                extra_full_PV=896,
                witness_extra_full_PV=144,
            )
            and ns["raw_bytes"] <= 256 * 1024**2,
            "Native work and storage",
        )
        nm = case_records(root, dest, ns, d["cases"])
        checks(dest, "checks.csv", na)
        require(na["witness_kernels"] == 144, "Combined witness coverage")
        diagnostics(dest, "raw-logits.csv", na)
        for key, m in nm.items():
            require(
                m["native_started_utc"] > proof["accepted_utc"]
                and m["forecast_arrays_sha256"] == fm[key]["arrays_sha256"],
                "Native ordering",
            )
            for arm in ARMS:
                require(
                    m["source_hashes"]["T"] == fm[key]["source_hashes"][arm],
                    "Native source invariance",
                )
        stats = read(dest / "vector-records.json")["statistics"]
        ix = {(r["key"], r["arm"], r["scope"], r["target"]): r for r in stats}
        require(len(stats) == len(ix) == 448, "All full/effect vector records")
        for r in stats:
            vstats(r)
        layer = read(dest / "layer-records.json")["statistics"]
        require(
            len(layer) == 32 * 28 * 3
            and len({(r["key"], r["layer"], r["arm"]) for r in layer}) == len(layer),
            "All layerwise signed policy records",
        )
        for r in layer:
            vstats(r)
        local = read(dest / "local-records.json")["statistics"]
        witnesses = [c for c in d["cases"] if c["group"]["reverse"] == c["group"]["target"] == 0]
        require(
            len(witnesses) == 8
            and len(local)
            == sum(len(c["false_ids"]) - c["position"] - 1 for c in witnesses) * 3 * 6,
            "All native common-parent records",
        )
        require(
            len({(r["key"], r["layer"], r["position"], r["arm"]) for r in local}) == len(local),
            "Unique local coverage",
        )
        for r in local:
            n, _, _ = vstats(r)
            require(r["target_eligible"] == (n >= 0.001), "Local norm guard preserves rows")
        fs = {r["key"]: r["scores"] for r in read(pre / "scores.json")["cases"]}
        ts = {r["key"]: r["scores"] for r in read(dest / "scores.json")["cases"]}
        native = rows(dest / "native.csv")
        require(
            len(native) == 96 and len({(r["key"], r["arm"]) for r in native}) == 96,
            "All native decisions",
        )
        for r in native:
            key, arm = r["key"], r["arm"]
            mt = ts[key]["T_scores"][1] - ts[key]["T_scores"][0]
            mc = fs[key][arm + "_scores"][1] - fs[key][arm + "_scores"][0]
            mb = fs[key]["B_scores"][1] - fs[key]["B_scores"][0]
            error, material = abs(mc - mt), abs(mt - mb)
            close(number(r, "margin_error"), error, "Actual finite margin")
            close(number(r, "query_margin_magnitude"), material, "Conditional effect materiality")
            ok, effect_ok = error <= 0.1, error <= 0.1 and material >= 0.02
            for scope in ["receiving", "verdict"]:
                n, re, co = vstats(ix[key, arm, scope, "native"])
                ok = ok and n >= 0.001 and re <= 0.05 and co >= 0.99
                close(number(r, scope + "_relative"), re, "Native vector metric")
                n, re, co = vstats(ix[key, arm, scope, "query_effect"])
                effect_ok = effect_ok and n >= 0.001 and re <= 0.25 and co >= 0.95
                close(number(r, "query_" + scope + "_relative"), re, "Conditional vector metric")
            require(
                boolean(r, "passes") == ok and boolean(r, "effect_passes") == effect_ok,
                "Frozen decisions",
            )
        pop, effect_pop = population(native, "passes"), population(native, "effect_passes")
        require(
            na["populations"] == pop
            and na["query_effect_populations"] == effect_pop
            and na["H1_source"]
            == (pop["S"]["sufficient"] and pop["C"]["passes"] <= 8 and pop["B"]["passes"] <= 8)
            and na["H2_other"]
            == (pop["C"]["sufficient"] and pop["S"]["passes"] <= 8 and pop["B"]["passes"] <= 8),
            "Both competing population claims",
        )
        cost = read(dest / "cost.json")
        require(
            cost["total_checkpoint_full_forwards"] == 288
            and cost["completed_forecast_counts"] == st["counts"]
            and cost["native_counts"] == ns["counts"]
            and cost["ordinary_native_true_calls"] == 32
            and cost["total_extra_full_QK"] == 12640
            and cost["total_extra_full_softmax"] == 18960
            and cost["total_extra_full_PV"] == 6704
            and not cost["wallclock_speedup_claimed"],
            "Complete computational work",
        )
        prov = read(root / "docs/figures/query-routes-297/provenance.json")
        require(
            prov["all_32_cases"]
            and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe297.py"),
            "Figure provenance",
        )
        for name, digest in prov["inputs"].items():
            require(sha(root / name) == digest, "Figure input")
        for name, digest in prov["outputs"].items():
            require(sha(root / "docs/figures/query-routes-297" / name) == digest, "Figure output")
        result.update(
            native_observed=True,
            native_required_checks=na["required_checks"],
            raw_combined_failures=na["raw_logit_failed"],
            populations=pop,
            H1_source=na["H1_source"],
            H2_other=na["H2_other"],
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["forecast", "native"], default="forecast")
    verify(ROOT, parser.parse_args().phase)
