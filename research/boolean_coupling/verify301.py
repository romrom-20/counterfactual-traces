"""Stdlib verification of sealed equivalent-alias routes; no activation replication."""

import argparse
import hashlib
import json
import statistics
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
DEST = Path("results/alias-source-routes-301")


def manifest(root, dest, phase):
    m = read(dest / "release-manifest.json")
    require(m["experiment"] == 301 and m["phase"] == phase, "Release identity")
    for n, h in m["files"].items():
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
        actual == {n for n in m["files"] if n.startswith(str(dest.relative_to(root)) + "/")},
        "Exact result inventory",
    )


def diagnostics(dest, name, audit):
    data = rows(dest / name)
    require(len(data) == audit["raw_checks"] == 2 * audit["witness_kernels"], "Raw coverage")
    for r in data:
        require(
            number(r, "bound") == 1e-4 and boolean(r, "passes") == (number(r, "error") <= 1e-4),
            "Original raw criterion",
        )
    require(sum(not boolean(r, "passes") for r in data) == audit["raw_failed"], "Failures retained")


def populations(effects):
    out = {}
    for account in ("source", "remaining", "random"):
        rr = [r for r in effects if r["account"] == account]
        layout = {
            s: sum(boolean(r, "passes") for r in rr if r["layout"] == s)
            for s in sorted({r["layout"] for r in rr})
        }
        order = {
            str(s): sum(boolean(r, "passes") for r in rr if int(r["mention_order"]) == s)
            for s in (0, 1)
        }
        n = sum(layout.values())
        out[account] = dict(
            passes=n,
            total=len(rr),
            layouts=layout,
            mention_orders=order,
            sufficient=n >= 24 and min(layout.values()) >= 6 and min(order.values()) >= 12,
        )
    return out


def verify(root, phase):
    pre = root / DEST / "pre-native"
    manifest(root, pre, "pre-native")
    d, st, a, seal = [
        read(pre / n)
        for n in ("design.json", "forecast-status.json", "pre-audit.json", "forecast-seal.json")
    ]
    source_seal(root, d, "alias301")
    require(len(d["source_hashes"]) == 163, "163 immutable scientific sources")
    require(
        d["primary"]
        == dict(
            norm=0.001,
            relative=0.25,
            cosine=0.95,
            material=0.25,
            margin_abs=0.1,
            margin_fraction=0.25,
            population=24,
            per_layout=6,
            per_mention_order=12,
            control_max=8,
        ),
        "Frozen alias-contrast criterion",
    )
    require(
        d["native"] == dict(norm=0.001, relative=0.05, cosine=0.99, margin=0.1),
        "Separate native criterion",
    )
    require(
        d["bounds"]
        == dict(
            composed=1e-4,
            projection_abs=1e-3,
            projection_relative=1e-5,
            head=1e-4,
            raw=1e-4,
            random_norm_relative=1e-4,
            random_norm_floor=0.001,
        ),
        "Numerical bounds",
    )
    require(
        d["guards_MiB"] == dict(free=8192, forecast=512, native=256, maps=64), "Resource bounds"
    )
    require(
        st["ordinary_true"] == seal["ordinary_true"] == d["ordinary_true"] == 0
        and len(d["cases"]) == 32,
        "Pre-native paired scope",
    )
    fc = dict(
        full_forwards=320,
        extra_full_QK_softmax_PV_each=7168,
        witness_extra_full_QK=192,
        extra_full_final_norm=64,
        extra_full_vocabulary_head=64,
    )
    require(st["counts"] == fc and st["raw_bytes"] <= 512 * 1024**2, "Forecast work and cap")
    require(
        a["status"] == "passed"
        and not a["scientific_outcomes_withheld"]
        and a["design_sha256"] == sha(pre / "design.json")
        and a["forecast_status_sha256"] == sha(pre / "forecast-status.json"),
        "Pre-native gate",
    )
    require(
        seal["status"] == "sealed"
        and seal["design_sha256"] == sha(pre / "design.json")
        and seal["status_sha256"] == sha(pre / "forecast-status.json")
        and seal["pre_audit_sha256"] == sha(pre / "pre-audit.json"),
        "Qualified global seal",
    )
    checks(pre, "pre-checks.csv", a)
    diagnostics(pre, "pre-raw-coordinates.csv", a)
    require(a["witness_kernels"] == 192, "Forecast witness scope")
    cpu = read(pre / "cpu-qualification.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["required_checks"] == 2424
        and cpu["full_forwards"] == 56
        and len(cpu["rejected"]) == 13
        and cpu["known_causal_tables"] == ["source", "remaining", "joint"]
        and cpu["source_hashes"] == d["source_hashes"]
        and sha(pre / "cpu-qualification.json") == d["cpu_qualification_sha256"],
        "Tiny qualification binding",
    )
    catalog = set(read(root / "research/boolean_coupling/history301.json")["hashes"])
    seen = set()
    pairs = []
    flat = read(root / "research/boolean_coupling/cases301-flat.json")
    require(len(flat) == 64, "Future catalog exposes all paired inputs")
    flat_ix = {(r["key"]): r for r in flat}
    for c in d["cases"]:
        for alias in ("A", "B"):
            item = c["aliases"][alias]
            f, t = item["false_ids"], item["true_ids"]
            j = c["position"]
            hs = [
                hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                for ids in (f, t)
            ]
            require(
                hs == item["input_hashes"]
                and not set(hs) & (catalog | seen)
                and len(f) == len(t)
                and [p for p, (x, y) in enumerate(zip(f, t, strict=True)) if x != y] == [j]
                and f[j] == 895
                and t[j] == 830,
                "Fresh single-Boolean edit",
            )
            require(
                flat_ix[c["key"] + "-" + alias]["false_ids"] == f
                and flat_ix[c["key"] + "-" + alias]["true_ids"] == t,
                "Flat input registry binding",
            )
            seen.update(hs)
        f, g = [c["aliases"][alias]["false_ids"] for alias in ("A", "B")]
        diff = [j for j, (x, y) in enumerate(zip(f, g, strict=True)) if x != y]
        require(
            len(diff) == 2
            and diff == c["alias_changed"]
            and min(diff) == c["alias_start"]
            and max(diff) < c["position"]
            and f[c["position"] :] == g[c["position"] :],
            "Alias-only length/position control",
        )
        pairs.append(c["key"])
    require(
        len(seen) == d["fresh_input_hashes"] == 128
        and len(catalog) == d["historical_input_hashes"],
        "Catalog freshness",
    )
    fm = case_records(root, pre, st, d["cases"])
    for m in fm.values():
        require(
            m["ordinary_true"] == 0
            and m["source_hashes"]["AB_F"] == m["source_hashes"]["B_F"]
            and m["source_hashes"]["BA_F"] == m["source_hashes"]["A_F"],
            "False source-generator identity",
        )
    result = dict(
        passed=True,
        experiment=301,
        phase=phase,
        pre_required_checks=a["required_checks"],
        raw_pre_failed=a["raw_failed"],
        native_observed=False,
        omitted_arrays_replicated=False,
        general_question_B_solved=False,
    )
    if phase == "native":
        dest = root / DEST / "native-phase"
        manifest(root, dest, "native-phase")
        na, ns, proof = [
            read(dest / n) for n in ("audit.json", "native-status.json", "measurement-proof.json")
        ]
        require(
            na["status"] == "passed"
            and not na["scientific_outcomes_withheld"]
            and na["native_status_sha256"] == sha(dest / "native-status.json"),
            "Native numerical gate",
        )
        require(
            ns["forecast_seal_sha256"] == proof["seal_sha256"] == sha(pre / "forecast-seal.json")
            and ns["proof_sha256"] == sha(dest / "measurement-proof.json")
            and proof["accepted_utc"] > seal["created_utc"]
            and proof["remote_head"] == proof["commit"],
            "Public pre-native receipt",
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
            "Seal ancestry",
        )
        nc = dict(full_forwards=128, extra_full_QK_softmax_PV_each=1792, witness_extra_full_QK=96)
        require(
            ns["counts"] == nc and ns["ordinary_true"] == 64 and ns["raw_bytes"] <= 256 * 1024**2,
            "Native paired work",
        )
        nm = case_records(root, dest, ns, d["cases"])
        checks(dest, "checks.csv", na)
        diagnostics(dest, "raw-coordinates.csv", na)
        require(na["witness_kernels"] == 288, "Combined witness coverage")
        for key, m in nm.items():
            require(
                m["ordinary_true"] == 2
                and m["native_started_utc"] > proof["accepted_utc"]
                and m["forecast_arrays_sha256"] == fm[key]["arrays_sha256"],
                "Native endpoint ordering",
            )
            require(
                m["source_hashes"]["B_T"] == fm[key]["source_hashes"]["AB_T"]
                and m["source_hashes"]["A_T"] == fm[key]["source_hashes"]["BA_T"],
                "True source-generator identity",
            )
        v = read(dest / "vector-records.json")["statistics"]
        ix = {(r["key"], r["arm"], r["scope"]): r for r in v}
        require(len(v) == len(ix) == 512, "Complete vector coverage")
        for r in v:
            vstats(r)
        fs = {r["key"]: r["scores"] for r in read(pre / "scores.json")["cases"]}
        ts = {r["key"]: r["scores"] for r in read(dest / "scores.json")["cases"]}
        coupling = rows(dest / "couplings.csv")
        require(len(coupling) == 160, "All coupling corners")
        ci = {(r["key"], r["corner"]): number(r, "value") for r in coupling}
        require(len(ci) == 160, "Unique coupling corners")
        mi = {}
        for key in pairs:

            def margin(bank, arm):
                ss = bank[key][arm + "_scores"]
                return ss[1] - ss[0]

            m = dict(
                AA=margin(ts, "A_T") - margin(fs, "A_F"),
                BB=margin(ts, "B_T") - margin(fs, "B_F"),
                **{h: margin(fs, h + "_T") - margin(fs, h + "_F") for h in ("AB", "BA", "RA")},
            )
            for c, value in m.items():
                close(ci[key, c], value, "Finite Boolean coupling")
            mi[key] = dict(
                target=m["BB"] - m["AA"],
                source_A=m["AB"] - m["AA"],
                source_B=m["BB"] - m["BA"],
                remaining_A=m["BA"] - m["AA"],
                remaining_B=m["BB"] - m["AB"],
                random_A=m["RA"] - m["AA"],
            )
        effects = rows(dest / "effects.csv")
        require(
            len(effects) == 96 and len({(r["key"], r["account"]) for r in effects}) == 96,
            "All account decisions",
        )
        for r in effects:
            key, account = r["key"], r["account"]
            t = mi[key]["target"]
            bound = min(0.1, 0.25 * abs(t))
            arms = {
                "source": ("source_A", "source_B"),
                "remaining": ("remaining_A", "remaining_B"),
                "random": ("random_A",),
            }[account]
            err = max(abs(mi[key][a] - t) for a in arms)
            ok = abs(t) >= 0.25 and err <= bound
            close(number(r, "native_alias_margin_contrast"), t, "Native alias contrast")
            close(number(r, "margin_bound"), bound, "Contrast bound")
            close(number(r, "margin_error"), err, "Both-reference margin error")
            for scope in ("receiving", "verdict"):
                stats = [vstats(ix[key, a, scope]) for a in arms]
                ok = ok and all(n >= 0.001 and re <= 0.25 and co >= 0.95 for n, re, co in stats)
                close(
                    number(r, scope + "_relative"),
                    max(s[1] for s in stats),
                    "Both-reference vector error",
                )
            require(
                boolean(r, "passes") == ok and boolean(r, "material") == (abs(t) >= 0.25),
                "Frozen complete criterion",
            )
        pop = populations(effects)
        require(
            na["populations"] == pop
            and na["H1_source"]
            == (
                pop["source"]["sufficient"]
                and pop["remaining"]["passes"] <= 8
                and pop["random"]["passes"] <= 8
            )
            and na["H2_remaining"]
            == (
                pop["remaining"]["sufficient"]
                and pop["source"]["passes"] <= 8
                and pop["random"]["passes"] <= 8
            ),
            "Competing population claims",
        )
        require(
            na["material_alias_cases"]
            == sum(boolean(r, "material") for r in effects if r["account"] == "source"),
            "All low-effect cases retained",
        )
        native = rows(dest / "native-predictions.csv")
        require(
            len(native) == 64 and len({(r["key"], r["alias"]) for r in native}) == 64,
            "Both native forecasts per pair",
        )
        for r in native:
            key, alias = r["key"], r["alias"]
            p = fs[key]["P_" + alias + "_scores"]
            t = ts[key][alias + "_T_scores"]
            err = abs((p[1] - p[0]) - (t[1] - t[0]))
            ok = err <= 0.1
            close(number(r, "margin_error"), err, "Finite transfer margin")
            for scope in ("receiving", "verdict"):
                norm, re, co = vstats(ix[key, "P_" + alias, scope])
                close(number(r, scope + "_relative"), re, "Transfer displacement")
                ok = ok and norm >= 0.001 and re <= 0.05 and co >= 0.99
            require(boolean(r, "passes") == ok, "Separate native transfer criterion")
        require(
            na["native_individual_passes"] == sum(boolean(r, "passes") for r in native)
            and na["native_pair_passes"]
            == sum(all(boolean(r, "passes") for r in native if r["key"] == key) for key in pairs),
            "Separate pair success, no primary rescue",
        )
        summary = read(dest / "descriptive-summary.json")
        require(summary["descriptive"] and summary["all_32_pairs"], "Descriptive scope")
        for account in ("source", "remaining", "random"):
            rr = [r for r in effects if r["account"] == account]
            for field, value in summary["medians"][account].items():
                close(value, statistics.median(number(r, field) for r in rr), "All-case median")
        cost = read(dest / "cost.json")
        require(
            cost["total_checkpoint_full_forwards"] == 448
            and cost["completed_forecast_counts"] == fc
            and cost["native_counts"] == nc
            and cost["ordinary_native_true_calls"] == 64
            and cost["total_extra_full_QK_softmax_PV_each"] == 8960
            and cost["total_witness_full_QK"] == 288
            and cost["extra_final_norms"] == 64
            and cost["extra_full_vocabulary_heads"] == 64
            and not cost["wallclock_speedup_claimed"],
            "Full budget",
        )
        prov = read(root / "docs/figures/alias-source-routes-301/provenance.json")
        require(
            prov["all_32_pairs"]
            and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe301.py"),
            "Figure provenance",
        )
        for n, h in prov["inputs"].items():
            require(sha(root / n) == h, "Figure input")
        for n, h in prov["outputs"].items():
            require(sha(root / "docs/figures/alias-source-routes-301" / n) == h, "Figure output")
        result.update(
            native_observed=True,
            native_required_checks=na["required_checks"],
            raw_combined_failed=na["raw_failed"],
            populations=pop,
            H1_source=na["H1_source"],
            H2_remaining=na["H2_remaining"],
            material_alias_cases=na["material_alias_cases"],
            native_pair_passes=na["native_pair_passes"],
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["forecast", "native"], default="forecast")
    verify(ROOT, parser.parse_args().phase)
