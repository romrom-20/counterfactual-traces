"""Stdlib293 source/record verification; no checkpoint or omitted-array rerun."""

import ast
import hashlib
import json
from pathlib import Path

try:
    from .verify285 import boolean, close, number, read, relative, require, sha
    from .verify286 import margin, source_seal
    from .verify287 import vector
    from .verify288 import rows
except ImportError:
    from verify285 import boolean, close, number, read, relative, require, sha
    from verify286 import margin, source_seal
    from verify287 import vector
    from verify288 import rows

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/response-subspace-293")
ARMS = ("P", "L", "F", "E", "R")


def vstats(v):
    p, t, d, e = (
        v[k]
        for k in (
            "prediction_squared_norm",
            "target_squared_norm",
            "prediction_target_dot",
            "error_squared_norm",
        )
    )
    require(min(p, t, e) >= 0, "Squared norms")
    import math

    require(math.isclose(e, p + t - 2 * d, rel_tol=1e-10, abs_tol=1e-9), "Vector error identity")
    n = t**0.5
    rel = e**0.5 / max(n, 1e-30)
    cos = d / max((p * t) ** 0.5, 1e-30)
    for k, x in [
        ("norm", n),
        ("relative", rel),
        ("cosine", cos),
        ("signed_target_fraction", d / max(t, 1e-30)),
    ]:
        close(v[k], x, "Vector statistic")
    return n, rel, cos


def verify(root):
    dest = root / DEST
    release = read(dest / "release-manifest.json")
    require(release["experiment"] == 293 and release["retrospective"], "Identity")
    for name, h in release["files"].items():
        require(sha(relative(root, name)) == h, "Release bytes " + name)
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
        actual == {n for n in release["files"] if n.startswith(str(DEST) + "/")},
        "Exact result inventory",
    )
    d, st = (read(dest / (n + ".json")) for n in ("design", "status"))
    aold = read(dest / "invalid-original/audit.json")
    a = read(dest / "audit.json")
    cd = read(dest / "namespace-qualification/design.json")
    require(
        not a["original_scientific_decisions_valid"] and cd["model_calls"] == 0,
        "Original science invalid; read-only correction",
    )
    require(release["namespace_sources"] == cd["source_hashes"], "Namespace source map")
    for name, h in cd["source_hashes"].items():
        require(sha(root / name) == h, "Correction source bytes")
    for name, h in cd["parent_hashes"].items():
        path = dest / ("invalid-original/audit.json" if name == "audit.json" else name)
        require(sha(path) == h, "Correction parent binding")
    source_seal(root, d, "subspace293")
    source_seal(root, cd, "namespace293c")
    require(
        len(d["source_hashes"]) == 107 and release["scientific_sources"] == d["source_hashes"],
        "107 immutable sources",
    )
    require(
        d["retrospective"]
        and not d["synthetic"]
        and d["ordinary_checkpoint_true"] == 0
        and d["guards_MiB"] == [512, 384, 384],
        "Qualified retrospective design",
    )
    require(
        d["primary"]
        == dict(
            rank=32,
            power_steps=2,
            native_relative=0.05,
            native_cosine=0.99,
            margin=0.1,
            pair_relative=0.1,
            pair_cosine=0.995,
            norm=0.001,
            population=6,
            per_layout=1,
            pairs=3,
        ),
        "Frozen criteria",
    )
    require(
        d["bounds"]
        == dict(
            orthogonal=1e-10,
            closure=1e-10,
            replay=0.0,
            primitive_abs=1e-4,
            primitive_rel=1e-5,
            FD_abs=1e-6,
            FD_rel=1e-5,
            head=0.005,
            range=1e-3,
        ),
        "Frozen numerical bounds",
    )
    old = root / "results/joint-gate-292"
    for name, phase in [
        ("design", "pre-native"),
        ("forecast-status", "pre-native"),
        ("native-status", "native-phase"),
        ("native-audit", "native-phase"),
    ]:
        require(
            d["reference_hashes"][name + ".json"] == sha(old / phase / (name + ".json")),
            "292 reference",
        )
    expected = [
        c
        for c in read(old / "pre-native/design.json")["cases"]
        if c["group"]["reverse"] == c["group"]["target"] == 0
    ]
    require(d["cases"] == expected and len(expected) == 8, "Eight exact old cases")
    require(
        st["status"] == "complete"
        and a["status"] == "passed"
        and a["failed_checks"] == 0
        and not a["general_question_B_solved"]
        and not aold["learned_circuit_identified"],
        "Qualified scope",
    )
    require(
        aold["design_sha256"] == sha(dest / "design.json")
        and aold["status_sha256"] == sha(dest / "status.json"),
        "Audit inputs",
    )
    proof = read(dest / "resource-qualification.json")
    require(
        proof["method_unchanged"]
        and proof["qualified_total_cap_MiB"] == 384
        and proof["original_total_cap_MiB"] == 256
        and proof["initial_and_percase_free_MiB"] == [512, 384]
        and len(proof["replayed_arrays"]) == 7
        and set(proof["replayed_arrays"].values()) == {237},
        "Resource-only continuation and exact replay",
    )
    require(
        proof["executor_sha256"] == sha(root / "research/boolean_coupling/resource293b.py")
        and proof["protocol_sha256"]
        == sha(root / "docs/experiments/293b-resource-continuation.md"),
        "Execution bytes",
    )
    for name, replacements in [
        (
            "run",
            [
                ('d["guards_MiB"] == [512, 384, 256]', 'd["guards_MiB"] == [512, 384, 384]'),
                ("guards_MiB=[512, 384, 256]", "guards_MiB=[512, 384, 384]"),
                ("total <= 256 * 1024**2", "total <= 384 * 1024**2"),
            ],
        ),
        ("audit", [("<= 256 * 1024**2", "<= 384 * 1024**2")]),
    ]:
        p = root / "research/boolean_coupling" / (name + "293.py")
        s = p.read_text()
        t = s
        for before, after in replacements:
            require(t.count(before) == 1, "Exact resource edit")
            t = t.replace(before, after)
        x, y = ast.parse(s), ast.parse(t)
        xc = [n for n in ast.walk(x) if isinstance(n, ast.Constant)]
        yc = [n for n in ast.walk(y) if isinstance(n, ast.Constant)]
        diff = [(i.value, j.value) for i, j in zip(xc, yc, strict=True) if i.value != j.value]
        require(diff == [(256, 384)] * len(replacements), "Only resource literals change")
        for i, j in zip(xc, yc, strict=True):
            j.value = i.value
        require(
            ast.dump(x, include_attributes=False) == ast.dump(y, include_attributes=False),
            "Computation AST unchanged",
        )
        require(
            proof["compiled_sources"][name]
            == dict(
                original=sha(p),
                executed=hashlib.sha256(t.encode()).hexdigest(),
                resource_literals=len(replacements),
            ),
            "Compiled resource source seal",
        )
    for name, h in aold["csv_sha256"].items():
        path = dest / ("invalid-original/" + name if name in ("native.csv", "pairs.csv") else name)
        require(sha(path) == h, "Original table retained")
    for name, h in a["csv_sha256"].items():
        path = dest / ("namespace-qualification/checks.csv" if name == "checks.csv" else name)
        require(sha(path) == h, "Qualified comparison table")
    checks = rows(dest / "checks.csv")
    extra = rows(dest / "namespace-qualification/checks.csv")
    require(len(extra) == a["new_required_checks"] == 8, "New baseline checks")
    require(len(checks) == aold["required_checks"], "Numeric denominator")
    for r in checks + extra:
        require(
            0 <= number(r, "error") <= number(r, "bound") and boolean(r, "passes"),
            "Required numerical decision",
        )
    inv = read(dest / "omitted-array-inventory.json")
    require(
        inv["case_bytes"]
        == st["raw_bytes"]
        == sum(e["bytes"] for e in st["cases"])
        <= 384 * 1024**2,
        "Actual archive budget",
    )
    require([e["key"] for e in st["cases"]] == [c["key"] for c in expected], "Ordered cases")
    counts = {}
    for e, v in zip(st["cases"], inv["arrays"], strict=True):
        p = dest / "cases" / (e["key"] + ".json")
        require(
            sha(p) == e["metadata_sha256"]
            and v["sha256"] == e["arrays_sha256"]
            and v["bytes"] == e["bytes"]
            and len(v["arrays"]) == 237,
            "Omitted arrays and metadata",
        )
        m = read(p)
        require(
            m["ordinary_checkpoint_true"] == 0
            and set(m["arms"]) == {*ARMS, *(n + "0" for n in ARMS)},
            "All arms and zero controls",
        )
        for arm, meta in m["arms"].items():
            require(
                meta["no_current_native_SiLU"]
                and meta["engine"]["current_affected_finite_SiLU_coordinates"] == 0
                and meta["rank"] == 32,
                "No native edited activation; exact rank",
            )
            require(
                all(r["passes"] and r["error"] <= r["bound"] for r in meta["checks"]),
                "Primitive/install/zero checks",
            )
            for k, val in meta["counts"].items():
                counts[k] = counts.get(k, 0) + val
    cost = read(dest / "cost.json")
    require(
        cost["completed_producer_full_F"] == 8
        and cost["completed_candidate_and_zero_forwards"] == counts.pop("full_forwards") == 80
        and cost["original_stopped_forwards"] == 77
        and cost["additional_component_calls"] == counts
        and cost["ordinary_checkpoint_true"] == 0
        and cost["namespace_correction_model_calls"] == 0
        and not cost["wallclock_speedup_claimed"],
        "All actual work",
    )
    native, pairs, local = (rows(dest / (n + ".csv")) for n in ("native", "pairs", "local"))
    vr = read(dest / "vector-records.json")
    require(vr["model_calls"] == 0, "No correction model calls")
    nv = [v for v in vr["statistics"] if v["contrast"] == "native"]
    pv = [v for v in vr["statistics"] if v["contrast"] == "pairs"]
    ni = {(v["key"], v["arm"], v["scope"]): v for v in nv}
    pi = {(v["block"], v["arm"], v["scope"]): v for v in pv}
    require(
        len(native) == 40
        and len(pairs) == 20
        and len(ni) == 80
        and len(pi) == 40
        and len(local) == 8 * 28 * 2 * 3,
        "Complete scientific coverage",
    )
    for v in vr["statistics"]:
        vstats(v)
    scores = {r["key"]: r["scores"] for r in read(dest / "scores.json")["cases"]}

    oldF = {v["key"]: v["scores"] for v in read(old / "pre-native/scores.json")["cases"]}
    oldT = {v["key"]: v["scores"] for v in read(old / "native-phase/scores.json")["cases"]}
    bp = read(dest / "namespace-qualification/baseline-proof.json")["cases"]
    require(
        len(bp) == 8 and all(not r["fixed_condition_equals_baseline"] for r in bp),
        "Distinct B0 and fixed condition",
    )
    for v in bp:
        key = v["key"]
        require(
            scores[key]["B0_scores"] == oldF[key]["F_scores"]
            and scores[key]["T_scores"] == oldT[key]["scores"],
            "Sealed original False/native score provenance",
        )

    def passed(r, index, id_, scopes, rel, cos):
        result = True
        for scope in scopes:
            v = index[(*id_, scope)]
            n, re, co = vstats(v)
            close(number(r, scope + "_relative"), re, "Recorded relative error")
            close(number(r, scope + "_cosine"), co, "Recorded alignment")
            result = result and n >= 0.001 and re <= rel and co >= cos
        return result

    for r in native:
        sc = scores[r["key"]]
        err = abs(margin(sc[r["arm"] + "_scores"]) - margin(sc["T_scores"]))
        close(number(r, "margin_error"), err, "Actual finite margin")
        require(
            boolean(r, "passes")
            == (
                passed(r, ni, (r["key"], r["arm"]), ("receiving", "verdict"), 0.05, 0.99)
                and err <= 0.1
            ),
            "Native criterion",
        )
    for r in pairs:
        cs = sorted(
            [c for c in expected if c["group"]["block"] == r["block"]],
            key=lambda c: c["group"]["schema"],
        )
        sa, sb = (scores[c["key"]] for c in cs)

        def difference(arm):
            return (
                margin(sa[arm + "_scores"])
                - margin(sa["B0_scores"])
                - margin(sb[arm + "_scores"])
                + margin(sb["B0_scores"])
            )

        pred, target = difference(r["arm"]), difference("T")
        for name, val in [("prediction", pred), ("target", target), ("error", abs(pred - target))]:
            close(number(r, name), val, "Signed paired margin")
        require(
            boolean(r, "passes")
            == (
                passed(r, pi, (r["block"], r["arm"]), ("source", "verdict"), 0.1, 0.995)
                and abs(pred - target) <= 0.1
            ),
            "Paired criterion",
        )
    for r in local:
        norm, _, _ = vector(r)
        require(boolean(r, "eligible") == (norm >= 0.001), "Local norm guard")
    for arm in ARMS:
        rr = [r for r in native if r["arm"] == arm]
        pp = [r for r in pairs if r["arm"] == arm]
        lc = {
            b: sum(boolean(r, "passes") for r in rr if r["block"] == b)
            for b in sorted({r["block"] for r in rr})
        }
        n = sum(lc.values())
        pn = sum(boolean(r, "passes") for r in pp)
        require(
            a["outcomes"][arm]
            == dict(
                native=n,
                total=8,
                layouts=lc,
                pairs=pn,
                pair_total=4,
                sufficient=n >= 6 and min(lc.values()) >= 1 and pn >= 3,
            ),
            "Frozen sufficient-account rule",
        )
    require(
        a["primary_pass"] == a["outcomes"]["E"]["sufficient"]
        and a["adaptation_specific"]
        == (
            a["primary_pass"]
            and a["outcomes"]["F"]["native"] <= 2
            and a["outcomes"]["R"]["native"] <= 2
        ),
        "Primary and specificity",
    )
    cpu = read(dest / "cpu/preflight-summary.json")
    rcpu = read(dest / "resource-cpu/preflight-summary.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["rejected"] == ["bounds", "basis"]
        and rcpu["passed"]
        and rcpu["resource_only_literals"] == 4
        and rcpu["replayed_cases"] == 8
        and rcpu["coherent_mismatch_rejected"],
        "CPU complete/negative/resource fixtures",
    )
    require(
        read(dest / "cpu/expected-corruption-audit.json")["status"] == "failed",
        "Intentional corruption rejected",
    )
    nc = read(dest / "namespace-cpu/preflight-summary.json")
    require(
        nc["passed"]
        and nc["all_baselines_distinct"]
        and nc["candidate_arrays_unchanged"]
        and nc["coherent_baseline_corruption_rejected"]
        and nc["correction_model_calls"] == 0,
        "Independent namespace fixture",
    )
    fig = root / "docs/figures/response-subspace-293"
    prov = read(fig / "provenance.json")
    require(
        prov["all_eight_cases"]
        and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe293.py"),
        "Figure provenance",
    )
    for name, h in prov["inputs"].items():
        require(sha(root / name) == h, "Figure inputs")
    for name, h in prov["outputs"].items():
        require(sha(fig / name) == h, "Figure outputs")
    print(
        json.dumps(
            dict(
                passed=True,
                experiment=293,
                retrospective=True,
                required_checks=aold["required_checks"],
                outcomes=a["outcomes"],
                primary_pass=a["primary_pass"],
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    verify(ROOT)
