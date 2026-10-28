"""Stdlib verification of292's pre-native seals, signed records and frozen decisions.

Does not rerun the checkpoint or independently replicate omitted activation arrays.
"""

import hashlib
import itertools
import json
import math
import subprocess
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
DEST = Path("results/joint-gate-292")
ARMS = ("P", "D", "C", "B", "A", "T")
LAYOUTS = ("table", "dialogue", "user_rule", "xml")


def vector_record(r):
    p, t, d, e = (
        r[k]
        for k in (
            "prediction_squared_norm",
            "target_squared_norm",
            "prediction_target_dot",
            "error_squared_norm",
        )
    )
    require(
        all(math.isfinite(v) for v in (p, t, d, e)) and min(p, t, e) >= 0, "Finite squared norms"
    )
    require(
        math.isclose(e, p + t - 2 * d, rel_tol=1e-10, abs_tol=1e-9), "Signed squared-error identity"
    )
    n = t**0.5
    for field, value in [
        ("norm", n),
        ("relative", e**0.5 / max(n, 1e-30)),
        ("cosine", d / max((p * t) ** 0.5, 1e-30)),
        ("signed_target_fraction", d / max(t, 1e-30)),
    ]:
        close(r[field], value, "Signed vector " + field)
    return n, r["relative"], r["cosine"]


def verify(root):
    dest = root / DEST
    pre = dest / "pre-native"
    post = dest / "native-phase"
    for mf in ("pre-manifest.json", "release-manifest.json"):
        m = read(dest / mf)
        require(m["experiment"] == 292, "Manifest identity")
        for name, h in m["files"].items():
            require(sha(relative(root, name)) == h, "Sealed bytes " + name)
            require(
                not ({".context", "drafts"} & set(Path(name).parts))
                and Path(name).suffix not in (".npz", ".npy", ".pt", ".safetensors"),
                "Public scope",
            )
    release = read(dest / "release-manifest.json")
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    require(
        actual == {n for n in release["files"] if n.startswith(str(DEST) + "/")},
        "Exact public inventory",
    )
    d, fs, fa, seal = (
        read(pre / (n + ".json"))
        for n in ("design", "forecast-status", "forecast-audit", "forecast-seal")
    )
    ns, na, start = (
        read(post / (n + ".json")) for n in ("native-status", "native-audit", "native-started")
    )
    source_seal(root, d, "joint292")
    require(
        len(d["source_hashes"]) == 102
        and read(dest / "pre-manifest.json")["scientific_sources"] == d["source_hashes"],
        "102 immutable scientific files",
    )
    require(
        d["globally_pre_native"]
        and not d["synthetic"]
        and d["guards_MiB"] == [512, 384, 256]
        and d["arms"] == list(ARMS[:-1]),
        "Production design",
    )
    require(
        d["bounds"]
        == dict(replay=0.0, witness=1e-4, head=0.005, derivative_AD=1e-8, derivative_FD=2e-6),
        "Original numerical bounds",
    )
    require(
        d["primary"]
        == dict(
            native_relative=0.05,
            native_cosine=0.99,
            margin=0.1,
            trajectory_relative=0.1,
            trajectory_cosine=0.995,
            norm_min=0.001,
            effect_relative=0.25,
            effect_cosine=0.95,
            effect_absolute=0.005,
            effect_margin_relative=0.25,
            material=0.02,
            removal_vector=0.1,
            removal_margin=0.2,
            population=24,
            per_layout=6,
        ),
        "Original endpoints",
    )
    require(
        fs["ordinary_true"]
        == fa["ordinary_true"]
        == seal["ordinary_true"]
        == start["ordinary_true_before"]
        == 0,
        "No preseal native True",
    )
    require(
        fs["status"] == ns["status"] == "complete"
        and fa["status"] == na["status"] == "passed"
        and fa["failed_checks"] == na["failed_checks"] == 0,
        "Required gates",
    )
    require(
        ns["ordinary_true"] == 32
        and not na["general_question_B_solved"]
        and not na["learned_circuit_identified"],
        "Claim scope",
    )
    for field, name in [
        ("design_sha256", "design"),
        ("status_sha256", "forecast-status"),
        ("audit_sha256", "forecast-audit"),
    ]:
        require(seal[field] == sha(pre / (name + ".json")), "Global seal " + field)
    require(
        fa["design_sha256"] == sha(pre / "design.json")
        and fa["status_sha256"] == sha(pre / "forecast-status.json"),
        "Forecast audit inputs",
    )
    require(
        seal["source_hashes"] == d["source_hashes"] and seal["cases"] == fs["cases"],
        "Global sources and cases",
    )
    require(
        na["native_status_sha256"] == sha(post / "native-status.json")
        and ns["native_started_sha256"] == sha(post / "native-started.json"),
        "Native audit inputs",
    )
    require(
        ns["forecast_seal_sha256"]
        == na["forecast_seal_sha256"]
        == start["forecast_seal_sha256"]
        == sha(pre / "forecast-seal.json"),
        "Native seal binding",
    )
    require(start["remote_head_verified"] and not start["synthetic"], "Original pushed receipt")
    published = subprocess.check_output(
        ["git", "show", start["public_commit"] + ":" + str(DEST / "pre-native/forecast-seal.json")],
        cwd=root,
    )
    require(
        hashlib.sha256(published).hexdigest() == sha(pre / "forecast-seal.json"),
        "Public forecast bytes before measurement",
    )
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", start["public_commit"], "HEAD"],
            cwd=root,
            capture_output=True,
        ).returncode
        == 0,
        "Public receipt ancestry",
    )
    for audit, folder, total in [(fa, pre, 76128), (na, post, 96)]:
        rr = []
        for name, h in audit["csv_sha256"].items():
            require(sha(folder / name) == h, "Audit CSV bytes")
            if name.endswith("checks.csv"):
                rr.extend(rows(folder / name))
        require(audit["required_checks"] == len(rr) == total, "Required numerical denominator")
        for r in rr:
            require(
                0 <= number(r, "error") <= number(r, "bound") and boolean(r, "passes"),
                "Numeric decision",
            )
    cases = {c["key"]: c for c in d["cases"]}
    require(len(cases) == 32 and len(d["historical_hashes"]) == 2176, "Freshness coverage")
    seen = set(d["historical_hashes"])
    for c in cases.values():
        f, t = c["false_ids"], c["true_ids"]
        require(
            len(f) == len(t)
            and [j for j, (a, b) in enumerate(zip(f, t, strict=True)) if a != b] == [c["position"]],
            "Single edit",
        )
        require(f[c["position"]] == 895 and t[c["position"]] == 830, "Known Boolean edit")
        hs = [
            hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()
            for x in (f, t)
        ]
        require(hs == c["input_hashes"] and not set(hs) & seen, "Fresh complete pairs")
        seen.update(hs)
    for b in LAYOUTS:
        require(
            {
                (c["group"]["schema"], c["group"]["reverse"], c["group"]["target"])
                for c in cases.values()
                if c["group"]["block"] == b
            }
            == set(itertools.product(("generic", "descriptive"), (0, 1), (0, 1))),
            "Crossed eight per layout",
        )
    scorebanks = []
    for folder, status in [(pre, fs), (post, ns)]:
        require([e["key"] for e in status["cases"]] == list(cases), "Ordered all32")
        inv = read(folder / "omitted-array-inventory.json")
        require(
            inv["case_bytes"] == status["raw_bytes"] == sum(e["bytes"] for e in status["cases"]),
            "Archive size",
        )
        for e, v in zip(status["cases"], inv["arrays"], strict=True):
            require(
                v["path"] == "cases/" + e["key"] + ".npz"
                and v["sha256"] == e["arrays_sha256"]
                and v["bytes"] == e["bytes"],
                "Omitted array binding",
            )
            p = folder / "cases" / (e["key"] + ".json")
            require(sha(p) == e["metadata_sha256"], "Metadata binding")
            m = read(p)
            require(m["arrays_sha256"] == e["arrays_sha256"], "Case array seal")
            if folder == pre:
                require(
                    set(m["arms"]) == {"F", *ARMS[:-1], *(a + "0" for a in ARMS[:-1])},
                    "All forecast/zero arms",
                )
                for arm, meta in m["arms"].items():
                    require(
                        meta["ordinary_true"] == 0 and meta["activation_calls"] == 28,
                        "Forecast computation count",
                    )
                    if arm != "F":
                        require(
                            meta["editing_hooks_applied"]
                            and meta["current_affected_finite_SiLU_coordinates"] == 0
                            and meta["prefixes"] == m["arms"]["F"]["prefixes"],
                            "No finite edited activation; exact prefix",
                        )
                    require(
                        all(r["passes"] and r["error"] <= r["bound"] for r in meta["checks"]),
                        "Perlayer witnesses",
                    )
            else:
                require(
                    m["arm"]["ordinary_true"] == 1 and not m["arm"]["editing_hooks_applied"],
                    "Native ordinary computation",
                )
        scorebanks.append({e["key"]: e["scores"] for e in read(folder / "scores.json")["cases"]})
    require(fs["raw_bytes"] + ns["raw_bytes"] <= 256 * 1024**2, "Combined frozen budget")
    require(
        fs["counts"]
        == dict(full_F=32, full_forecast=160, full_Fzero=160, affected_current_finite_SiLU=0)
        and fa["counts"] == dict(full_F=32, full_P=32, full_Pzero=32)
        and na["counts"] == dict(full_unhooked_true_replays=32),
        "Full work counts",
    )
    review = read(dest / "posthoc-vector-review.json")
    require(
        review["posthoc"]
        and review["model_calls"] == 0
        and review["script_sha256"] == sha(root / "research/boolean_coupling/review292.py")
        and review["native_audit_sha256"] == sha(post / "native-audit.json"),
        "Posthoc scope",
    )
    for field, status in [("forecast_archives", fs), ("native_archives", ns)]:
        require(
            review[field] == {e["key"]: e["arrays_sha256"] for e in status["cases"]},
            "Posthoc archive binding",
        )
    vi = {(r["key"], r["arm"], r["contrast"], r["scope"]): r for r in review["final"]}
    require(len(vi) == 32 * 6 * 2 * 2, "Vector coverage")
    for r in (*review["final"], *review["layers"]):
        vector_record(r)
    require(len(review["layers"]) == 32 * 28 * 3, "All layer accounting")
    native, effects, trajectories, coefficients = (
        rows(post / (n + ".csv")) for n in ("native", "effects", "trajectories", "coefficients")
    )
    ni = {(r["key"], r["arm"]): r for r in native}
    require(
        len(ni) == len(native) == 192
        and len(effects) == 192
        and len(trajectories) == 96
        and len(coefficients) == 192,
        "Scientific record coverage",
    )

    def vp(r, scope, contrast, rel, cos):
        v = vi[r["key"], r["arm"], contrast, scope]
        for f in ("relative", "cosine", "norm"):
            close(number(r, scope + "_" + f), v[f], "Published vector metric")
        return v["norm"] >= 0.001 and v["relative"] <= rel and v["cosine"] >= cos

    for r in native + effects:
        key, arm = r["key"], r["arm"]
        f = scorebanks[0][key]
        t = scorebanks[1][key]
        tm = margin(t["scores"])
        pm = tm if arm == "T" else margin(f[arm + "_scores"])
        am = margin(f["A_scores"])
        if r in native:
            err = abs(pm - tm)
            close(number(r, "margin_error"), err, "Finite margin")
            close(
                number(r, "native_margin_change"),
                tm - margin(f["F_scores"]),
                "Native margin change",
            )
            passed = (
                vp(r, "receiving", "native", 0.05, 0.99)
                and vp(r, "verdict", "native", 0.05, 0.99)
                and err <= 0.1
            )
        else:
            pred, target = pm - am, tm - am
            for field, value in [
                ("prediction", pred),
                ("effect_target", target),
                ("error", abs(pred - target)),
            ]:
                close(number(r, field), value, "Signed finite effect")
            a = ni[key, "A"]
            material = abs(target) >= 0.02 and (
                number(a, "receiving_relative") >= 0.1 or number(a, "margin_error") >= 0.2
            )
            require(boolean(r, "material") == material, "Materiality includes all32")
            passed = (
                material
                and vp(r, "receiving", "effects", 0.25, 0.95)
                and vp(r, "verdict", "effects", 0.25, 0.95)
                and abs(pred - target) <= 0.005 + 0.25 * abs(target)
            )
        require(boolean(r, "passes") == passed, "Frozen decision")
    for r in trajectories + coefficients:
        norm, rel, cos = vector(r)
        require(
            boolean(r, "passes") == (norm >= 0.001 and rel <= 0.1 and cos >= 0.995),
            "Trajectory/local coefficient decision",
        )

    def population(rr):
        counts = {b: sum(boolean(r, "passes") for r in rr if r["block"] == b) for b in LAYOUTS}
        total = sum(counts.values())
        return dict(
            passes=total,
            total=32,
            layouts=counts,
            population_pass=total >= 24 and min(counts.values()) >= 6,
        )

    for name, rr in [("native", native), ("effects", effects)]:
        for arm in ARMS:
            require(
                na[name][arm] == population([r for r in rr if r["arm"] == arm]),
                "Population criterion",
            )
    joint = [
        dict(
            key=k,
            block=c["group"]["block"],
            passes=str(
                boolean(ni[k, "P"], "passes")
                and all(boolean(r, "passes") for r in trajectories if r["key"] == k)
            ),
        )
        for k, c in cases.items()
    ]
    require(na["trajectory_joint"] == population(joint), "H2 population")
    require(
        na["primary_pass"]
        == dict(
            H1_native=na["native"]["P"]["population_pass"],
            H2_trajectory=na["trajectory_joint"]["population_pass"],
            H3_joint_product_effect=na["effects"]["P"]["population_pass"],
        ),
        "Primary priority",
    )
    continuation = read(pre / "continuation.json")
    resident = read(post / "resident-complete.json")
    require(
        continuation["method_unchanged"]
        and continuation["first_case_replayed_before_any_remaining_case"]
        and continuation["replayed_arrays"] == 42,
        "Resource replay qualification",
    )
    require(
        resident["model_loads"] == 1
        and resident["parameter_objects_versions_unchanged"]
        and resident["native_gate_unchanged"]
        and resident["calculation_source_hashes"] == d["source_hashes"],
        "Resident model integrity",
    )
    cpu = read(dest / "cpu-qualification/preflight-summary.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["forecast_audit"] == 1216
        and cpu["native_audit"] == 12
        and cpu["rejected"] == ["bounds", "qualification", "last_array", "receipt", "primitive"],
        "CPU pipeline and negative fixtures",
    )
    prov = read(root / "docs/figures/joint-gate-292/provenance.json")
    require(
        prov["all_32_cases"]
        and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe292.py"),
        "Figure provenance",
    )
    for name, h in prov["inputs"].items():
        require(sha(root / name) == h, "Figure input")
    for name, h in prov["outputs"].items():
        require(sha(root / "docs/figures/joint-gate-292" / name) == h, "Figure output")
    print(
        json.dumps(
            dict(
                passed=True,
                experiment=292,
                pre_native_checks=fa["required_checks"],
                native_checks=na["required_checks"],
                primary_pass=na["primary_pass"],
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    verify(ROOT)
