"""Stdlib291 record verification; omitted primitive parents/gradients are not replicated."""

import itertools
import json
import math
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
DEST = Path("results/receiving-coefficients-291")
CORNERS = tuple(f"{j:03b}" for j in range(8))
ARMS = (*CORNERS, "R0", "R1", "R2")


def verify(root):
    dest = root / DEST
    release = read(dest / "release-manifest.json")
    require(release["experiment"] == 291 and release["retrospective"], "Release identity")
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
    d, st, audit = (read(dest / (n + ".json")) for n in ("design", "status", "audit"))
    source_seal(root, d, "coeff291")
    require(
        len(d["source_hashes"]) == 94 and release["scientific_sources"] == d["source_hashes"],
        "Scientific source coverage",
    )
    require(
        d["experiment"] == 291 and d["retrospective"] and not d["synthetic"], "Production design"
    )
    require(
        d["arms"] == list(ARMS) and d["FD_steps"] == [0.05, 0.025] and d["FD_unit_delta"],
        "Arms and FD directions",
    )
    require(
        d["factors"] == ["gate_response", "up_multiplier", "up_response_coefficient"],
        "Primitive-family definitions",
    )
    require(
        d["primary"]
        == dict(
            hypotheses={
                "GATE": "100",
                "UP_MULTIPLIER": "010",
                "UP_COEFFICIENT": "001",
                "GATE_AND_UP": "110",
                "GATE_AND_COEFFICIENT": "101",
                "UP_AND_COEFFICIENT": "011",
            },
            norm_min=0.001,
            relative_max=0.25,
            cosine_min=0.95,
            finite_absolute=0.005,
            finite_relative=0.25,
            material_min=0.02,
            population_min=24,
            per_layout_min=6,
        ),
        "Frozen coefficient endpoints",
    )
    require(
        d["numerical_bounds"]
        == dict(
            primal=1e-4,
            boundary_gradient=1e-4,
            primitive=1e-4,
            cached_finite_coordinate=0.002,
            cached_finite_score=0.001,
            CPU64_head=0.005,
            AD_FD_absolute=0.001,
            AD_FD_relative=0.02,
            halving_absolute=0.002,
            halving_relative=0.03,
        ),
        "Frozen numeric bounds",
    )
    require(d["resource_guards_MiB"] == [512, 384, 192], "Frozen guards")
    require(
        st["status"] == "complete" and audit["status"] == "passed" and audit["failed_checks"] == 0,
        "Required gate",
    )
    require(
        audit["design_sha256"] == sha(dest / "design.json")
        and audit["status_sha256"] == sha(dest / "status.json"),
        "Audit inputs",
    )
    require(
        audit["audit_source_sha256"] == sha(root / "research/boolean_coupling/audit291.py"),
        "Frozen auditor",
    )
    require(
        not audit["general_question_B_solved"] and not audit["learned_circuit_identified"],
        "Claim scope",
    )
    old285 = root / "results/native-structure-285"
    for name, phase in (
        ("design", "pre-native"),
        ("forecast-status", "pre-native"),
        ("native-status", "native-phase"),
    ):
        require(
            d["reference_hashes"]["285_" + name] == sha(old285 / phase / (name + ".json")),
            "285 reference",
        )
    old290 = root / "results/receiving-nonlinearity-290"
    for name in ("design", "status", "audit"):
        require(
            d["reference_hashes"]["290_" + name] == sha(old290 / (name + ".json")), "290 reference"
        )
    require(
        d["reference_hashes"]["287_head"]
        == sha(root / "results/source-receiver-readout-287/head.json"),
        "Head reference",
    )
    old_failure = read(root / "results/source-receiver-readout-287/audit.json")
    require(
        old_failure["status"] == "failed" and old_failure["failed_checks"] == 1,
        "287 failure retained",
    )
    for name, h in audit["csv_sha256"].items():
        require(sha(dest / name) == h, "Audit table " + name)
    cases = {c["key"]: c for c in d["cases"]}
    require(
        len(cases) == len(st["cases"]) == 32 and list(cases) == [e["key"] for e in st["cases"]],
        "All32 ordered cases",
    )
    inv = read(dest / "omitted-array-inventory.json")
    require(
        inv["case_bytes"]
        == st["raw_bytes"]
        == sum(e["bytes"] for e in st["cases"])
        <= 192 * 1024**2,
        "Archive budget",
    )
    scores = {e["key"]: e["scores"] for e in read(dest / "scores.json")["cases"]}
    checks, derivatives, primitives = (
        rows(dest / (n + ".csv")) for n in ("checks", "derivatives", "primitives")
    )
    require(
        (len(checks), len(derivatives), len(primitives)) == (2112, 3168, 3264),
        "Required record coverage",
    )
    require(
        audit["required_checks"] == len(checks) + len(derivatives) + len(primitives) == 8544,
        "Required denominator",
    )
    for r in checks + derivatives + primitives:
        require(
            0 <= number(r, "error") <= number(r, "bound") and boolean(r, "passes"),
            "Numerical decision",
        )
    fd_index = {(r["key"], r["arm"], r["direction"], r["step"]): r for r in derivatives}
    collected_counts = []
    for key, entry, inventory in zip(cases, st["cases"], inv["arrays"], strict=True):
        require(
            inventory["path"] == "cases/" + key + ".npz"
            and inventory["sha256"] == entry["arrays_sha256"]
            and inventory["bytes"] == entry["bytes"],
            "Raw archive identity",
        )
        path = dest / "cases" / (key + ".json")
        require(sha(path) == entry["metadata_sha256"], "Case metadata hash")
        m = read(path)
        require(
            m["arrays_sha256"] == entry["arrays_sha256"] and m["prefix_unchanged"],
            "Case/prefix seal",
        )
        require(
            m["first_source_policy"]["activation_calls"] == 1
            and m["first_source_policy"]["False_primal_exact"],
            "Before-SiLU gate witness",
        )
        require(len(m["checks"]) == 66 and len(m["derivatives"]) == 99, "Per-case checks")
        require(
            len(m["primitive_checks"]) == 102 and len(m["primitive_zero_checks"]) == 0,
            "Primitive coverage",
        )
        require(
            set(m["source_paths"]) == set(ARMS)
            and all(
                v == m["source_paths"]["000"] and len(v) == 27 for v in m["source_paths"].values()
            ),
            "Source path invariance",
        )
        cc = m["counts"]
        require(
            cc
            == dict(
                full_F=1,
                first_block=1,
                anchor_suffixes=2,
                anchor_blocks=54,
                modified_forwards=1,
                modified_blocks=27,
                modified_heads=1,
                tape_backwards=0,
                triangular_blocks=0,
                triangular_block_VJPs=0,
                triangular_head_VJPs=0,
                native_suffixes=1,
                native_blocks=27,
                interface_forwards=154,
                interface_blocks=8316,
                interface_heads=154,
                interface_backwards=11,
                blockwise_blocks=594,
                blockwise_VJPs=297,
                blockwise_head_VJPs=11,
                encoding_norm_calls=4455,
                reference_forwards=1,
                reference_backwards=1,
                primitive_norm_VJPs=0,
                primitive_attention_VJPs=0,
                anchor_qk_recomputations=54,
                receiving_norm_calls=0,
                receiving_MLP_reanchors=0,
                receiver_norm_probe_VJPs=0,
                coefficient_maps=4455,
                mixed_base_down=1620,
                coefficient_projections=11340,
                coefficient_probe_VJPs=11,
                coefficient_probe_down=11,
                coefficient_probe_finite_down=88,
            ),
            "Full work accounting",
        )
        collected_counts.append(cc)
        for r in m["checks"]:
            matches = [s for s in checks if s["key"] == key and s["name"] == r["name"]]
            require(
                len(matches) == 1
                and number(matches[0], "error") == r["error"]
                and number(matches[0], "bound") == r["bound"],
                "Replay witness",
            )
        for r in m["primitive_checks"] + m["primitive_zero_checks"]:
            matches = [
                s
                for s in primitives
                if s["key"] == key
                and s["arm"] == r["arm"]
                and int(s["layer"]) == r["layer"]
                and s["node"] == r["node"]
            ]
            require(
                len(matches) == 1
                and number(matches[0], "error") == r["error"]
                and number(matches[0], "bound") == r["bound"],
                "Primitive AD/formula witness",
            )
        for r in m["derivatives"]:
            arm, label, step, ad = r["arm"], r["direction"], r["step"], r["AD"]
            s = fd_index[key, arm, label, str(step)]
            close(number(s, "AD"), ad, "Projected AD binding")
            if step == "halving":
                err = abs(
                    number(fd_index[key, arm, label, "0.05"], "FD")
                    - number(fd_index[key, arm, label, "0.025"], "FD")
                )
                bound = 0.002 + 0.03 * abs(ad)
            else:
                estimate = (margin(r["plus_scores"]) - margin(r["minus_scores"])) / (2 * step)
                close(estimate, r["FD"], "Actual finite scores")
                err, bound = abs(ad - estimate), 0.001 + 0.02 * abs(ad)
            close(number(s, "error"), err, "FD error")
            close(number(s, "bound"), bound, "FD bound")
    pp = read(dest / "primitive-probe-scores.json")["records"]
    require(len(pp) == 32 * 11 * 2, "Local primitive FD coverage")
    for p in pp:
        estimates = []
        for step in (0.05, 0.025):
            plus, minus = p["scores"][str(step)]
            fd = (plus - minus) / (2 * step)
            estimates.append(fd)
            match = [
                x
                for x in primitives
                if x["key"] == p["key"]
                and x["arm"] == p["arm"]
                and x["node"] == p["kind"] + "_FD_" + str(step)
            ]
            require(len(match) == 1, "Primitive FD row identity")
            close(number(match[0], "error"), abs(fd - p["AD"]), "Actual local primitive FD")
            close(number(match[0], "bound"), 0.001 + 0.02 * abs(p["AD"]), "Primitive FD bound")
        match = [
            x
            for x in primitives
            if x["key"] == p["key"] and x["arm"] == p["arm"] and x["node"] == p["kind"] + "_halving"
        ]
        require(len(match) == 1, "Primitive halving row")
        close(
            number(match[0], "error"), abs(estimates[0] - estimates[1]), "Primitive halving error"
        )
        close(number(match[0], "bound"), 0.002 + 0.03 * abs(p["AD"]), "Primitive halving bound")
    require(
        st["counts"] == {n: sum(c[n] for c in collected_counts) for n in collected_counts[0]},
        "Total work",
    )
    raw, trace = (
        rows(dest / (n + ".csv")) for n in ("original_coordinates", "intermediate_diagnostics")
    )
    require(len(raw) == 96 and len(trace) == 32 * 11 * 27, "Original diagnostic coverage")
    for r in raw + trace:
        require(
            number(r, "bound") == 1e-4 and boolean(r, "passes") == (number(r, "error") <= 1e-4),
            "Retained narrow bound",
        )
    require(
        sum(not boolean(r, "passes") for r in raw) == audit["original_coordinate_failures"],
        "Raw failure count",
    )
    require(
        sum(not boolean(r, "passes") for r in trace) == audit["intermediate_diagnostic_failures"],
        "Trace failure count",
    )
    h, v, n, ledger = (
        rows(dest / (name + ".csv"))
        for name in ("hypotheses", "receiving_vectors", "native_scalar", "ledger")
    )
    for data in (h, v, n):
        require(
            len(data) == 352
            and {(r["key"], r["arm"]) for r in data} == set(itertools.product(cases, ARMS)),
            "Complete eleven-arm cohort",
        )
    require(
        len(ledger) == 224
        and {(r["key"], r["term"]) for r in ledger} == set(itertools.product(cases, CORNERS[1:])),
        "All interaction terms",
    )
    ni = {(r["key"], r["arm"]): r for r in n}
    ei = {
        (key, arm): margin(sc[arm + "_finite_raw_scores"]) - margin(sc["anchor1_scores"])
        for key, sc in scores.items()
        for arm in ARMS
    }
    old_native = {r["key"]: number(r, "native_target") for r in rows(old290 / "native_scalar.csv")}
    for r in h:
        key, arm = r["key"], r["arm"]
        norm, rel, cos = vector(r)
        target = ei[key, "111"] - ei[key, "000"]
        pred = ei[key, arm] - ei[key, "000"]
        err = abs(pred - target)
        for field, value in (
            ("finite_target", target),
            ("finite_prediction", pred),
            ("finite_error", err),
        ):
            close(number(r, field), value, "Signed finite comparison")
        vp, ep, material = (
            norm >= 0.001 and rel <= 0.25 and cos >= 0.95,
            err <= 0.005 + 0.25 * abs(target),
            abs(target) >= 0.02,
        )
        for field, val in (
            ("vector_pass", vp),
            ("finite_pass", ep),
            ("passes", vp and ep),
            ("material_eligible", material),
            ("material_pass", vp and ep and material),
        ):
            require(boolean(r, field) == val, "Hypothesis decision")
    promoted_records = {
        (r["key"], r["arm"]): r for r in read(dest / "posthoc-review.json")["records"]
    }
    for r in v:
        norm, rel, cos = vector(r)
        require(
            boolean(r, "passes") == (norm >= 0.001 and rel <= 0.25 and cos >= 0.95),
            "Receiving-vector decision",
        )
        close(
            number(r, "signed_target_projection"),
            promoted_records[r["key"], r["arm"]]["original_F32_dot"]
            / max(promoted_records[r["key"], r["arm"]]["original_F32_target_norm"], 1e-30),
            "Original Float32 signed vector projection",
        )
    for r in n:
        y, pred = old_native[r["key"]], number(r, "prediction")
        close(number(r, "native_target"), y, "Old native scalar reference")
        close(number(r, "finite_effect"), ei[r["key"], r["arm"]], "Actual finite effect")
        close(number(r, "error"), abs(y - pred), "Scalar error")
        require(boolean(r, "passes") == (abs(y - pred) <= 0.1), "Scalar criterion")
    for r in ledger:
        key, term = r["key"], r["term"]
        subset = [j for j, b in enumerate(term) if b == "1"]
        finite, scalar = 0.0, 0.0
        for bits in itertools.product((0, 1), repeat=len(subset)):
            c = ["0"] * 3
            for j, b in zip(subset, bits, strict=True):
                c[j] = str(b)
            c = "".join(c)
            sign = (-1) ** (len(subset) - sum(bits))
            finite += sign * ei[key, c]
            scalar += sign * number(ni[key, c], "prediction")
        close(number(r, "finite_contribution"), finite, "Finite interaction accounting")
        close(number(r, "projection_on_delta"), scalar, "Projected gradient accounting")
    for arm in ARMS:
        rr = [r for r in h if r["arm"] == arm]
        layouts = {
            b: sum(boolean(r, "passes") for r in rr if r["block"] == b)
            for b in ("table", "dialogue", "user_rule", "xml")
        }
        count = sum(boolean(r, "passes") for r in rr)
        require(
            audit["arms"][arm]
            == dict(
                passes=count,
                total=32,
                layouts=layouts,
                population_pass=count >= 24 and all(c >= 6 for c in layouts.values()),
                material_passes=sum(boolean(r, "material_pass") for r in rr),
                material_eligible=sum(boolean(r, "material_eligible") for r in rr),
            ),
            "Population rule",
        )
    require(
        audit["primary_pass"]
        == {
            name: audit["arms"][arm]["population_pass"]
            for name, arm in d["primary"]["hypotheses"].items()
        },
        "Primary has priority",
    )
    review = read(dest / "posthoc-review.json")
    require(review["posthoc"] and review["model_calls"] == 0, "Posthoc scope")
    require(
        review["script_sha256"] == sha(root / "research/boolean_coupling/review291.py")
        and review["audit_sha256"] == sha(dest / "audit.json"),
        "Posthoc provenance",
    )
    require(
        review["raw_archives"] == {e["key"]: e["arrays_sha256"] for e in st["cases"]},
        "Posthoc archive binding",
    )
    require(
        review["vector_comparisons"] == 352
        and review["interaction_comparisons"] == 224
        and review["classification_changes"] == 0,
        "Promoted rounding scope",
    )
    require(len(review["gradient_accounting"]) == 32, "Posthoc gradient coverage")
    hi = {(r["key"], r["arm"]): r for r in h}
    for r in review["gradient_accounting"]:
        gram = r["corner_gram"]
        require(len(gram) == 8 and all(len(v) == 8 for v in gram), "Gradient Gram dimensions")

        def product(a, b):
            return sum(a[i] * b[j] * gram[i][j] for i in range(8) for j in range(8))

        target = [-1, 0, 0, 0, 0, 0, 0, 1]
        norm = math.sqrt(max(0, product(target, target)))
        close(norm, r["target_norm"], "Gram target norm")
        for j, arm in enumerate(CORNERS):
            pred = [0] * 8
            pred[j] += 1
            pred[0] -= 1
            close(
                product(pred, pred),
                number(hi[r["key"], arm], "prediction_squared_norm"),
                "Gram prediction norm",
            )
            close(
                product(pred, target),
                number(hi[r["key"], arm], "prediction_target_dot"),
                "Gram signed alignment",
            )
        coefficients = {}
        for term in CORNERS[1:]:
            subset = [j for j, b in enumerate(term) if b == "1"]
            coeff = [0] * 8
            for bits in itertools.product((0, 1), repeat=len(subset)):
                c = ["0"] * 3
                for j, b in zip(subset, bits, strict=True):
                    c[j] = str(b)
                coeff[int("".join(c), 2)] += (-1) ** (len(subset) - sum(bits))
            coefficients[term] = coeff
            expected = dict(
                norm_ratio=math.sqrt(max(0, product(coeff, coeff))) / norm,
                target_projection=product(coeff, target) / norm**2,
            )
            for field, value in expected.items():
                require(
                    math.isclose(r["terms"][term][field], value, rel_tol=1e-8, abs_tol=1e-8),
                    "Posthoc factorial Gram arithmetic",
                )
        for field, degree in [("main_relative", 1), ("through_pairs_relative", 2)]:
            coeff = [
                sum(c[j] for k, c in coefficients.items() if k.count("1") <= degree) - target[j]
                for j in range(8)
            ]
            value = math.sqrt(max(0, product(coeff, coeff))) / norm
            require(
                math.isclose(r[field], value, rel_tol=1e-8, abs_tol=1e-8), "Posthoc additive error"
            )
    cpu = read(dest / "cpu-qualification/preflight-summary.json")
    require(
        cpu["passed"]
        and cpu["synthetic"]
        and not cpu["checkpoint_loaded"]
        and cpu["all_coordinate_FD_checks"] == 704,
        "Tiny CPU qualification",
    )
    require(cpu["tamper_rejected"] == ["bounds", "FD", "primitive"], "Negative fixtures")
    require(
        read(dest / "cpu-qualification/complete/design.json")["source_hashes"]
        == d["source_hashes"],
        "Final qualified code",
    )
    ca = read(dest / "cpu-qualification/complete/audit.json")
    require(ca["status"] == "passed" and ca["required_checks"] == 1068, "CPU saved audit")
    for r in read(dest / "cpu-qualification/all-coordinates.json")["checks"]:
        require(
            all(abs(r["AD"] - f) <= 0.001 + 0.02 * abs(r["AD"]) for f in r["FD"]),
            "Tiny all-coordinate FD",
        )
        require(abs(r["FD"][0] - r["FD"][1]) <= 0.002 + 0.03 * abs(r["AD"]), "Tiny FD halving")
    fig = root / "docs/figures/receiving-coefficients-291"
    prov = read(fig / "provenance.json")
    require(
        prov["all_32_cases"]
        and prov["script_sha256"] == sha(root / "research/boolean_coupling/describe291.py"),
        "Plot provenance",
    )
    for name, h in prov["inputs"].items():
        require(sha(root / name) == h, "Plot inputs")
    for name, h in prov["outputs"].items():
        require(sha(fig / name) == h, "Plot outputs")
    print(
        json.dumps(
            dict(
                passed=True,
                experiment=291,
                retrospective=True,
                required_checks=audit["required_checks"],
                primary_pass=audit["primary_pass"],
                arms=audit["arms"],
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    verify(ROOT)
