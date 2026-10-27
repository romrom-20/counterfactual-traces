"""Stdlib verification of287 records; failed original gate stays failed.

This checks published sufficient statistics, not omitted vectors or a model run.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from .verify285 import boolean, close, number, read, relative, require, rows, sha
    from .verify286 import margin, source_seal
except ImportError:
    from verify285 import boolean, close, number, read, relative, require, rows, sha
    from verify286 import margin, source_seal

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/source-receiver-readout-287")
CORNERS = ("FF", "MF", "FM", "MM")


def decision(row):
    require(
        boolean(row, "passes") == (0 <= number(row, "error") <= number(row, "bound")),
        "Numeric decision",
    )


def vector(row, prefix=""):
    p, t, dot, e = (
        number(row, prefix + k)
        for k in (
            "prediction_squared_norm",
            "target_squared_norm",
            "prediction_target_dot",
            "error_squared_norm",
        )
    )
    require(min(p, t, e) >= 0, "Squared norms")
    close(e, p + t - 2 * dot, "Vector squared-error identity")
    norm = math.sqrt(t)
    rel = math.sqrt(e) / max(norm, 1e-30)
    cos = dot / max(math.sqrt(p) * norm, 1e-30)
    for name, value in (("norm", norm), ("relative", rel), ("cosine", cos)):
        close(number(row, prefix + name), value, "Vector sufficient statistic: " + name)
    return norm, rel, cos


def verify(root):
    dest, qdest = root / DEST, root / DEST / "boundary-qualification"
    release = read(dest / "release-manifest.json")
    require(
        release["experiment"] == 287
        and release["retrospective"]
        and not release["original_numeric_gate_passed"],
        "Release identity",
    )
    for name, h in release["files"].items():
        require(sha(relative(root, name)) == h, "Release file: " + name)
        require(
            not ({".context", "drafts"} & set(Path(name).parts))
            and Path(name).suffix not in (".npz", ".npy"),
            "Public file scope",
        )
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    require(
        actual == {n for n in release["files"] if n.startswith(str(DEST) + "/")},
        "Exact public inventory",
    )
    design, status, audit = (read(dest / (n + ".json")) for n in ("design", "status", "audit"))
    qdesign, qa = read(qdest / "design.json"), read(qdest / "audit.json")
    source_seal(root, design, "factor287")
    source_seal(root, qdesign, "qualify287b")
    require(
        len(design["source_hashes"]) == 70 and len(qdesign["source_hashes"]) == 73,
        "Source coverage",
    )
    require(release["scientific_sources"] == qdesign["source_hashes"], "Release source union")
    require(
        all(qdesign["source_hashes"][n] == h for n, h in design["source_hashes"].items()),
        "Original science unchanged",
    )
    require(
        design["experiment"] == 287
        and design["retrospective"]
        and not design["synthetic"]
        and design["source_first"],
        "Production design",
    )
    require(
        design["arms"] == list(CORNERS) and design["FD_steps"] == [0.05, 0.025],
        "Four corners and FD steps",
    )
    require(
        design["primary"]
        == dict(
            target="gMM-gFF source-row margin adjoint shift",
            vector_norm_min=0.001,
            relative_max=0.25,
            cosine_min=0.95,
            finite_contrast_error_absolute=0.005,
            finite_contrast_error_relative=0.25,
            population_min=24,
            per_layout_min=6,
            hypotheses=["S", "R"],
        ),
        "Frozen scientific criteria",
    )
    require(
        design["numerical_bounds"]
        == dict(
            cached_coordinate=0.002,
            cached_relative=1e-4,
            cached_score=0.001,
            full_replay=1e-4,
            source_adjoint=1e-4,
            projected_response=1e-4,
            CPU64_head=0.005,
            projected_AD_FD_absolute=0.001,
            projected_AD_FD_relative=0.02,
            FD_halving_absolute=0.002,
            FD_halving_relative=0.03,
        ),
        "Original numerical bounds",
    )
    require(
        audit["status"] == "failed"
        and audit["failed_checks"] == 1
        and audit["required_checks"] == 3072,
        "Original gate remains failed",
    )
    require(
        qa["status"] == "qualified_boundary_only"
        and not qa["original_numeric_gate_passed"]
        and qdesign["posthoc_numerical_scope"]
        and qdesign["new_model_calls"] == qa["new_model_calls"] == 0,
        "Narrow posthoc qualification",
    )
    require(qdesign["boundary_coordinate_bound"] == 1e-4, "Unchanged boundary bound")
    require(
        qdesign["original_audit_sha256"] == qa["original_audit_sha256"] == sha(dest / "audit.json"),
        "Original failed audit binding",
    )
    require(
        audit["design_sha256"] == sha(dest / "design.json")
        and audit["status_sha256"] == sha(dest / "status.json"),
        "Original inputs",
    )
    require(
        audit["audit_source_sha256"]
        == design["source_hashes"]["research/boolean_coupling/audit287.py"],
        "Original audit source",
    )
    for directory, a in ((dest, audit), (qdest, qa)):
        for name, h in a["csv_sha256"].items():
            require(sha(directory / name) == h, "CSV binding")
    old = root / "results/native-structure-285"
    rd = root / "results/false-readout-286"
    refs = {
        "285_design": old / "pre-native/design.json",
        "285_forecast_status": old / "pre-native/forecast-status.json",
        "285_native_status": old / "native-phase/native-status.json",
        "286_design": rd / "design.json",
        "286_status": rd / "status.json",
        "286_audit": rd / "audit.json",
        "286b_design": rd / "full-shape-qualifier/design.json",
        "286b_audit": rd / "full-shape-qualifier/audit.json",
    }
    require(
        {n: sha(p) for n, p in refs.items()} == design["reference_hashes"], "Historical references"
    )
    require(design["cases"] == read(rd / "design.json")["cases"], "All previously observed cases")
    keys = [c["key"] for c in design["cases"]]
    require(
        len(keys) == len(set(keys)) == 32 and keys == [e["key"] for e in status["cases"]],
        "Ordered cohort coverage",
    )
    counts = dict(
        full_F=32,
        first_block=32,
        suffix_forwards=1792,
        backwards=128,
        full_shape_block_forwards=3456,
        block_recomputations=3456,
        block_VJPs=3456,
        head_VJPs=128,
    )
    require(
        status["status"] == "complete" and status["counts"] == audit["counts"] == counts,
        "Actual model work",
    )
    inventory = read(dest / "omitted-array-inventory.json")
    require(
        inventory["case_bytes"]
        == status["raw_bytes"]
        == sum(e["bytes"] for e in status["cases"])
        <= 64 * 1024**2,
        "Array storage",
    )
    require(
        inventory["arrays"]
        == [
            dict(path="cases/" + e["key"] + ".npz", sha256=e["arrays_sha256"], bytes=e["bytes"])
            for e in status["cases"]
        ],
        "Array inventory",
    )
    require(
        read(dest / "head.json")["arrays_sha256"] == inventory["additional_omitted"][0]["sha256"],
        "Omitted head",
    )
    metas, derived_checks, derived_fd, fd_index = {}, [], [], {}
    for entry in status["cases"]:
        key = entry["key"]
        path = dest / "cases" / (key + ".json")
        require(sha(path) == entry["metadata_sha256"], "Case metadata")
        meta = read(path)
        require(
            meta["key"] == key and meta["arrays_sha256"] == entry["arrays_sha256"], "Case binding"
        )
        require(
            meta["prefix_unchanged"]
            and meta["first_source_policy"]["activation_calls"] == 1
            and meta["first_source_policy"]["False_primal_exact"],
            "Prefix and False-gate witness",
        )
        require(meta["counts"] == {n: v // 32 for n, v in counts.items()}, "Per-case work")
        for left, right in (
            ("FF", "FM"),
            ("MF", "MM"),
            ("FF_plus_delta", "FM_plus_delta"),
            ("MF_plus_delta", "MM_plus_delta"),
        ):
            require(
                len(meta["source_path_hashes"][left]) == 27
                and meta["source_path_hashes"][left] == meta["source_path_hashes"][right],
                "Causal source trajectory",
            )
        derived_checks.extend(dict(key=key, **r) for r in meta["checks"])
        idx = {(r["corner"], r["direction"], str(r["step"])): r for r in meta["derivatives"]}
        labels = {n for _, n, _ in idx}
        require(
            len(labels) == 3
            and "delta" in labels
            and sum(n.startswith("coordinate_") and 0 <= int(n[11:]) < 1536 for n in labels) == 2,
            "Derivative directions",
        )
        require(
            set(idx)
            == {(c, n, s) for c in CORNERS for n in labels for s in ("0.05", "0.025", "halving")},
            "Projected derivative grid",
        )
        for (c, n, s), r in idx.items():
            if s == "halving":
                error = abs(idx[c, n, "0.05"]["FD"] - idx[c, n, "0.025"]["FD"])
                bound = 0.002 + 0.03 * abs(r["AD"])
                close(r["FD"], idx[c, n, "0.025"]["FD"], "Halved FD")
            else:
                close(
                    r["FD"],
                    (margin(r["plus_scores"]) - margin(r["minus_scores"])) / (2 * float(s)),
                    "Central FD",
                )
                error, bound = abs(r["AD"] - r["FD"]), 0.001 + 0.02 * abs(r["AD"])
            close(r["error"], error, "FD error")
            close(r["bound"], bound, "FD bound")
            require(r["passes"] and error <= bound, "Cached FD passes")
            fd_index[key, c, n, s] = r
        derived_fd.extend(dict(key=key, **r) for r in meta["derivatives"])
        metas[key] = meta
    for name, generated in (("checks", derived_checks), ("projected_derivatives", derived_fd)):
        published = rows(dest, name)
        require(len(published) == len(generated), "Original record coverage")
        for row, expected in zip(published, generated, strict=True):
            require(all(row[k] == str(v) for k, v in expected.items()), "Original generated record")
    checks = rows(dest, "checks")
    for r in checks:
        decision(r)
    failures = [r for r in checks if not boolean(r, "passes")]
    require(
        failures == qa["original_failures"]
        and len(failures) == 1
        and failures[0]["name"] == "FM_source_adjoint",
        "Retained failure",
    )
    raw = rows(dest, "raw_coordinates")
    require(
        len(raw) == 160
        and audit["raw_original_coordinate_checks"] == dict(total=160, failed=160, bound=1e-4),
        "Raw failures retained",
    )
    for r in raw:
        require(
            not boolean(r, "passes") and number(r, "error") > number(r, "original_bound") == 1e-4,
            "Raw failed decision",
        )
    trace, boundary, fullfd = (
        rows(qdest, n) for n in ("all_trace_checks", "boundary_checks", "full_shape_FD")
    )
    require(
        len(trace) == 3456
        and {(r["key"], r["corner"], int(r["input_before_block"])) for r in trace}
        == {(k, c, b) for k in keys for c in CORNERS for b in range(2, 29)},
        "Full trace grid",
    )
    for r in trace:
        decision(dict(error=r["maximum_error"], bound=r["bound"], passes=r["passes"]))
        require(
            boolean(r, "used_boundary") == (int(r["input_before_block"]) == 2), "Boundary selection"
        )
    failed_trace = [r for r in trace if not boolean(r, "passes")]
    for key in keys:
        for corner in CORNERS:
            maximum = max(
                number(r, "maximum_error")
                for r in trace
                if r["key"] == key and r["corner"] == corner
            )
            original = next(
                r for r in checks if r["key"] == key and r["name"] == corner + "_source_adjoint"
            )
            close(maximum, number(original, "error"), "Original whole-trace discrepancy")
            require(number(original, "bound") == 1e-4, "Original trace bound")
    require(
        len(failed_trace) == qa["intermediate_trace_failures"] == 1
        and int(failed_trace[0]["input_before_block"]) == 16,
        "Unqualified intermediate state retained",
    )
    require(
        len(boundary) == qa["boundary_checks"] == 128
        and {(r["key"], r["corner"]) for r in boundary} == {(k, c) for k in keys for c in CORNERS},
        "Boundary coverage",
    )
    for r in boundary:
        decision(r)
        require(
            number(r, "bound") == 1e-4 and boolean(r, "passes"), "Boundary original bound passes"
        )
        t = next(
            x
            for x in trace
            if x["key"] == r["key"] and x["corner"] == r["corner"] and x["used_boundary"] == "True"
        )
        close(number(r, "error"), number(t, "maximum_error"), "Boundary from trace")
    close(
        qa["maximum_boundary_error"],
        max(number(r, "error") for r in boundary),
        "Maximum boundary discrepancy",
    )
    full_ad = {}
    require(len(fullfd) == qa["full_shape_projected_checks"] == 1152, "Full-shape FD coverage")
    require(
        {(r["key"], r["corner"], r["direction"], r["step"]) for r in fullfd} == set(fd_index),
        "Full-shape FD grid",
    )
    for r in fullfd:
        k, c, n, s = r["key"], r["corner"], r["direction"], r["step"]
        cached = fd_index[k, c, n, s]
        ad = number(r, "AD")
        close(number(r, "FD"), cached["FD"], "Same saved finite scores")
        require(abs(ad - cached["AD"]) <= 1e-4, "Boundary AD implementations")
        if s == "halving":
            err = abs(fd_index[k, c, n, "0.05"]["FD"] - fd_index[k, c, n, "0.025"]["FD"])
            bound = 0.002 + 0.03 * abs(ad)
        else:
            err, bound = abs(ad - cached["FD"]), 0.001 + 0.02 * abs(ad)
        close(number(r, "error"), err, "Full-shape FD error")
        close(number(r, "bound"), bound, "Full-shape FD bound")
        require(boolean(r, "passes") and err <= bound, "Full-shape FD qualified")
        full_ad[k, c, n] = ad
    scores = {r["key"]: r["scores"] for r in read(dest / "scores.json")["cases"]}
    require(set(scores) == set(keys), "Finite score coverage")
    native_old = {(r["key"], r["arm"]): r for r in rows(old / "native-phase", "native")}

    def native_margin(k):
        return margin(scores[k]["reference_F_scores"]) + number(native_old[k, "C"], "target_margin")

    def effects(k):
        return {
            c: margin(scores[k][c + "_plus_delta_scores"]) - margin(scores[k][c + "_scores"])
            for c in CORNERS
        }

    hypo, ledger, readouts, native = (
        rows(qdest, n) for n in ("hypotheses", "ledger", "readouts", "native")
    )
    require(
        len(hypo) == 192
        and {(r["key"], r["arm"], r["gradient_path"]) for r in hypo}
        == {(k, a, p) for k in keys for a in ("S", "R", "ADD") for p in ("cached", "full_shape")},
        "Hypothesis grid",
    )
    for r in hypo:
        n, rel, cos = vector(r)
        es = effects(r["key"])
        d, se, re = (es[c] - es["FF"] for c in ("MM", "MF", "FM"))
        pred = {"S": se, "R": re, "ADD": se + re}[r["arm"]]
        for name, v in (
            ("finite_target", d),
            ("finite_prediction", pred),
            ("finite_error", abs(pred - d)),
        ):
            close(number(r, name), v, "Finite factorial effect")
        vp, ep = n >= 0.001 and rel <= 0.25 and cos >= 0.95, abs(pred - d) <= 0.005 + 0.25 * abs(d)
        for name, v in dict(
            vector_pass=vp,
            finite_pass=ep,
            passes=vp and ep,
            finite_material_eligible=abs(d) >= 0.02,
            material_pass=vp and ep and abs(d) >= 0.02,
        ).items():
            require(boolean(r, name) == v, "Frozen joint classification")
    for path in ("cached", "full_shape"):
        for arm in ("S", "R", "ADD"):
            rr = [r for r in hypo if r["gradient_path"] == path and r["arm"] == arm]
            layouts = {
                layout: sum(boolean(r, "passes") for r in rr if r["block"] == layout)
                for layout in ("dialogue", "table", "user_rule", "xml")
            }
            computed = dict(
                passes=sum(boolean(r, "passes") for r in rr),
                layouts=layouts,
                population_pass=sum(boolean(r, "passes") for r in rr) >= 24
                and all(n >= 6 for n in layouts.values()),
                material_passes=sum(boolean(r, "material_pass") for r in rr),
                median_relative=statistics.median(number(r, "relative") for r in rr),
                median_cosine=statistics.median(number(r, "cosine") for r in rr),
                median_finite_error=statistics.median(number(r, "finite_error") for r in rr),
            )
            require(
                computed == qa["hypotheses"][path][arm], "Population denominators and summaries"
            )
    ix = {(r["key"], r["arm"], r["gradient_path"]): r for r in hypo}
    disagreements = [
        dict(key=k, arm=a)
        for k in keys
        for a in ("S", "R", "ADD")
        if any(
            ix[k, a, "cached"][n] != ix[k, a, "full_shape"][n]
            for n in ("vector_pass", "finite_pass", "passes", "material_pass")
        )
    ]
    require(
        disagreements == qa["classification_disagreements"], "Implementation disagreement inventory"
    )
    require(
        len(readouts) == 256
        and {(r["key"], r["corner"], r["gradient_path"]) for r in readouts}
        == {(k, c, p) for k in keys for c in CORNERS for p in ("cached", "full_shape")},
        "Scalar grid",
    )
    for r in readouts:
        k, c, p = r["key"], r["corner"], r["gradient_path"]
        q = fd_index[k, c, "delta", "0.05"]["AD"] if p == "cached" else full_ad[k, c, "delta"]
        y, e = native_margin(k) - margin(scores[k]["reference_M_scores"]), effects(k)[c]
        for name, v in dict(
            directional=q,
            finite_effect=e,
            local_error=abs(q - e),
            native_target=y,
            native_error=abs(q - y),
        ).items():
            close(number(r, name), v, "Scalar effect formula")
        for name, v in dict(
            native_scalar_pass=abs(q - y) <= 0.1,
            native_large_error=abs(y) > 0.1,
            predicted_large_error=abs(q) > 0.1,
        ).items():
            require(boolean(r, name) == v, "Scalar classification")
    require(
        len(native) == 128
        and {(r["key"], r["corner"]) for r in native} == {(k, c) for k in keys for c in CORNERS},
        "Native grid",
    )
    for r in native:
        met = [vector(r, p) for p in ("receiving_", "verdict_")]
        predicted = margin(scores[r["key"]][r["corner"] + "_plus_delta_scores"])
        target = native_margin(r["key"])
        close(number(r, "predicted_margin"), predicted, "Native predicted margin")
        close(number(r, "target_margin"), target, "Native reference margin")
        close(number(r, "margin_error"), abs(predicted - target), "Native margin error")
        require(
            boolean(r, "passes")
            == (
                all(n >= 0.001 and rel <= 0.05 and cos >= 0.99 for n, rel, cos in met)
                and abs(predicted - target) <= 0.1
            ),
            "Original full native criterion",
        )
    require(
        len(ledger) == 64
        and {(r["key"], r["gradient_path"]) for r in ledger}
        == {(k, p) for k in keys for p in ("cached", "full_shape")},
        "Ledger grid",
    )
    for r in ledger:
        es = effects(r["key"])
        d, se, re = (es[c] - es["FF"] for c in ("MM", "MF", "FM"))
        for name, v in dict(
            full_finite_shift=d,
            source_finite_shift=se,
            receiving_finite_shift=re,
            mixed_finite_shift=d - se - re,
        ).items():
            close(number(r, name), v, "Signed finite ledger")
        close(
            sum(number(r, n + "_signed_projection") for n in ("source", "receiving", "mixed")),
            1.0,
            "Signed vector ledger",
        )
        close(
            sum(number(r, n + "_directional_shift") for n in ("source", "receiving", "mixed")),
            number(r, "full_directional_shift"),
            "Signed directional ledger",
        )
        close(
            number(r, "mixed_norm_fraction"),
            number(ix[r["key"], "ADD", r["gradient_path"]], "relative"),
            "Interaction is additive error",
        )
    require(
        qa["native_counts"]
        == {c: sum(boolean(r, "passes") for r in native if r["corner"] == c) for c in CORNERS},
        "Native summary",
    )
    require(
        qa["native_scalar_counts"]
        == {
            p: {
                c: sum(
                    boolean(r, "native_scalar_pass")
                    for r in readouts
                    if r["gradient_path"] == p and r["corner"] == c
                )
                for c in CORNERS
            }
            for p in ("cached", "full_shape")
        },
        "Scalar summary",
    )
    require(
        qa["material_eligible"]
        == sum(
            boolean(r, "finite_material_eligible")
            for r in hypo
            if r["arm"] == "S" and r["gradient_path"] == "cached"
        ),
        "Material denominator",
    )
    plot = root / "docs/figures/source-receiver-readout-287"
    prov, summary = read(plot / "provenance.json"), read(plot / "summary.json")
    require(
        prov["script_sha256"] == sha(root / "research/boolean_coupling/describe287.py")
        and prov["original_audit_sha256"] == sha(dest / "audit.json")
        and prov["qualifier_audit_sha256"] == sha(qdest / "audit.json")
        and prov["csv_sha256"] == qa["csv_sha256"],
        "Figure inputs",
    )
    require(
        prov["all_cases"] == summary["all_cases"] == 32 and prov["selected_or_omitted_cases"] == 0,
        "Figure cohort",
    )
    for n, h in prov["figures_sha256"].items():
        require(sha(plot / n) == h, "Figure hash")
    require(
        summary["hypotheses"] == qa["hypotheses"]
        and summary["native_counts"] == qa["native_counts"]
        and not summary["original_numeric_gate_passed"],
        "Figure results and failure",
    )
    stopped = read(dest / "retained-zero-case-stop/outcome.json")
    sd = read(dest / "retained-zero-case-stop/design.json")
    require(
        stopped["completed_cases"]
        == stopped["ordinary_F_forwards"]
        == stopped["scientific_case_calls"]
        == 0
        and sd["source_hashes"] == design["source_hashes"]
        and sd["numerical_bounds"] == design["numerical_bounds"],
        "Unchanged zero-case retry",
    )
    for name in ("287", "287b"):
        require(
            read(dest / "cpu-qualification" / name / "preflight-summary.json")["passed"],
            "CPU qualification",
        )
    return dict(
        passed=True,
        original_gate="FAILED, retained",
        qualification="Boundary only, posthoc",
        scientific_counts=qa["hypotheses"]["cached"],
        native_counts=qa["native_counts"],
        boundary="Scalar records/sufficient statistics, not omitted arrays, model replication, learned circuit or generalQB solution.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    print(json.dumps(verify(a.root.resolve()), indent=2))
