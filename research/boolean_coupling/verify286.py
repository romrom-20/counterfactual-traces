"""Dependency-free checks of 286 scalar records, derivatives, failures and seals.

No omitted arrays are loaded. CI verifies this record, not a checkpoint replica.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

try:
    from .verify285 import (
        boolean,
        close,
        declared_sources,
        number,
        read,
        relative,
        require,
        rows,
        sha,
    )
except ImportError:
    from verify285 import (
        boolean,
        close,
        declared_sources,
        number,
        read,
        relative,
        require,
        rows,
        sha,
    )

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/false-readout-286")
COMPONENTS = ("parallel_F", "orthogonal_F", "parallel_M", "orthogonal_M", "random0", "random1")


def margin(scores):
    require(
        len(scores) == 2 and all(isinstance(v, (float, int)) for v in scores), "Candidate scores"
    )
    return scores[1] - scores[0]


def numeric(row):
    error, bound = number(row, "error"), number(row, "bound")
    require(0 <= error <= bound and boolean(row, "passes"), "Required numerical check failed")


def manifest(root, dest):
    record = read(dest / "release-manifest.json")
    require(record["experiment"] == 286 and record["retrospective"], "Release identity")
    files = record["files"]
    for name, h in files.items():
        require(sha(relative(root, name)) == h, "Release hash: " + name)
    actual = {
        str(p.relative_to(root))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "release-manifest.json"
    }
    declared = {name for name in files if name.startswith(str(DEST) + "/")}
    require(actual == declared, "Exact result file inventory")
    require(
        all("drafts" not in Path(name).parts and not name.endswith(".npz") for name in files),
        "Private/array release entry",
    )
    return record


def source_seal(root, design, module):
    require(
        set(design["source_hashes"]) == set(declared_sources(root, module)),
        "Scientific source coverage",
    )
    for name, h in design["source_hashes"].items():
        require(sha(root / name) == h, "Scientific source: " + name)
        frozen = subprocess.check_output(
            ["git", "show", design["source_commit"] + ":" + name], cwd=root
        )
        import hashlib

        require(hashlib.sha256(frozen).hexdigest() == h, "Frozen commit source: " + name)
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", design["source_commit"], "HEAD"],
            cwd=root,
            capture_output=True,
        ).returncode
        == 0,
        "Freeze ancestry",
    )


def entries(dest, design, status):
    require(
        [c["key"] for c in design["cases"]] == [e["key"] for e in status["cases"]],
        "Ordered case coverage",
    )
    inventory = read(dest / "omitted-array-inventory.json")
    require(
        inventory["case_bytes"]
        == status["raw_bytes"]
        == sum(e["bytes"] for e in status["cases"])
        <= 32 * 1024**2,
        "Raw storage budget",
    )
    require(
        inventory["arrays"]
        == [
            dict(path="cases/" + e["key"] + ".npz", sha256=e["arrays_sha256"], bytes=e["bytes"])
            for e in status["cases"]
        ],
        "Omitted array inventory",
    )
    result = {}
    for e in status["cases"]:
        path = dest / "cases" / (e["key"] + ".json")
        require(sha(path) == e["metadata_sha256"], "Metadata hash")
        meta = read(path)
        require(
            meta["key"] == e["key"] and meta["arrays_sha256"] == e["arrays_sha256"], "Case identity"
        )
        require(
            meta["first_source_policy"]["activation_calls"] == 1
            and meta["first_source_policy"]["False_primal_exact"],
            "False activation witness",
        )
        result[e["key"]] = meta
    return result


def verify(root):
    dest = root / DEST
    release = manifest(root, dest)
    design, status, audit = (
        read(dest / name) for name in ("design.json", "status.json", "audit.json")
    )
    require(
        design["experiment"] == 286 and design["retrospective"] and not design["synthetic"],
        "Production retrospective design",
    )
    require(len(design["cases"]) == 32 and not design["sentinel"], "All old contexts")
    require(
        design["numerical_bounds"]
        == dict(
            replay_coordinate=0.002,
            replay_relative=0.0001,
            replay_score=0.001,
            projected_AD_FD_absolute=0.001,
            projected_AD_FD_relative=0.02,
            FD_halving_absolute=0.002,
            FD_halving_relative=0.03,
            CPU64_head=0.005,
        ),
        "Original numerical bounds",
    )
    require(design["FD_steps"] == [0.05, 0.025], "Frozen FD steps")
    source_seal(root, design, "readout286")
    require(status["status"] == "complete" and audit["status"] == "passed", "Audited completion")
    require(
        sha(dest / "design.json") == audit["design_sha256"]
        and sha(dest / "status.json") == audit["status_sha256"],
        "Audit input seals",
    )
    require(
        audit["source_sha256"] == design["source_hashes"]["research/boolean_coupling/audit286.py"],
        "Original audit source",
    )
    for name, h in audit["csv_sha256"].items():
        require(sha(dest / name) == h, "Audit CSV seal")
    require(
        audit["counts"]
        == status["counts"]
        == dict(full_F=32, first_block=32, suffix_forwards=1664, backwards=64),
        "Actual computation counts",
    )
    old = root / "results/native-structure-285"
    for name, h in design["reference_hashes"].items():
        folder = "native-phase" if name.startswith("native") else "pre-native"
        require(sha(old / folder / name) == h, "285 reference seal")
    require(
        design["cases"] == read(old / "pre-native/design.json")["cases"],
        "Same previously observed cases",
    )
    metas = entries(dest, design, status)
    scores = {r["key"]: r for r in read(dest / "scores.json")["cases"]}
    require(set(scores) == set(metas), "Scalar score coverage")
    prior_scores = {(r["key"], r["arm"]): r for r in rows(old / "pre-native", "forecast_scores")}
    prior_native = {(r["key"], r["arm"]): r for r in rows(old / "native-phase", "native")}
    derivatives, numeric_rows, raw_rows = [], [], []
    fd_deltas = {}
    for key, meta in metas.items():
        require(
            meta["counts"] == dict(full_F=1, first_block=1, suffix_forwards=52, backwards=2),
            "Case computation counts",
        )
        require(
            meta["prefix_unchanged"]
            and meta["old_native_opened_only_after_defined_predictions_and_components"],
            "Recorded information order",
        )
        for r in meta["checks"]:
            numeric({**r, "passes": str(r["passes"])})
            numeric_rows.append(dict(key=key, **r))
            if r["name"].endswith("_raw_coordinate"):
                raw_rows.append(
                    dict(
                        key=key,
                        name=r["name"],
                        error=r["error"],
                        original_bound=0.0001,
                        passes=r["error"] <= 0.0001,
                    )
                )
        index = {(r["anchor"], r["direction"], r["step"]): r for r in meta["derivatives"]}
        require(len(index) == 30, "Derivative case coverage")
        labels = {direction for _, direction, _ in index}
        require(
            len(labels) == 5 and {"delta", "parallel_F", "orthogonal_F"} <= labels,
            "Derivative directions",
        )
        require(
            len([n for n in labels if n.startswith("coordinate_") and 0 <= int(n[11:]) < 1536])
            == 2,
            "Sampled coordinates",
        )
        require(
            set(index)
            == {
                (a, n, step)
                for a in ("F", "M")
                for n in labels
                for step in (0.05, 0.025, "halving")
            },
            "Derivative grid",
        )
        for (anchor, label, step), r in index.items():
            ad = r["AD"]
            if step == "halving":
                error = abs(index[anchor, label, 0.05]["FD"] - index[anchor, label, 0.025]["FD"])
                close(r["FD"], index[anchor, label, 0.025]["FD"], "Halved FD")
                bound = 0.002 + 0.03 * abs(ad)
            else:
                close(r["plus_margin"], margin(r["plus_scores"]), "FD plus scores")
                close(r["minus_margin"], margin(r["minus_scores"]), "FD minus scores")
                close(r["FD"], (r["plus_margin"] - r["minus_margin"]) / (2 * step), "Central FD")
                error, bound = abs(ad - r["FD"]), 0.001 + 0.02 * abs(ad)
            close(r["error"], error, "Derivative error")
            close(r["bound"], bound, "Derivative bound")
            numeric({**r, "passes": str(r["passes"])})
            derivatives.append(dict(key=key, **r))
        for anchor in ("F", "M"):
            fd_deltas[key, anchor] = index[anchor, "delta", 0.05]["AD"]
        for arm in ("F", "M", "C"):
            previous = prior_scores[key, arm]
            close(
                scores[key]["reference_scores"][arm][0],
                number(previous, "false_score"),
                "Reference False logit",
            )
            close(
                scores[key]["reference_scores"][arm][1],
                number(previous, "true_score"),
                "Reference True logit",
            )
        close(
            margin(scores[key]["reference_scores"]["T"])
            - margin(scores[key]["reference_scores"]["F"]),
            number(prior_native[key, "C"], "target_margin"),
            "Native T-minus-F margin displacement",
        )
    require(
        len(derivatives) == audit["projected_derivative_checks"] == 960,
        "Projected derivative count",
    )
    for name, generated in (
        ("checks", numeric_rows),
        ("raw_coordinates", raw_rows),
        ("projected_derivatives", derivatives),
    ):
        published = rows(dest, name)
        require(len(published) == len(generated), "Generated numeric count")
        for row, expected in zip(published, generated, strict=True):
            require(all(row[k] == str(v) for k, v in expected.items()), "Generated numeric row")
    raw = rows(dest, "raw_coordinates")
    require(
        audit["raw_original_coordinate_checks"]
        == dict(total=len(raw), failed=sum(not boolean(r, "passes") for r in raw), bound=0.0001),
        "Original failed diagnostics retained",
    )
    heads = rows(dest, "heads")
    for r in heads:
        require(
            number(r, "bound") == 0.005 and 0 <= number(r, "error") <= 0.005,
            "Finite CPU64 head check",
        )
    require(
        len(heads) == 320
        and audit["required_numeric_checks"]
        == len(numeric_rows) + len(derivatives) + len(heads)
        == 2592,
        "Required numerical count",
    )
    predictions, interventions = rows(dest, "predictions"), rows(dest, "interventions")
    require(
        {(r["key"], r["anchor"]) for r in predictions}
        == {(k, a) for k in metas for a in ("F", "M")}
        and len(predictions) == 64,
        "Prediction coverage",
    )
    for r in predictions:
        key, anchor = r["key"], r["anchor"]
        ref = scores[key]["reference_scores"]
        target = margin(ref["T"]) - margin(ref["M"])
        finite = margin(ref["C"]) - margin(ref["M"])
        predicted = fd_deltas[key, anchor]
        error = abs(target - predicted)
        for name, v in dict(
            target_effect=target,
            prediction=predicted,
            error=error,
            source_prediction_error=target - finite,
            transport_error=finite - predicted,
        ).items():
            close(number(r, name), v, "Prediction formula: " + name)
        for name, v in dict(
            native_margin_pass=error <= 0.1,
            baseline_native_margin_pass=abs(target) <= 0.1,
            target_large_error=abs(target) > 0.1,
            predicted_large_error=abs(predicted) > 0.1,
            material_eligible=abs(target) >= 0.02,
            material_pass=abs(target) >= 0.02 and error <= 0.005 + 0.25 * abs(target),
        ).items():
            require(boolean(r, name) == v, "Prediction decision: " + name)
    for anchor in ("F", "M"):
        rr = [r for r in predictions if r["anchor"] == anchor]
        summary = dict(
            native_margin_passes=sum(boolean(r, "native_margin_pass") for r in rr),
            material_passes=sum(boolean(r, "material_pass") for r in rr),
            median_absolute_error=statistics.median(number(r, "error") for r in rr),
            maximum_absolute_error=max(number(r, "error") for r in rr),
            confusion={
                n: sum(
                    boolean(r, "target_large_error") == t
                    and boolean(r, "predicted_large_error") == p
                    for r in rr
                )
                for n, t, p in (
                    ("TP", True, True),
                    ("FN", True, False),
                    ("FP", False, True),
                    ("TN", False, False),
                )
            },
        )
        require(summary == audit["predictions"][anchor], "Recomputed prediction summary")
    require(
        len(interventions) == 192
        and {(r["key"], r["arm"]) for r in interventions}
        == {(k, a) for k in metas for a in COMPONENTS},
        "Causal intervention coverage",
    )
    for r in interventions:
        s = scores[r["key"]]
        effect = margin(s["runtime_scores"][r["arm"]]) - margin(s["runtime_scores"]["M"])
        finite = margin(s["runtime_scores"]["C"]) - margin(s["runtime_scores"]["M"])
        target = margin(s["reference_scores"]["T"]) - margin(s["reference_scores"]["M"])
        for name, value in dict(
            effect=effect,
            finite_target=finite,
            native_target=target,
            native_error=abs(effect - target),
            finite_error=abs(effect - finite),
        ).items():
            close(number(r, name), value, "Intervention formula")
        for name, value in dict(
            native_margin_pass=abs(effect - target) <= 0.1,
            finite_material_eligible=abs(finite) >= 0.02,
            finite_material_pass=abs(finite) >= 0.02
            and abs(effect - finite) <= 0.005 + 0.25 * abs(finite),
        ).items():
            require(boolean(r, name) == value, "Causal endpoint")
    by_arm = {(r["key"], r["arm"]): r for r in interventions}
    geometry = rows(dest, "geometry")
    require(len(geometry) == 32 and {r["key"] for r in geometry} == set(metas), "Geometry coverage")
    for r in geometry:
        key = r["key"]
        for anchor in ("F", "M"):
            close(
                number(r, "interaction_" + anchor),
                number(r, "target_finite")
                - number(by_arm[key, "parallel_" + anchor], "effect")
                - number(by_arm[key, "orthogonal_" + anchor], "effect"),
                "Finite interaction",
            )
        for arm in ("random0", "random1"):
            close(
                number(by_arm[key, arm], "component_norm"),
                number(by_arm[key, "parallel_F"], "component_norm"),
                "Matched random norm",
            )
    plot = root / "docs/figures/false-readout-286"
    provenance, summary = read(plot / "provenance.json"), read(plot / "summary.json")
    require(
        provenance["script_sha256"] == sha(root / "research/boolean_coupling/describe286.py"),
        "Plotter provenance",
    )
    require(
        provenance["audit_sha256"] == sha(dest / "audit.json")
        and provenance["csv_sha256"] == audit["csv_sha256"],
        "Plot input seals",
    )
    for name, h in provenance["figures_sha256"].items():
        require(sha(plot / name) == h, "Figure hash")
    require(
        summary["predictions"] == audit["predictions"] and summary["all_cases"] == 32,
        "Figure prediction summary",
    )
    close(
        summary["median_gradient_cosine"],
        statistics.median(number(r, "gradient_cosine") for r in geometry),
        "Figure geometry summary",
    )
    for arm in COMPONENTS:
        require(
            summary["finite_material_counts"][arm]
            == sum(boolean(r, "finite_material_pass") for r in interventions if r["arm"] == arm),
            "Figure causal counts",
        )
    qdest = dest / "full-shape-qualifier"
    qdesign, qstatus, qaudit = (
        read(qdest / name) for name in ("design.json", "status.json", "audit.json")
    )
    require(
        qdesign["experiment"] == "286b" and qdesign["numerical_only"] and not qdesign["synthetic"],
        "Numerical qualifier identity",
    )
    require(
        qdesign["cases"] == design["cases"]
        and qdesign["numerical_bounds"]
        == dict(full_replay=0.0001, source_adjoint=0.0001, projected_response=0.0001),
        "Qualifier cases and original bounds",
    )
    source_seal(root, qdesign, "qualify286b")
    require(
        release["scientific_sources"] == qdesign["source_hashes"], "Complete scientific source map"
    )
    entries(qdest, qdesign, qstatus)
    for name, h in qaudit["csv_sha256"].items():
        require(sha(qdest / name) == h, "Qualifier CSV hash")
    qrows = rows(qdest, "checks")
    failures = sum(not boolean(r, "passes") for r in qrows)
    require(
        len(qrows) == qaudit["checks"] == 512 and failures == qaudit["failed_checks"],
        "Qualifier numerical coverage",
    )
    require(qaudit["status"] == ("failed" if failures else "passed"), "Qualifier status")
    for r in qrows:
        require(
            boolean(r, "passes") == (0 <= number(r, "error") <= number(r, "bound")),
            "Qualifier recorded decision",
        )
        require(number(r, "bound") in (0, 0.0001), "Qualifier original numerical bound")
    layers = rows(qdest, "layers")
    require(
        len(layers) == 1728
        and {(r["key"], r["anchor"], int(r["receiving_block"])) for r in layers}
        == {(k, a, b) for k in metas for a in ("F", "M") for b in range(2, 29)},
        "All later source adjoints covered",
    )
    close(
        qaudit["maximum_source_adjoint_coordinate_error"],
        max(number(r, "maximum_coordinate_error") for r in layers),
        "Maximum source-adjoint discrepancy",
    )
    close(
        qaudit["maximum_projected_response_error"],
        max(number(r, "error") for r in qrows if r["name"].endswith("_projected_response")),
        "Maximum projection discrepancy",
    )
    require(
        provenance["qualifier_audit_sha256"] == sha(qdest / "audit.json")
        and summary["full_shape_qualification"] == qaudit,
        "Figure numerical qualification",
    )
    require(
        sha(qdest / "design.json") == qaudit["design_sha256"]
        and sha(qdest / "status.json") == qaudit["status_sha256"],
        "Qualifier audit seals",
    )
    require(
        qdesign["reference_hashes"]
        == dict(
            **{"285_design": sha(old / "pre-native/design.json")},
            **{
                "286_design": sha(dest / "design.json"),
                "286_status": sha(dest / "status.json"),
                "286_audit": sha(dest / "audit.json"),
            },
        ),
        "Qualifier references",
    )
    require(
        qstatus["counts"]
        == qaudit["counts"]
        == dict(
            full_F=32,
            first_block=32,
            full_shape_block_forwards=1728,
            block_recomputations=1728,
            block_VJPs=1728,
            head_VJPs=64,
        ),
        "Qualifier cost",
    )
    return dict(
        passed=True,
        retrospective=True,
        projected_derivative_checks=len(derivatives),
        original_raw_failures=sum(not boolean(r, "passes") for r in raw),
        predictions=audit["predictions"],
        full_shape_qualification=qaudit["status"],
        boundary="Published scalar records and hashes only; not a new model replication, full Jacobian, fresh forecast, learned circuit or general Question B solution.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))
