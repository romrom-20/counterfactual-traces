"""Posthoc boundary-only qualification; retain 287's original failed audit."""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .audit287 import metrics
from .cache272 import save_json
from .factor287 import CORNERS
from .factor287 import SOURCES as ORIGINAL
from .readout286 import ROOT, margin
from .run272 import stamp
from .run286 import entry_data

SOURCES = (
    *ORIGINAL,
    "research/boolean_coupling/qualify287b.py",
    "research/boolean_coupling/preflight287b.py",
    "docs/experiments/287b-boundary-qualification.md",
)


def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def vector_stats(p, t):
    p, t = np.asarray(p, float).ravel(), np.asarray(t, float).ravel()
    return dict(
        prediction_squared_norm=float(p @ p),
        target_squared_norm=float(t @ t),
        prediction_target_dot=float(p @ t),
        error_squared_norm=float((p - t) @ (p - t)),
    )


def qualify(run):
    design, status, audit = (read(run / (n + ".json")) for n in ("design", "status", "audit"))
    assert design["experiment"] == 287 and status["status"] == "complete"
    assert audit["design_sha256"] == sha(run / "design.json")
    assert audit["status_sha256"] == sha(run / "status.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "research/boolean_coupling/audit287.py")
    assert design["numerical_bounds"]["source_adjoint"] == 1e-4
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    for name, h in audit["csv_sha256"].items():
        assert sha(run / name) == h
    original = records(run / "checks.csv")
    original_fd = records(run / "projected_derivatives.csv")
    failures = [r for r in original + original_fd if r["passes"] != "True"]
    assert len(failures) == audit["failed_checks"]
    assert all(r in original and r["name"].endswith("_source_adjoint") for r in failures)
    assert len(original) + len(original_fd) == audit["required_checks"]
    bank, checks, fd, trace = {}, [], [], []
    for entry, case in zip(status["cases"], design["cases"], strict=True):
        assert entry["key"] == case["key"]
        key = entry["key"]
        a = entry_data(run, "cases", entry)
        meta = read(run / "cases" / (key + ".json"))
        assert all(np.isfinite(v).all() for v in a.values())
        for corner in CORNERS:
            cached, full = a[corner + "_source_trace"], a[corner + "_full_source_trace"]
            assert np.array_equal(cached[0], a[corner + "_gradient"])
            error = np.abs(cached.astype(float) - full.astype(float))
            checks.append(dict(key=key, corner=corner, error=float(error[0].max()), bound=1e-4))
            for i, row in enumerate(error):
                trace.append(
                    dict(
                        key=key,
                        corner=corner,
                        input_before_block=i + 2,
                        maximum_error=float(row.max()),
                        bound=1e-4,
                        passes=bool(row.max() <= 1e-4),
                        used_boundary=i == 0,
                    )
                )
            directions = {"delta": a["delta"]}
            for record in meta["derivatives"]:
                name = record["direction"]
                if name.startswith("coordinate_") and name not in directions:
                    d = np.zeros_like(a["delta"], dtype=float)
                    d[int(name[11:])] = 1.0
                    directions[name] = d
            assert len(directions) == 3
            idx = {(r["corner"], r["direction"], r["step"]): r for r in meta["derivatives"]}
            for name, direction in directions.items():
                ad = float(full[0].astype(float) @ direction)
                for step in (0.05, 0.025, "halving"):
                    r = idx[corner, name, step]
                    if step == "halving":
                        err = abs(idx[corner, name, 0.05]["FD"] - idx[corner, name, 0.025]["FD"])
                        bound = 0.002 + 0.03 * abs(ad)
                    else:
                        actual = (margin(r["plus_scores"]) - margin(r["minus_scores"])) / (2 * step)
                        assert actual == r["FD"]
                        err, bound = abs(ad - actual), 0.001 + 0.02 * abs(ad)
                    fd.append(
                        dict(
                            key=key,
                            corner=corner,
                            direction=name,
                            step=step,
                            AD=ad,
                            FD=r["FD"],
                            error=err,
                            bound=bound,
                            passes=bool(np.isfinite(err) and err <= bound),
                        )
                    )
        bank[key] = a
    for r in checks:
        r["passes"] = bool(r["error"] <= r["bound"])
    assert all(r["passes"] for r in checks + fd), "Boundary or projected FD qualification failed"
    assert sum(not r["passes"] for r in trace) == len(failures)
    return bank, checks, fd, trace, failures


def scientific(run, reference, bank, out):
    design = read(run / "design.json")
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ns = {e["key"]: e for e in read(reference / "native-status.json")["cases"]}
    hypotheses, ledger, readouts, native = [], [], [], []
    for case in design["cases"]:
        key, group, a = case["key"], case["group"], bank[case["key"]]
        f, t = entry_data(reference, "forecasts", fs[key]), entry_data(reference, "native", ns[key])
        effect = {
            c: margin(a[c + "_plus_delta_scores"]) - margin(a[c + "_scores"]) for c in CORNERS
        }
        de, se, re = (effect[c] - effect["FF"] for c in ("MM", "MF", "FM"))
        ie = de - se - re
        y = margin(t["T_scores"]) - margin(f["M_scores"])
        for path in ("cached", "full_shape"):
            g = {
                c: a[c + ("_gradient" if path == "cached" else "_full_source_trace")].astype(float)
                for c in CORNERS
            }
            if path == "full_shape":
                g = {c: x[0] for c, x in g.items()}
            v, s, r = (g[c] - g["FF"] for c in ("MM", "MF", "FM"))
            mixed = v - s - r
            for arm, p, pe in (("S", s, se), ("R", r, re), ("ADD", s + r, se + re)):
                met = metrics(p, v)
                err = abs(pe - de)
                vp = met["norm"] >= 0.001 and met["relative"] <= 0.25 and met["cosine"] >= 0.95
                ep = err <= 0.005 + 0.25 * abs(de)
                hypotheses.append(
                    dict(
                        key=key,
                        **group,
                        gradient_path=path,
                        arm=arm,
                        **met,
                        **vector_stats(p, v),
                        finite_target=de,
                        finite_prediction=pe,
                        finite_error=err,
                        vector_pass=vp,
                        finite_pass=ep,
                        passes=vp and ep,
                        finite_material_eligible=abs(de) >= 0.02,
                        material_pass=vp and ep and abs(de) >= 0.02,
                    )
                )
            row = dict(
                key=key,
                **group,
                gradient_path=path,
                full_gradient_shift_norm=float(np.linalg.norm(v)),
                source_input_step_norm=float(
                    np.linalg.norm(a["M_tail"][0].astype(float) - a["F_tail"][0])
                ),
                receiving_input_step_norm=float(
                    np.linalg.norm(a["M_tail"][1:].astype(float) - a["F_tail"][1:])
                ),
                full_finite_shift=de,
                source_finite_shift=se,
                receiving_finite_shift=re,
                mixed_finite_shift=ie,
            )
            for name, x in (("source", s), ("receiving", r), ("mixed", mixed)):
                row[name + "_norm_fraction"] = float(np.linalg.norm(x)) / max(
                    float(np.linalg.norm(v)), 1e-30
                )
                row[name + "_signed_projection"] = float(x @ v) / max(float(v @ v), 1e-30)
                row[name + "_directional_shift"] = float(x @ a["delta"])
            row["full_directional_shift"] = float(v @ a["delta"])
            ledger.append(row)
            for c in CORNERS:
                q = float(g[c] @ a["delta"])
                readouts.append(
                    dict(
                        key=key,
                        **group,
                        gradient_path=path,
                        corner=c,
                        directional=q,
                        finite_effect=effect[c],
                        local_error=abs(q - effect[c]),
                        native_target=y,
                        native_error=abs(q - y),
                        native_scalar_pass=abs(q - y) <= 0.1,
                        native_large_error=abs(y) > 0.1,
                        predicted_large_error=abs(q) > 0.1,
                    )
                )
        for c in CORNERS:
            actual = a[c + "_plus_delta_final"].astype(float) - f["F_final"].astype(float)
            target = t["T_final"].astype(float) - f["F_final"].astype(float)
            receiving, verdict = metrics(actual[1:], target[1:]), metrics(actual[-1], target[-1])
            me = abs(margin(a[c + "_plus_delta_scores"]) - margin(t["T_scores"]))
            ok = (
                all(
                    m["norm"] >= 0.001 and m["relative"] <= 0.05 and m["cosine"] >= 0.99
                    for m in (receiving, verdict)
                )
                and me <= 0.1
            )
            native.append(
                dict(
                    key=key,
                    **group,
                    corner=c,
                    **{"receiving_" + k: v for k, v in receiving.items()},
                    **{"verdict_" + k: v for k, v in verdict.items()},
                    **{
                        "receiving_" + k: v for k, v in vector_stats(actual[1:], target[1:]).items()
                    },
                    **{"verdict_" + k: v for k, v in vector_stats(actual[-1], target[-1]).items()},
                    predicted_margin=margin(a[c + "_plus_delta_scores"]),
                    target_margin=margin(t["T_scores"]),
                    margin_error=me,
                    passes=ok,
                )
            )
    for name, rows in (
        ("hypotheses", hypotheses),
        ("ledger", ledger),
        ("readouts", readouts),
        ("native", native),
    ):
        table(out / (name + ".csv"), rows)
    result = {}
    for path in ("cached", "full_shape"):
        result[path] = {}
        for arm in ("S", "R", "ADD"):
            rows = [r for r in hypotheses if r["gradient_path"] == path and r["arm"] == arm]
            layouts = {
                layout: sum(r["passes"] for r in rows if r["block"] == layout)
                for layout in sorted({r["block"] for r in rows})
            }
            result[path][arm] = dict(
                passes=sum(r["passes"] for r in rows),
                layouts=layouts,
                population_pass=sum(r["passes"] for r in rows) >= 24
                and all(n >= 6 for n in layouts.values()),
                material_passes=sum(r["material_pass"] for r in rows),
                median_relative=float(np.median([r["relative"] for r in rows])),
                median_cosine=float(np.median([r["cosine"] for r in rows])),
                median_finite_error=float(np.median([r["finite_error"] for r in rows])),
            )
    indexed = {(r["key"], r["arm"], r["gradient_path"]): r for r in hypotheses}
    differences = []
    for case in design["cases"]:
        for arm in ("S", "R", "ADD"):
            ca, fu = (indexed[case["key"], arm, p] for p in ("cached", "full_shape"))
            if any(
                ca[n] != fu[n] for n in ("vector_pass", "finite_pass", "passes", "material_pass")
            ):
                differences.append(dict(key=case["key"], arm=arm))
    return dict(
        hypotheses=result,
        classification_disagreements=differences,
        material_eligible=sum(
            r["finite_material_eligible"]
            for r in hypotheses
            if r["gradient_path"] == "cached" and r["arm"] == "S"
        ),
        native_counts={c: sum(r["passes"] for r in native if r["corner"] == c) for c in CORNERS},
        native_scalar_counts={
            p: {
                c: sum(
                    r["native_scalar_pass"]
                    for r in readouts
                    if r["gradient_path"] == p and r["corner"] == c
                )
                for c in CORNERS
            }
            for p in ("cached", "full_shape")
        },
    )


def main(run, reference, out):
    assert not out.exists()
    expected = read(run / "design.json")["reference_hashes"]
    for key, name in (
        ("285_design", "design.json"),
        ("285_forecast_status", "forecast-status.json"),
        ("285_native_status", "native-status.json"),
    ):
        assert sha(reference / name) == expected[key]
    out.mkdir()
    (out / "sources").mkdir()
    source_hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment="287b",
            posthoc_numerical_scope=True,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=source_hashes,
            original_audit_sha256=sha(run / "audit.json"),
            boundary_coordinate_bound=1e-4,
            new_model_calls=0,
            scientific_thresholds="Unchanged frozen287",
            began=stamp(),
        ),
    )
    bank, checks, fd, trace, failures = qualify(run)
    for name, rows in (
        ("boundary_checks", checks),
        ("full_shape_FD", fd),
        ("all_trace_checks", trace),
    ):
        table(out / (name + ".csv"), rows)
    result = scientific(run, reference, bank, out)
    summary = dict(
        experiment="287b",
        status="qualified_boundary_only",
        retrospective=True,
        original_numeric_gate_passed=not failures,
        original_failures=failures,
        original_audit_sha256=sha(run / "audit.json"),
        boundary_checks=len(checks),
        full_shape_projected_checks=len(fd),
        maximum_boundary_error=max(r["error"] for r in checks),
        intermediate_trace_failures=sum(not r["passes"] for r in trace),
        new_model_calls=0,
        **result,
        boundary="Posthoc boundary scope; original full-trace failure retained. All later receivers remain adaptive; no learned circuit, fresh forecast, efficiency or generalQB claim.",
        ended=stamp(),
    )
    summary["csv_sha256"] = {p.name: sha(p) for p in out.glob("*.csv")}
    save_json(out / "audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference-285", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.reference_285.resolve(), a.out.resolve())
