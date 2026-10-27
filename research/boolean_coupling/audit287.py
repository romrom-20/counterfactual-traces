"""Independent saved-array arithmetic for the frozen 287 state factorial."""

import argparse
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .cache272 import save_json
from .checkpoint238 import locate, read_head
from .composition280 import digest
from .factor287 import BOUNDS, CORNERS
from .readout286 import ROOT, margin
from .run272 import stamp
from .run286 import cpu_head, entry_data


def metrics(prediction, target):
    p, t = np.asarray(prediction, float).ravel(), np.asarray(target, float).ravel()
    pn, tn = float(np.linalg.norm(p)), float(np.linalg.norm(t))
    return dict(
        norm=tn,
        relative=float(np.linalg.norm(p - t)) / max(tn, 1e-30),
        cosine=float(np.dot(p, t)) / max(pn * tn, 1e-30),
    )


def main(run, reference, readout, qualifier):
    assert not (run / "audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert (
        design["experiment"] == 287 and design["retrospective"] and status["status"] == "complete"
    )
    assert design["numerical_bounds"] == BOUNDS and design["FD_steps"] == [0.05, 0.025]
    assert design["arms"] == list(CORNERS) and design["source_first"]
    assert design["primary"] == dict(
        target="gMM-gFF source-row margin adjoint shift",
        vector_norm_min=0.001,
        relative_max=0.25,
        cosine_min=0.95,
        finite_contrast_error_absolute=0.005,
        finite_contrast_error_relative=0.25,
        population_min=24,
        per_layout_min=6,
        hypotheses=["S", "R"],
    )
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    assert design["reference_hashes"] == {
        "285_design": sha(reference / "design.json"),
        "285_forecast_status": sha(reference / "forecast-status.json"),
        "285_native_status": sha(reference / "native-status.json"),
        "286_design": sha(readout / "design.json"),
        "286_status": sha(readout / "status.json"),
        "286_audit": sha(readout / "audit.json"),
        "286b_design": sha(qualifier / "design.json"),
        "286b_audit": sha(qualifier / "audit.json"),
    }
    assert [e["key"] for e in status["cases"]] == [c["key"] for c in design["cases"]]
    assert len(status["cases"]) == (4 if design["synthetic"] else 32)
    assert sum(e["bytes"] for e in status["cases"]) == status["raw_bytes"] <= 64 * 1024**2
    assert sha(run / "head.npz") == read(run / "head.json")["arrays_sha256"]
    with np.load(run / "head.npz") as h:
        head = {name: h[name] for name in h.files}
    if not design["synthetic"]:
        folder, checkpoint = locate(ROOT)
        assert checkpoint == design["checkpoint"]
        expected = read_head(folder)
        for name, field in (("norm", "final_norm"), ("weight", "final_head"), ("eps", "final_eps")):
            assert np.array_equal(head[name], expected[field])
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ns = {e["key"]: e for e in read(reference / "native-status.json")["cases"]}
    rs = {e["key"]: e for e in read(readout / "status.json")["cases"]}
    bank, checks, fd_rows, raw_rows, counts = {}, [], [], [], []
    max_adj, max_projection = 0.0, 0.0
    for entry, case in zip(status["cases"], design["cases"], strict=True):
        key = entry["key"]
        data = entry_data(run, "cases", entry)
        f, r = entry_data(reference, "forecasts", fs[key]), entry_data(readout, "cases", rs[key])
        meta = read(run / "cases" / (key + ".json"))
        assert meta["forecast_entry"] == fs[key] and meta["readout_entry"] == rs[key]
        assert meta["prefix_unchanged"] and meta["first_source_policy"]["activation_calls"] == 1
        assert meta["first_source_policy"]["False_primal_exact"]
        assert meta["first_source_policy"]["input_gate_sha256"] == digest(f["F_gate"])
        assert meta["first_source_policy"]["output_phi_sha256"] == digest(f["F_phi"])
        assert all(np.isfinite(value).all() for value in data.values())
        counts.append(meta["counts"])
        layers = len(data["FF_source_trace"])
        assert meta["counts"] == dict(
            full_F=1,
            first_block=1,
            suffix_forwards=56,
            backwards=4,
            full_shape_block_forwards=4 * layers,
            block_recomputations=4 * layers,
            block_VJPs=4 * layers,
            head_VJPs=4,
        )
        recalculated = {}

        def compare(name, actual, expected, bound=0.0):
            err = float(np.max(abs(np.asarray(actual, float) - np.asarray(expected, float))))
            recalculated[name] = dict(
                name=name, error=err, bound=bound, passes=bool(np.isfinite(err) and err <= bound)
            )

        compare("ordinary_F_source", data["F_tail"][0], f["F_source"])
        compare("ordinary_F_final", data["ordinary_F_final"], f["F_final"])
        compare("ordinary_F_scores", data["ordinary_F_scores"], f["F_scores"])
        compare("first_M_source", data["M_tail"][0], f["M_source"])
        compare(
            "delta_source_identity",
            data["delta"],
            f["C_source"].astype(float) - f["M_source"].astype(float),
        )
        for name in ("gate", "phi", "write"):
            compare("ordinary_F_" + name, data["ordinary_F_" + name], f["F_" + name])
        for corner in CORNERS:
            g, tr = data[corner + "_gradient"], data[corner + "_source_trace"]
            full = data[corner + "_full_source_trace"]
            assert tr.shape == full.shape and np.array_equal(tr[0], g)
            assert len(tr.shape) == 2 and tr.shape[1:] == data["delta"].shape
            compare(corner + "_source_adjoint", tr, full, 1e-4)
            compare(
                corner + "_projected_response",
                np.dot(g.astype(float), data["delta"]),
                np.dot(full[0].astype(float), data["delta"]),
                1e-4,
            )
            max_adj = max(max_adj, recalculated[corner + "_source_adjoint"]["error"])
            max_projection = max(
                max_projection, recalculated[corner + "_projected_response"]["error"]
            )
            compare(
                corner + "_reverse_primal", data[corner + "_full_reverse_primal_errors"], 0.0, 1e-4
            )
            compare(corner + "_full_prefix", data[corner + "_full_prefix_errors"], 0.0)
            final, ffinal = data[corner + "_final"], data[corner + "_full_final"]
            compare(corner + "_cached_coordinate", final, ffinal, 0.002)
            relative = np.linalg.norm(final.astype(float) - ffinal.astype(float)) / max(
                np.linalg.norm(ffinal.astype(float)), 1e-30
            )
            compare(corner + "_cached_relative", relative, 0.0, 1e-4)
            compare(
                corner + "_cached_scores",
                data[corner + "_scores"],
                data[corner + "_full_scores"],
                0.001,
            )
            compare(
                corner + "_cached_CPU64_head",
                cpu_head(head, final),
                data[corner + "_scores"],
                0.005,
            )
            compare(
                corner + "_full_CPU64_head",
                cpu_head(head, ffinal),
                data[corner + "_full_scores"],
                0.005,
            )
            compare(
                corner + "_plus_delta_CPU64_head",
                cpu_head(head, data[corner + "_plus_delta_final"]),
                data[corner + "_plus_delta_scores"],
                0.005,
            )
            if corner in ("FF", "MM"):
                anchor = corner[0]
                compare(corner + "_full_reference_final", ffinal, f[anchor + "_final"], 1e-4)
                compare(
                    corner + "_full_reference_scores",
                    data[corner + "_full_scores"],
                    f[anchor + "_scores"],
                    1e-4,
                )
                compare(corner + "_old_gradient", g, r["g" + anchor], 1e-4)
        for left, right in (
            ("FF", "FM"),
            ("MF", "MM"),
            ("FF_plus_delta", "FM_plus_delta"),
            ("MF_plus_delta", "MM_plus_delta"),
        ):
            assert len(meta["source_path_hashes"][left]) == layers
            compare(
                left + "_source_path_identity_" + right,
                float(meta["source_path_hashes"][left] != meta["source_path_hashes"][right]),
                0.0,
            )
        compare("MM_plus_delta_old_C_final", data["MM_plus_delta_final"], f["C_final"], 0.002)
        compare("MM_plus_delta_old_C_scores", data["MM_plus_delta_scores"], f["C_scores"], 0.001)
        assert set(recalculated) == {row["name"] for row in meta["checks"]}
        for row in meta["checks"]:
            new = recalculated[row["name"]]
            assert new["bound"] == row["bound"] and abs(new["error"] - row["error"]) <= 1e-10
            assert new["passes"] == row["passes"]
            checks.append(dict(key=key, **new))
            if (
                row["name"].endswith("_cached_coordinate")
                or row["name"] == "MM_plus_delta_old_C_final"
            ):
                raw_rows.append(
                    dict(
                        key=key,
                        name=row["name"],
                        error=new["error"],
                        original_bound=1e-4,
                        passes=new["error"] <= 1e-4,
                    )
                )
        coords = np.random.default_rng(int(key[:16], 16)).choice(
            len(data["delta"]), 2, replace=False
        )
        labels = {"delta", *("coordinate_" + str(c) for c in coords)}
        index = {(row["corner"], row["direction"], row["step"]): row for row in meta["derivatives"]}
        assert len(index) == len(meta["derivatives"]) == 36
        assert set(index) == {
            (c, n, s) for c in CORNERS for n in labels for s in (0.05, 0.025, "halving")
        }
        for (corner, name, step), row in index.items():
            direction = (
                data["delta"]
                if name == "delta"
                else np.eye(1, len(data["delta"]), int(name[11:]), dtype=float)[0]
            )
            ad = float(np.dot(data[corner + "_gradient"].astype(float), direction))
            assert row["AD"] == ad
            if step == "halving":
                error = abs(index[corner, name, 0.05]["FD"] - index[corner, name, 0.025]["FD"])
                bound = 0.002 + 0.03 * abs(ad)
                assert row["FD"] == index[corner, name, 0.025]["FD"]
            else:
                assert (
                    margin(row["plus_scores"]) == row["plus_margin"]
                    and margin(row["minus_scores"]) == row["minus_margin"]
                )
                assert row["FD"] == (row["plus_margin"] - row["minus_margin"]) / (2 * step)
                error = abs(ad - row["FD"])
                bound = 0.001 + 0.02 * abs(ad)
            assert row["error"] == error and row["bound"] == bound
            assert row["passes"] == bool(np.isfinite(error) and error <= bound)
            fd_rows.append(dict(key=key, **row))
        bank[key] = data
    assert status["counts"] == {name: sum(c[name] for c in counts) for name in counts[0]}
    for name, contents in (
        ("checks", checks),
        ("projected_derivatives", fd_rows),
        ("raw_coordinates", raw_rows),
    ):
        table(run / (name + ".csv"), contents)
    failed = sum(not row["passes"] for row in checks + fd_rows)
    summary = dict(
        experiment=287,
        retrospective=True,
        synthetic=design["synthetic"],
        status="failed" if failed else "passed",
        required_checks=len(checks) + len(fd_rows),
        failed_checks=failed,
        projected_derivative_checks=len(fd_rows),
        maximum_source_adjoint_error=max_adj,
        maximum_projected_response_error=max_projection,
        raw_original_coordinate_checks=dict(
            total=len(raw_rows), failed=sum(not row["passes"] for row in raw_rows), bound=1e-4
        ),
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"),
        audit_source_sha256=sha(Path(__file__)),
        counts=status["counts"],
        ended=stamp(),
    )
    if failed:
        summary["csv_sha256"] = {
            name + ".csv": sha(run / (name + ".csv"))
            for name in ("checks", "projected_derivatives", "raw_coordinates")
        }
        save_json(run / "audit.json", summary)
        raise AssertionError(("Numeric gate failed; no scientific interpretation", failed))
    hypotheses, ledger, readouts, native = [], [], [], []
    for case in design["cases"]:
        key, group = case["key"], case["group"]
        data = bank[key]
        g = {c: data[c + "_gradient"].astype(float) for c in CORNERS}
        effect = {
            c: margin(data[c + "_plus_delta_scores"]) - margin(data[c + "_scores"]) for c in CORNERS
        }
        v = g["MM"] - g["FF"]
        s = g["MF"] - g["FF"]
        r = g["FM"] - g["FF"]
        mixed = g["MM"] - g["MF"] - g["FM"] + g["FF"]
        assert np.max(abs(v - s - r - mixed)) < 1e-12
        de = effect["MM"] - effect["FF"]
        se = effect["MF"] - effect["FF"]
        re = effect["FM"] - effect["FF"]
        ie = effect["MM"] - effect["MF"] - effect["FM"] + effect["FF"]
        oldt = entry_data(reference, "native", ns[key])
        oldf = entry_data(reference, "forecasts", fs[key])
        y = margin(oldt["T_scores"]) - margin(oldf["M_scores"])
        for arm, pred, pe in (("S", s, se), ("R", r, re), ("ADD", s + r, se + re)):
            met = metrics(pred, v)
            error = abs(pe - de)
            vp = met["norm"] >= 0.001 and met["relative"] <= 0.25 and met["cosine"] >= 0.95
            ep = error <= 0.005 + 0.25 * abs(de)
            hypotheses.append(
                dict(
                    key=key,
                    **group,
                    arm=arm,
                    **met,
                    finite_target=de,
                    finite_prediction=pe,
                    finite_error=error,
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
            full_gradient_shift_norm=float(np.linalg.norm(v)),
            source_input_step_norm=float(
                np.linalg.norm(data["M_tail"][0].astype(float) - data["F_tail"][0].astype(float))
            ),
            receiving_input_step_norm=float(
                np.linalg.norm(data["M_tail"][1:].astype(float) - data["F_tail"][1:].astype(float))
            ),
            full_finite_shift=de,
            source_finite_shift=se,
            receiving_finite_shift=re,
            mixed_finite_shift=ie,
        )
        for name, component in (("source", s), ("receiving", r), ("mixed", mixed)):
            row[name + "_norm_fraction"] = float(np.linalg.norm(component)) / max(
                np.linalg.norm(v), 1e-30
            )
            row[name + "_signed_projection"] = float(np.dot(component, v)) / max(
                float(np.dot(v, v)), 1e-30
            )
            row[name + "_directional_shift"] = float(np.dot(component, data["delta"]))
        row["full_directional_shift"] = float(np.dot(v, data["delta"]))
        ledger.append(row)
        for c in CORNERS:
            q = float(np.dot(g[c], data["delta"]))
            readouts.append(
                dict(
                    key=key,
                    **group,
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
            actual = data[c + "_plus_delta_final"].astype(float) - oldf["F_final"].astype(float)
            target = oldt["T_final"].astype(float) - oldf["F_final"].astype(float)
            receiving, verdict = metrics(actual[1:], target[1:]), metrics(actual[-1], target[-1])
            me = abs(margin(data[c + "_plus_delta_scores"]) - margin(oldt["T_scores"]))
            passes = (
                all(
                    met["norm"] >= 0.001 and met["relative"] <= 0.05 and met["cosine"] >= 0.99
                    for met in (receiving, verdict)
                )
                and me <= 0.1
            )
            native.append(
                dict(
                    key=key,
                    **group,
                    corner=c,
                    **{"receiving_" + n: x for n, x in receiving.items()},
                    **{"verdict_" + n: x for n, x in verdict.items()},
                    margin_error=me,
                    passes=passes,
                )
            )
    outputs = dict(hypotheses=hypotheses, ledger=ledger, readouts=readouts, native=native)
    for name, contents in outputs.items():
        table(run / (name + ".csv"), contents)
    result = {}
    for arm in ("S", "R", "ADD"):
        rr = [r for r in hypotheses if r["arm"] == arm]
        layouts = {
            layout: sum(r["passes"] for r in rr if r["block"] == layout)
            for layout in sorted({r["block"] for r in rr})
        }
        result[arm] = dict(
            passes=sum(r["passes"] for r in rr),
            layouts=layouts,
            population_pass=sum(r["passes"] for r in rr) >= 24
            and all(n >= 6 for n in layouts.values()),
            material_passes=sum(r["material_pass"] for r in rr),
            median_relative=float(np.median([r["relative"] for r in rr])),
            median_cosine=float(np.median([r["cosine"] for r in rr])),
            median_finite_error=float(np.median([r["finite_error"] for r in rr])),
        )
    summary.update(
        hypotheses=result,
        material_eligible=sum(r["finite_material_eligible"] for r in hypotheses if r["arm"] == "S"),
        native_counts={c: sum(r["passes"] for r in native if r["corner"] == c) for c in CORNERS},
        native_scalar_counts={
            c: sum(r["native_scalar_pass"] for r in readouts if r["corner"] == c) for c in CORNERS
        },
        csv_sha256={
            name + ".csv": sha(run / (name + ".csv"))
            for name in ("checks", "projected_derivatives", "raw_coordinates", *outputs)
        },
        boundary="Retrospective boundary-state gradient/finite interventions. Later receiver adaptation remains finite. No learned primitive circuit, fresh forecast, efficiency or generalQB claim.",
    )
    save_json(run / "audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference-285", type=Path, required=True)
    p.add_argument("--readout-286", type=Path, required=True)
    p.add_argument("--qualifier-286b", type=Path, required=True)
    a = p.parse_args()
    main(
        a.run.resolve(),
        a.reference_285.resolve(),
        a.readout_286.resolve(),
        a.qualifier_286b.resolve(),
    )
