"""Audit saved286 qualification, source geometry and scalar native-response tests."""

import argparse
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .cache272 import save_json
from .checkpoint238 import locate, read_head
from .composition280 import digest
from .readout286 import ROOT, margin, split
from .run272 import stamp
from .run286 import COMPONENTS, REF_FILES, cpu_head


def main(run, reference):
    assert not (run / "audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert (
        design["experiment"] == 286 and design["retrospective"] and status["status"] == "complete"
    )
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    assert set(design["reference_hashes"]) == set(REF_FILES)
    for n, h in design["reference_hashes"].items():
        assert sha(reference / n) == h
    hd = read(run / "head.json")
    assert sha(run / "head.npz") == hd["arrays_sha256"]
    with np.load(run / "head.npz") as saved:
        head = {n: saved[n] for n in saved.files}
    if not design["synthetic"]:
        folder, checkpoint = locate(ROOT)
        assert checkpoint == design["checkpoint"]
        expected_head = read_head(folder)
        for n, field in (("norm", "final_norm"), ("weight", "final_head"), ("eps", "final_eps")):
            assert np.array_equal(head[n], expected_head[field])
    forecasts = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    natives = {e["key"]: e for e in read(reference / "native-status.json")["cases"]}
    bounds = design["numerical_bounds"]
    assert bounds == dict(
        replay_coordinate=0.002,
        replay_relative=0.0001,
        replay_score=0.001,
        projected_AD_FD_absolute=0.001,
        projected_AD_FD_relative=0.02,
        FD_halving_absolute=0.002,
        FD_halving_relative=0.03,
        CPU64_head=0.005,
    )
    assert design["FD_steps"] == [0.05, 0.025]
    assert len(status["cases"]) == len(design["cases"]) in (4, 32)
    assert [r["key"] for r in status["cases"]] == [c["key"] for c in design["cases"]]
    assert status["raw_bytes"] == sum(r["bytes"] for r in status["cases"]) <= 32 * 1024**2
    cases = {c["key"]: c for c in design["cases"]}
    bank, checks, projected, raw, numeric_head_checks = {}, [], [], [], []
    for e in status["cases"]:
        p = run / "cases" / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and p.stat().st_size == e["bytes"]
        assert sha(p.with_suffix(".json")) == e["metadata_sha256"]
        m = read(p.with_suffix(".json"))
        assert m["arrays_sha256"] == e["arrays_sha256"] and m["prefix_unchanged"]
        assert m["first_source_policy"]["activation_calls"] == 1
        assert m["first_source_policy"]["False_primal_exact"] is True
        assert m["old_native_opened_only_after_defined_predictions_and_components"]
        assert m["counts"] == dict(full_F=1, first_block=1, suffix_forwards=52, backwards=2)
        assert len(m["derivatives"]) == 30
        for r in m["checks"]:
            assert r["passes"] is True and 0 <= r["error"] <= r["bound"]
            checks.append(dict(key=e["key"], **r))
            if r["name"].endswith("_raw_coordinate"):
                raw.append(
                    dict(
                        key=e["key"],
                        name=r["name"],
                        error=r["error"],
                        original_bound=0.0001,
                        passes=bool(r["error"] <= 0.0001),
                    )
                )
        with np.load(p) as saved:
            d = {n: saved[n] for n in saved.files}
        assert all(np.isfinite(a).all() for a in d.values())
        assert np.array_equal(d["delta"], d["C_source"].astype(float) - d["M_source"].astype(float))
        assert m["forecast_entry"] == forecasts[e["key"]] and m["native_entry"] == natives[e["key"]]
        for folder_name, original, arms in (
            ("forecasts", forecasts[e["key"]], ("F", "C", "M")),
            ("native", natives[e["key"]], ("T",)),
        ):
            ref = reference / folder_name / (e["key"] + ".npz")
            assert sha(ref) == original["arrays_sha256"]
            assert sha(ref.with_suffix(".json")) == original["metadata_sha256"]
            with np.load(ref) as saved:
                if folder_name == "forecasts":
                    assert m["first_source_policy"]["input_gate_sha256"] == digest(saved["F_gate"])
                    assert m["first_source_policy"]["output_phi_sha256"] == digest(saved["F_phi"])
                for arm in arms:
                    assert np.array_equal(d["reference_" + arm + "_scores"], saved[arm + "_scores"])
                    assert np.array_equal(d[arm + "_source"], saved[arm + "_source"])
        for anchor in ("F", "M"):
            assert np.array_equal(d["g" + anchor + "_source_trace"][0], d["g" + anchor])
            assert d["g" + anchor + "_source_trace"].shape[1:] == d["delta"].shape
            assert (
                d["g" + anchor + "_tail_norm_trace"].shape
                == d["g" + anchor + "_source_trace"].shape[:1]
            )
            pvec, ovec = split(d["g" + anchor], d["delta"])
            assert np.array_equal(pvec, d["parallel_" + anchor + "_component"])
            assert np.array_equal(ovec, d["orthogonal_" + anchor + "_component"])
            assert np.max(abs(pvec + ovec - d["delta"])) <= 1e-12
        fd_index = {(r["anchor"], r["direction"], r["step"]): r for r in m["derivatives"]}
        coords = np.random.default_rng(int(e["key"][:16], 16)).choice(
            len(d["delta"]), 2, replace=False
        )
        expected_directions = {
            "delta",
            "parallel_F",
            "orthogonal_F",
            *("coordinate_" + str(i) for i in coords),
        }
        assert len(fd_index) == 30
        assert set(fd_index) == {
            (a, n, step)
            for a in ("F", "M")
            for n in expected_directions
            for step in (0.05, 0.025, "halving")
        }
        for r in m["derivatives"]:
            if r["direction"] == "delta":
                v = d["delta"]
            elif r["direction"] in ("parallel_F", "orthogonal_F"):
                v = d[r["direction"] + "_component"]
            else:
                v = np.zeros_like(d["delta"])
                v[int(r["direction"].split("_")[-1])] = 1.0
            ad = float(np.dot(d["g" + r["anchor"]].astype(float), v))
            assert ad == r["AD"] and r["passes"] and 0 <= r["error"] <= r["bound"]
            if r["step"] == "halving":
                assert r["bound"] == 0.002 + 0.03 * abs(ad)
                one = fd_index[r["anchor"], r["direction"], 0.05]["FD"]
                two = fd_index[r["anchor"], r["direction"], 0.025]["FD"]
                assert r["FD"] == two and r["error"] == abs(one - two)
            else:
                assert r["step"] in (0.05, 0.025)
                assert len(r["plus_scores"]) == len(r["minus_scores"]) == 2
                assert margin(r["plus_scores"]) == r["plus_margin"]
                assert margin(r["minus_scores"]) == r["minus_margin"]
                assert r["FD"] == (r["plus_margin"] - r["minus_margin"]) / (2 * r["step"])
                assert r["error"] == abs(ad - r["FD"])
                assert r["bound"] == 0.001 + 0.02 * abs(ad)
            projected.append(dict(key=e["key"], **r))
        for name in ("F", "M", "C", "T", *COMPONENTS):
            error = float(
                np.max(abs(cpu_head(head, d[name + "_final"]) - d[name + "_scores"].astype(float)))
            )
            assert error <= 0.005
            numeric_head_checks.append(dict(key=e["key"], arm=name, error=error, bound=0.005))
        bank[e["key"]] = d
    # Compute scientific comparisons only after every saved numeric condition passes.
    predictions, interventions, geometry, traces = [], [], [], []
    for key, d in bank.items():
        group = cases[key]["group"]
        y = margin(d["reference_T_scores"]) - margin(d["reference_M_scores"])
        finite = margin(d["reference_C_scores"]) - margin(d["reference_M_scores"])
        source_error = margin(d["reference_T_scores"]) - margin(d["reference_C_scores"])
        for anchor in ("F", "M"):
            predicted = float(np.dot(d["g" + anchor].astype(float), d["delta"]))
            error = abs(predicted - y)
            transport = finite - predicted
            assert abs((y - predicted) - source_error - transport) <= 1e-10
            predictions.append(
                dict(
                    key=key,
                    **group,
                    anchor=anchor,
                    target_effect=y,
                    prediction=predicted,
                    source_prediction_error=source_error,
                    transport_error=transport,
                    error=error,
                    native_margin_pass=bool(error <= 0.1),
                    baseline_native_margin_pass=bool(abs(y) <= 0.1),
                    target_large_error=bool(abs(y) > 0.1),
                    predicted_large_error=bool(abs(predicted) > 0.1),
                    material_eligible=bool(abs(y) >= 0.02),
                    material_pass=bool(abs(y) >= 0.02 and error <= 0.005 + 0.25 * abs(y)),
                )
            )
        target_finite = margin(d["C_scores"]) - margin(d["M_scores"])
        effects = {}
        for arm in COMPONENTS:
            value = margin(d[arm + "_scores"]) - margin(d["M_scores"])
            effects[arm] = value
            interventions.append(
                dict(
                    key=key,
                    **group,
                    arm=arm,
                    effect=value,
                    native_target=y,
                    finite_target=target_finite,
                    native_error=abs(value - y),
                    finite_error=abs(value - target_finite),
                    native_margin_pass=bool(abs(value - y) <= 0.1),
                    finite_material_eligible=bool(abs(target_finite) >= 0.02),
                    finite_material_pass=bool(
                        abs(target_finite) >= 0.02
                        and abs(value - target_finite) <= 0.005 + 0.25 * abs(target_finite)
                    ),
                    component_norm=float(np.linalg.norm(d[arm + "_component"])),
                    component_norm_fraction=float(np.linalg.norm(d[arm + "_component"]))
                    / max(float(np.linalg.norm(d["delta"])), 1e-30),
                )
            )
        for index, (a, b) in enumerate(
            zip(d["gF_source_trace"].astype(float), d["gM_source_trace"].astype(float), strict=True)
        ):
            traces.append(
                dict(
                    key=key,
                    **group,
                    receiving_block=index + 2,
                    source_gradient_cosine=float(np.dot(a, b))
                    / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30),
                    source_gradient_norm_F=float(np.linalg.norm(a)),
                    source_gradient_norm_M=float(np.linalg.norm(b)),
                    tail_gradient_norm_F=float(d["gF_tail_norm_trace"][index]),
                    tail_gradient_norm_M=float(d["gM_tail_norm_trace"][index]),
                )
            )
        gf, gm = d["gF"].astype(float), d["gM"].astype(float)
        geometry.append(
            dict(
                key=key,
                **group,
                gradient_cosine=float(np.dot(gf, gm))
                / max(float(np.linalg.norm(gf) * np.linalg.norm(gm)), 1e-30),
                gradient_norm_F=float(np.linalg.norm(gf)),
                gradient_norm_M=float(np.linalg.norm(gm)),
                target_finite=target_finite,
                interaction_F=target_finite - effects["parallel_F"] - effects["orthogonal_F"],
                interaction_M=target_finite - effects["parallel_M"] - effects["orthogonal_M"],
            )
        )
    outputs = dict(
        checks=checks,
        projected_derivatives=projected,
        raw_coordinates=raw,
        heads=numeric_head_checks,
        predictions=predictions,
        interventions=interventions,
        geometry=geometry,
        source_sensitivity_trace=traces,
    )
    for name, rows in outputs.items():
        table(run / (name + ".csv"), rows)
    summary = {}
    for a in ("F", "M"):
        rows = [r for r in predictions if r["anchor"] == a]
        summary[a] = dict(
            native_margin_passes=sum(r["native_margin_pass"] for r in rows),
            material_passes=sum(r["material_pass"] for r in rows),
            median_absolute_error=float(np.median([r["error"] for r in rows])),
            maximum_absolute_error=max(r["error"] for r in rows),
            confusion={
                n: sum(
                    bool(r["target_large_error"]) is t and bool(r["predicted_large_error"]) is p
                    for r in rows
                )
                for n, t, p in (
                    ("TP", True, True),
                    ("FN", True, False),
                    ("FP", False, True),
                    ("TN", False, False),
                )
            },
        )
    save_json(
        run / "audit.json",
        dict(
            status="passed",
            experiment=286,
            retrospective=True,
            design_sha256=sha(run / "design.json"),
            status_sha256=sha(run / "status.json"),
            source_sha256=sha(Path(__file__)),
            required_numeric_checks=len(checks) + len(projected) + len(numeric_head_checks),
            projected_derivative_checks=len(projected),
            raw_original_coordinate_checks=dict(
                total=len(raw), failed=sum(not r["passes"] for r in raw), bound=0.0001
            ),
            margin_material_eligible=sum(
                r["material_eligible"] for r in predictions if r["anchor"] == "F"
            ),
            baseline_M_margin_passes=sum(
                r["baseline_native_margin_pass"] for r in predictions if r["anchor"] == "F"
            ),
            predictions=summary,
            counts=status["counts"],
            csv_sha256={n + ".csv": sha(run / (n + ".csv")) for n in outputs},
            ended=stamp(),
            boundary="Projected derivatives and old-context native margin/causal source-residual comparisons only. Not a complete Jacobian or native vector predictor, learned feature/circuit, fresh confirmation, priority or measured speedup.",
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference-285", type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.reference_285.resolve())
