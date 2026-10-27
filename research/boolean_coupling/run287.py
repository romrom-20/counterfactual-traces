"""Run the retrospective source/receiver-state factorial on the old 285 cohort."""

import argparse
import gc
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .core225b import array, load_model
from .factor287 import BOUNDS, CORNERS, SOURCES, Crossed, corners
from .readout286 import ROOT, finite_difference, release
from .run272 import stamp
from .run285 import model_signature, runtime
from .run286 import cpu_head, entry_data, head_data


def main(reference, readout, qualifier, out, model=None):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 768 * 1024**2
    old, rd, qd = (read(path / "design.json") for path in (reference, readout, qualifier))
    assert old["experiment"] == 285 and rd["experiment"] == 286 and qd["experiment"] == "286b"
    assert (
        read(readout / "audit.json")["status"]
        == read(qualifier / "audit.json")["status"]
        == "passed"
    )
    assert (
        old["installed_sources"]
        == rd["installed_sources"]
        == qd["installed_sources"]
        == installed_sources()
    )
    synthetic = model is not None
    assert synthetic is old["synthetic"] is rd["synthetic"] is qd["synthetic"]
    if synthetic:
        assert model_signature(model) == old["checkpoint"]
    for name, h in qd["source_hashes"].items():
        assert sha(ROOT / name) == h
    assert rd["cases"] == qd["cases"] and len(rd["cases"]) == (4 if synthetic else 32)
    out.mkdir()
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in hashes:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    save_json(
        out / "design.json",
        dict(
            experiment=287,
            retrospective=True,
            synthetic=synthetic,
            cases=rd["cases"],
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=old["checkpoint"],
            reference_hashes={
                "285_design": sha(reference / "design.json"),
                "285_forecast_status": sha(reference / "forecast-status.json"),
                "285_native_status": sha(reference / "native-status.json"),
                "286_design": sha(readout / "design.json"),
                "286_status": sha(readout / "status.json"),
                "286_audit": sha(readout / "audit.json"),
                "286b_design": sha(qualifier / "design.json"),
                "286b_audit": sha(qualifier / "audit.json"),
            },
            numerical_bounds=BOUNDS,
            FD_steps=[0.05, 0.025],
            arms=list(CORNERS),
            source_first=True,
            boundary_node="residual after block1",
            primary=dict(
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
            secondary="ADD interaction accounting; native vector/margin calibration for the four source+d finite continuations; no rescue of primaries.",
            boundary="Old32 contexts; state interventions, not a learned primitive circuit. All later source and receiver work remains finite and adapts. No new True gradients, fresh forecast, efficiency or generalQB claim.",
            began=stamp(),
        ),
    )
    model = load_model(True) if model is None else model
    assert not any(p.requires_grad for p in model.parameters())
    save_json(out / "runtime.json", runtime(model, synthetic))
    head = head_data(model)
    np.savez_compressed(out / "head.npz", **head)
    save_json(out / "head.json", dict(arrays_sha256=sha(out / "head.npz")))
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    rs = {e["key"]: e for e in read(readout / "status.json")["cases"]}
    entries, counts, raw_bytes = [], [], 0
    for case in rd["cases"]:
        release(model)
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        key = case["key"]
        f, r = entry_data(reference, "forecasts", fs[key]), entry_data(readout, "cases", rs[key])
        engine = Crossed(model, case)
        bases = corners(engine.false_tail, engine.first(f["M_write"]))
        data = dict(
            F_tail=array(bases["FF"][0]),
            M_tail=array(bases["MM"][0]),
            delta=r["delta"],
            ordinary_F_final=engine.false_final,
            ordinary_F_scores=engine.false_scores,
            **{"ordinary_F_" + name: value for name, value in engine.false_nodes.items()},
        )
        checks, derivatives, paths = [], [], {}

        def check(name, actual, expected, bound=0.0):
            err = float(np.max(np.abs(np.asarray(actual, float) - np.asarray(expected, float))))
            checks.append(
                dict(
                    name=name,
                    error=err,
                    bound=bound,
                    passes=bool(np.isfinite(err) and err <= bound),
                )
            )

        check("ordinary_F_source", data["F_tail"][0], f["F_source"])
        check("ordinary_F_final", engine.false_final, f["F_final"])
        check("ordinary_F_scores", engine.false_scores, f["F_scores"])
        check("first_M_source", data["M_tail"][0], f["M_source"])
        check(
            "delta_source_identity",
            data["delta"],
            f["C_source"].astype(float) - f["M_source"].astype(float),
        )
        for name in ("gate", "phi", "write"):
            check("ordinary_F_" + name, engine.false_nodes[name], f["F_" + name])
        coords = np.random.default_rng(int(key[:16], 16)).choice(len(r["delta"]), 2, replace=False)
        directions = {"delta": r["delta"]}
        for coordinate in coords:
            direction = np.zeros_like(r["delta"])
            direction[coordinate] = 1.0
            directions["coordinate_" + str(coordinate)] = direction
        for corner in CORNERS:
            base = bases[corner]
            grad, scores, final, trace, norms = engine.gradient(base)
            release(model)
            paths[corner] = list(engine.last_source_path)
            data.update(
                {
                    corner + "_" + n: v
                    for n, v in dict(
                        gradient=grad,
                        scores=scores,
                        final=final,
                        source_trace=trace,
                        tail_norm_trace=norms,
                    ).items()
                }
            )
            full = engine.full_shape.gradient(base)
            release(model)
            data.update({corner + "_full_" + n: v for n, v in full.items()})
            check(corner + "_source_adjoint", trace, full["source_trace"], 1e-4)
            check(
                corner + "_projected_response",
                np.dot(grad.astype(float), r["delta"]),
                np.dot(full["source_trace"][0].astype(float), r["delta"]),
                1e-4,
            )
            check(corner + "_reverse_primal", full["reverse_primal_errors"], 0.0, 1e-4)
            check(corner + "_full_prefix", full["prefix_errors"], 0.0)
            check(corner + "_cached_coordinate", final, full["final"], 0.002)
            relative = np.linalg.norm(final.astype(float) - full["final"].astype(float)) / max(
                np.linalg.norm(full["final"].astype(float)), 1e-30
            )
            check(corner + "_cached_relative", relative, 0.0, 1e-4)
            check(corner + "_cached_scores", scores, full["scores"], 0.001)
            check(corner + "_cached_CPU64_head", cpu_head(head, final), scores, 0.005)
            check(corner + "_full_CPU64_head", cpu_head(head, full["final"]), full["scores"], 0.005)
            if corner in ("FF", "MM"):
                anchor = corner[0]
                check(corner + "_full_reference_final", full["final"], f[anchor + "_final"], 1e-4)
                check(
                    corner + "_full_reference_scores", full["scores"], f[anchor + "_scores"], 1e-4
                )
                check(corner + "_old_gradient", grad, r["g" + anchor], 1e-4)
            for name, direction in directions.items():
                derivatives.extend(
                    dict(corner=corner, **row)
                    for row in finite_difference(engine, base, grad, direction, name)
                )
            bumped_final, bumped_scores = engine.finite(
                base, array(base[0, 0]).astype(float) + r["delta"]
            )
            release(model)
            paths[corner + "_plus_delta"] = list(engine.last_source_path)
            data.update(
                {
                    corner + "_plus_delta_final": bumped_final,
                    corner + "_plus_delta_scores": bumped_scores,
                }
            )
            check(
                corner + "_plus_delta_CPU64_head",
                cpu_head(head, bumped_final),
                bumped_scores,
                0.005,
            )
        # Causality: source trajectories cannot depend on later receiving rows.
        for left, right in (
            ("FF", "FM"),
            ("MF", "MM"),
            ("FF_plus_delta", "FM_plus_delta"),
            ("MF_plus_delta", "MM_plus_delta"),
        ):
            check(left + "_source_path_identity_" + right, float(paths[left] != paths[right]), 0.0)
        check("MM_plus_delta_old_C_final", data["MM_plus_delta_final"], f["C_final"], 0.002)
        check("MM_plus_delta_old_C_scores", data["MM_plus_delta_scores"], f["C_scores"], 0.001)
        engine.unchanged()
        data.update(
            reference_F_scores=f["F_scores"],
            reference_M_scores=f["M_scores"],
            reference_C_scores=f["C_scores"],
        )
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **data)
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 64 * 1024**2
        count = dict(
            full_F=1,
            first_block=engine.first_blocks,
            suffix_forwards=engine.calls,
            backwards=engine.backwards,
            full_shape_block_forwards=engine.full_shape.forward_calls,
            block_recomputations=engine.full_shape.reverse_calls,
            block_VJPs=engine.full_shape.reverse_calls,
            head_VJPs=engine.full_shape.head_reverse_calls,
        )
        save_json(
            path.with_suffix(".json"),
            dict(
                key=key,
                arrays_sha256=sha(path),
                checks=checks,
                derivatives=derivatives,
                source_path_hashes=paths,
                prefix_seals=engine.seals,
                first_source_policy=engine.first_policy,
                prefix_unchanged=True,
                forecast_entry=fs[key],
                readout_entry=rs[key],
                counts=count,
            ),
        )
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(path),
                metadata_sha256=sha(path.with_suffix(".json")),
                bytes=path.stat().st_size,
            )
        )
        counts.append(count)
        print(
            "287 factorial saved",
            len(entries),
            "/",
            len(rd["cases"]),
            "numeric failures",
            sum(not row["passes"] for row in checks + derivatives),
            flush=True,
        )
        del engine, bases, f, r, data, full, base
        gc.collect()
        release(model)
    save_json(
        out / "status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=raw_bytes,
            counts={name: sum(row[name] for row in counts) for name in counts[0]},
            ended=stamp(),
            outcomes_not_yet_audited=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-285", type=Path, required=True)
    p.add_argument("--readout-286", type=Path, required=True)
    p.add_argument("--qualifier-286b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(
        a.reference_285.resolve(),
        a.readout_286.resolve(),
        a.qualifier_286b.resolve(),
        a.out.resolve(),
    )
