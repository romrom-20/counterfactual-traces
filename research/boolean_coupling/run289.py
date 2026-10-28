"""Run the frozen attention-interface diagnostic; producer has no True calls."""

import argparse
import gc
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .composition280 import digest
from .core225b import array, load_model
from .interface289 import ARMS, BOUNDS, PRIMARY, SOURCES, Interface
from .probes289 import primitive_records, probes
from .readout286 import ROOT, release
from .routing288 import finite_differences
from .run272 import stamp
from .run285 import model_signature, runtime
from .run286 import cpu_head, entry_data, head_data


def refs(reference, previous, head):
    return (
        {
            "285_" + n: sha(reference / (n + ".json"))
            for n in ("design", "forecast-status", "native-status")
        }
        | {"288_" + n: sha(previous / (n + ".json")) for n in ("design", "status", "audit")}
        | {"287_head": sha(head / "head.json")}
    )


def main(reference, previous, head_path, out, model=None):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 384 * 1024**2
    old, prior = read(reference / "design.json"), read(previous / "design.json")
    assert old["experiment"] == 285 and prior["experiment"] == 288
    assert read(previous / "audit.json")["status"] == "passed"
    assert old["installed_sources"] == prior["installed_sources"] == installed_sources()
    synthetic = model is not None
    assert old["synthetic"] is prior["synthetic"] is synthetic
    assert (model_signature(model) if synthetic else locate(ROOT)[1]) == old["checkpoint"]
    for name, h in prior["source_hashes"].items():
        assert sha(ROOT / name) == h
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    if not synthetic:
        for name, h in hashes.items():
            content = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
            assert hashlib.sha256(content).hexdigest() == h
    out.mkdir()
    (out / "sources").mkdir()
    (out / "cases").mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=289,
            retrospective=True,
            synthetic=synthetic,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=old["checkpoint"],
            cases=prior["cases"],
            reference_hashes=refs(reference, previous, head_path),
            arms=list(ARMS),
            factors=["source_encoding_norm", "qk_softmax_routing", "probability_value_mixing"],
            primary=PRIMARY,
            numerical_bounds=BOUNDS,
            FD_steps=[0.05, 0.025],
            FD_unit_delta=True,
            resource_guards_MiB=[384, 256, 96],
            began=stamp(),
            boundary="Retrospective conditional primitive families; source recurrence/head E, other receiving parents F. No circuit/native-predictor/generalQB claim.",
        ),
    )
    model = load_model(True) if model is None else model
    assert not any(p.requires_grad for p in model.parameters())
    save_json(out / "runtime.json", runtime(model, synthetic))
    assert sha(head_path / "head.npz") == read(head_path / "head.json")["arrays_sha256"]
    with np.load(head_path / "head.npz") as z:
        head = {n: z[n] for n in z.files}
    assert all(np.array_equal(v, head[n]) for n, v in head_data(model).items())
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ps = {e["key"]: e for e in read(previous / "status.json")["cases"]}
    entries, costs, raw_bytes = [], [], 0
    for case in prior["cases"]:
        release(model)
        assert shutil.disk_usage(out).free >= 256 * 1024**2
        key = case["key"]
        f, p = entry_data(reference, "forecasts", fs[key]), entry_data(previous, "cases", ps[key])
        engine = Interface(model, case, f["M_write"])
        delta = p["delta"].copy()
        unit = delta / np.linalg.norm(delta)
        data = dict(
            delta=delta,
            unit_delta=unit,
            ordinary_F_final=engine.false_final,
            ordinary_F_scores=engine.false_scores,
            initial_F=array(engine.tapes[0][0][0]),
            initial_E=array(engine.tapes[1][0][0]),
            F_gate=engine.false_nodes["gate"],
            F_phi=engine.false_nodes["phi"],
            norm_eps=np.asarray(model.config.rms_norm_eps),
        )
        checks, fd, paths = [], [], {}

        def check(name, actual, expected, bound=0.0):
            err = float(np.max(abs(np.asarray(actual, float) - np.asarray(expected, float))))
            checks.append(
                dict(
                    name=name,
                    error=err,
                    bound=bound,
                    passes=bool(np.isfinite(err) and err <= bound),
                )
            )

        check("ordinary_F_final", engine.false_final, p["ordinary_F_final"], 1e-4)
        check("ordinary_F_scores", engine.false_scores, p["ordinary_F_scores"], 1e-4)
        check("initial_F", data["initial_F"], p["initial_F"], 1e-4)
        check("initial_E", data["initial_E"], p["initial_E"], 1e-4)
        check("delta_identity", delta, f["C_source"].astype(float) - f["M_source"])
        for a in (0, 1):
            data[f"anchor{a}_final"] = array(engine.tapes[a][-1][0])
            data[f"anchor{a}_scores"] = engine.scores[a]
            check(f"anchor{a}_final", data[f"anchor{a}_final"], p[f"anchor{a}_final"], 1e-4)
            check(f"anchor{a}_scores", engine.scores[a], p[f"anchor{a}_scores"], 1e-4)
        grad, scores, offset, state = engine.reference_gradient()
        data.update(reference_gradient=grad, reference_scores=scores, reference_zero_offset=offset)
        check("reference_gradient", grad, p["111_gradient"], 1e-4)
        check("reference_primal_offset", offset, 0.0)
        check("reference_primal_scores", scores, engine.scores[1])
        final, score = engine.native_bump(1, delta)
        data.update(
            reference_finite_final=final,
            reference_finite_scores=score,
            reference_finite_offset=final.astype(float) - data["anchor1_final"].astype(float),
        )
        check("reference_old_finite", final, p["anchor1_native_bump_final"], 0.002)
        check("reference_old_scores", score, p["anchor1_native_bump_scores"], 0.001)
        check("reference_CPU64_head", cpu_head(head, final), score, 0.005)
        release(model)
        primitive, zero_probes = probes(engine, key)
        data.update(primitive)
        primitive_rows = primitive_records(data, float(data["norm_eps"]))
        for row in zero_probes:
            row.update(bound=0.0, passes=row["error"] == 0.0)
        coords = np.random.default_rng(int(key[:16], 16)).choice(len(delta), 2, replace=False)
        directions = {"unit_delta": unit}
        for c in coords:
            v = np.zeros_like(delta)
            v[c] = 1.0
            directions["coordinate_" + str(c)] = v
        trace_diagnostics = []
        for arm in ARMS:
            g = engine.gradient(arm)
            release(model)
            b = engine.blockwise(arm)
            release(model)
            data[arm + "_gradient"], data[arm + "_blockwise_gradient"] = (
                g["gradient"],
                b["gradient"],
            )
            data[arm + "_zero_offset"] = g["zero_offset"]
            data[arm + "_zero_raw_scores"] = g["zero_raw_scores"]
            data[arm + "_trace_difference"] = g["source_trace"].astype(float) - b[
                "source_trace"
            ].astype(float)
            check(arm + "_zero_offset", g["zero_offset"], 0.0)
            check(arm + "_zero_head", g["zero_raw_scores"], engine.scores[1])
            check(arm + "_boundary_gradient", g["gradient"], b["gradient"], 1e-4)
            errs = abs(g["source_trace"].astype(float) - b["source_trace"].astype(float)).max(
                axis=1
            )
            trace_diagnostics.extend(
                dict(arm=arm, layer=j + 2, error=float(e), bound=1e-4, passes=bool(e <= 1e-4))
                for j, e in enumerate(errs)
            )
            engine.record_paths = True
            finite = engine.finite(arm, delta)
            paths[arm] = engine.last_paths
            engine.record_paths = False
            data.update({arm + "_finite_" + n: v for n, v in finite.items()})
            check(
                arm + "_CPU64_head",
                cpu_head(head, finite["head_state"]),
                finite["raw_scores"],
                0.005,
            )
            if arm == "000":
                check("baseline_old_gradient", g["gradient"], p["101_gradient"], 1e-4)
                check("baseline_old_final", finite["offset"], p["101_finite_offset"], 0.002)
                check(
                    "baseline_old_scores", finite["raw_scores"], p["101_finite_raw_scores"], 0.001
                )
            for name, direction in directions.items():
                fd.extend(finite_differences(engine, arm, g["gradient"], direction, name))
        check("source_paths_identical", float(any(x != paths["000"] for x in paths.values())), 0.0)
        engine.unchanged()
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **data)
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 96 * 1024**2
        count = engine.counts.copy()
        count["anchor_qk_recomputations"] = 2 * (len(engine.tapes[0]) - 1)
        save_json(
            path.with_suffix(".json"),
            dict(
                key=key,
                arrays_sha256=sha(path),
                forecast_entry=fs[key],
                previous_entry=ps[key],
                first_source_policy=engine.first_policy,
                prefix_seals=engine.seals,
                prefix_unchanged=True,
                anchor_parent_seals=[
                    {
                        str(j): {n: digest(array(v)) for n, v in a.items() if n != "scaling"}
                        for j, a in bank.items()
                    }
                    for bank in engine.attention
                ],
                checks=checks,
                derivatives=fd,
                primitive_checks=primitive_rows,
                primitive_zero_checks=zero_probes,
                source_paths=paths,
                trace_diagnostics=trace_diagnostics,
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
        costs.append(count)
        print(
            "289 saved",
            len(entries),
            "/",
            len(prior["cases"]),
            "numeric failures",
            sum(not x["passes"] for x in checks + primitive_rows + zero_probes)
            + sum(x["error"] > x["bound"] for x in fd),
            flush=True,
        )
        del engine, data, f, p, g, b, finite, primitive
        gc.collect()
        release(model)
    save_json(
        out / "status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=raw_bytes,
            counts={n: sum(c[n] for c in costs) for n in costs[0]},
            ended=stamp(),
            outcomes_not_yet_audited=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference-285", "previous-288", "head-287", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    main(a.reference_285.resolve(), a.previous_288.resolve(), a.head_287.resolve(), a.out.resolve())
