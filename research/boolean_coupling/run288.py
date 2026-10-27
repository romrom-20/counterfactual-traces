"""Run the frozen finite tangent-routing diagnostic on the retained 287 cohort."""

import argparse
import gc
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .readout286 import ROOT, release
from .routing288 import ARMS, BOUNDS, PRIMARY, SOURCES, Routing, finite_differences
from .run272 import stamp
from .run285 import model_signature, runtime
from .run286 import cpu_head, entry_data, head_data


def references(reference, previous, qualification):
    return {
        "285_" + n: sha(reference / (n + ".json"))
        for n in ("design", "forecast-status", "native-status")
    } | {
        "287_" + n: sha(previous / (n + ".json"))
        for n in ("design", "status", "audit", "head")
    } | {
        "287b_" + n: sha(qualification / (n + ".json")) for n in ("design", "audit")
    }


def main(reference, previous, qualification, out, model=None):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 640 * 1024**2
    old, prior, qual = (read(p / "design.json") for p in (reference, previous, qualification))
    assert old["experiment"] == 285 and prior["experiment"] == 287
    assert qual["experiment"] == "287b"
    assert read(qualification / "audit.json")["status"] == "qualified_boundary_only"
    assert old["installed_sources"] == prior["installed_sources"] == installed_sources()
    synthetic = model is not None
    assert synthetic is old["synthetic"] is prior["synthetic"]
    if synthetic:
        assert model_signature(model) == old["checkpoint"]
    else:
        assert locate(ROOT)[1] == old["checkpoint"]
    for name, value in qual["source_hashes"].items():
        assert sha(ROOT / name) == value
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    if not synthetic:
        import hashlib

        for name, value in hashes.items():
            sealed = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
            assert hashlib.sha256(sealed).hexdigest() == value
    out.mkdir()
    (out / "sources").mkdir()
    (out / "cases").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    save_json(out / "design.json", dict(
        experiment=288, retrospective=True, synthetic=synthetic, source_commit=commit,
        source_hashes=hashes, installed_sources=installed_sources(), checkpoint=old["checkpoint"],
        cases=prior["cases"], reference_hashes=references(reference, previous, qualification),
        numerical_bounds=BOUNDS, primary=PRIMARY, FD_steps=[0.05, 0.025], arms=list(ARMS),
        factors=["source_blocks", "receiving_blocks", "final_head"],
        anchors=["False", "source-only-MF"], resource_guards_MiB=[640, 384, 128],
        began=stamp(),
        boundary="Retrospective finite constructed networks; no True gradient, compact predictor, learned circuit, speedup or generalQB closure.",
    ))
    model = load_model(True) if model is None else model
    assert not any(p.requires_grad for p in model.parameters())
    save_json(out / "runtime.json", runtime(model, synthetic))
    with np.load(previous / "head.npz") as saved:
        head = {n: saved[n] for n in saved.files}
    assert all(np.array_equal(head[n], v) for n, v in head_data(model).items())
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ps = {e["key"]: e for e in read(previous / "status.json")["cases"]}
    entries, totals, raw_bytes = [], [], 0
    for case in prior["cases"]:
        release(model)
        assert shutil.disk_usage(out).free >= 384 * 1024**2
        key = case["key"]
        forecast, p = entry_data(reference, "forecasts", fs[key]), entry_data(previous, "cases", ps[key])
        engine = Routing(model, case, forecast["M_write"])
        delta = p["delta"].copy()
        data = dict(delta=delta, ordinary_F_final=engine.false_final,
                    ordinary_F_scores=engine.false_scores,
                    initial_F=array(engine.tapes[0][0][0]), initial_E=array(engine.tapes[1][0][0]))
        checks, fd = [], []

        def check(name, actual, expected, bound=0.0):
            error = float(np.max(abs(np.asarray(actual, float) - np.asarray(expected, float))))
            checks.append(dict(name=name, error=error, bound=bound,
                               passes=bool(np.isfinite(error) and error <= bound)))

        check("ordinary_F_final", engine.false_final, p["ordinary_F_final"], 1e-4)
        check("ordinary_F_scores", engine.false_scores, p["ordinary_F_scores"], 1e-4)
        check("initial_F_source", data["initial_F"][0], forecast["F_source"])
        check("initial_E_source", data["initial_E"][0], forecast["M_source"])
        check("initial_receivers_unchanged", data["initial_E"][1:], data["initial_F"][1:])
        check("delta_identity", delta, forecast["C_source"].astype(float) - forecast["M_source"])
        for n in ("gate", "phi", "write"):
            data["F_" + n] = engine.false_nodes[n]
            check("False_" + n, engine.false_nodes[n], forecast["F_" + n])
        for anchor, corner in ((0, "FF"), (1, "MF")):
            data[f"anchor{anchor}_final"] = array(engine.tapes[anchor][-1][0])
            data[f"anchor{anchor}_scores"] = engine.scores[anchor]
            check(f"anchor{anchor}_primal", data[f"anchor{anchor}_final"], p[corner + "_final"], 1e-4)
            check(f"anchor{anchor}_scores", engine.scores[anchor], p[corner + "_scores"], 1e-4)
            final, scores = engine.native_bump(anchor, delta)
            data[f"anchor{anchor}_native_bump_final"] = final
            data[f"anchor{anchor}_native_bump_scores"] = scores
            check(f"anchor{anchor}_old_finite_final", final, p[corner + "_plus_delta_final"], 0.002)
            check(f"anchor{anchor}_old_finite_scores", scores, p[corner + "_plus_delta_scores"], 0.001)
        coords = np.random.default_rng(int(key[:16], 16)).choice(len(delta), 2, replace=False)
        directions = {"delta": delta}
        for coordinate in coords:
            d = np.zeros_like(delta)
            d[coordinate] = 1.0
            directions["coordinate_" + str(coordinate)] = d
        for arm in ARMS:
            taped = engine.gradient(arm)
            release(model)
            triangular = engine.triangular(arm)
            release(model)
            finite = engine.finite(arm, delta)
            release(model)
            data.update({arm + "_" + n: v for n, v in taped.items()})
            data.update({arm + "_triangular_" + n: v for n, v in triangular.items()})
            data.update({arm + "_finite_" + n: v for n, v in finite.items()})
            check(arm + "_zero_offset", taped["zero_offset"], 0.0)
            check(arm + "_zero_head", taped["zero_raw_scores"], engine.scores[int(arm[-1])])
            check(arm + "_boundary_adjoint", taped["gradient"], triangular["gradient"], 1e-4)
            noncausal = triangular["noncausal"]
            check(arm + "_noncausal", np.max(noncausal) if noncausal.size else 0.0, 0.0)
            check(arm + "_CPU64_head", cpu_head(head, finite["head_state"]), finite["raw_scores"], 0.005)
            if arm in ("000", "111"):
                anchor, corner = (0, "FF") if arm == "000" else (1, "MF")
                check(arm + "_old_cached_gradient", taped["gradient"], p[corner + "_gradient"], 1e-4)
                check(arm + "_old_full_gradient", taped["gradient"], p[corner + "_full_source_trace"][0], 1e-4)
                matched = data[f"anchor{anchor}_final"].astype(float) + finite["offset"]
                check(arm + "_matched_native_final", matched, data[f"anchor{anchor}_native_bump_final"], 0.002)
                check(arm + "_matched_native_scores", finite["raw_scores"], data[f"anchor{anchor}_native_bump_scores"], 0.001)
            for name, direction in directions.items():
                fd.extend(finite_differences(engine, arm, taped["gradient"], direction, name))
        engine.unchanged()
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **data)
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 128 * 1024**2
        save_json(path.with_suffix(".json"), dict(
            key=key, arrays_sha256=sha(path), checks=checks, derivatives=fd,
            first_source_policy=engine.first_policy, prefix_seals=engine.seals,
            prefix_unchanged=True, forecast_entry=fs[key], previous_entry=ps[key],
            counts=engine.counts,
        ))
        entries.append(dict(key=key, arrays_sha256=sha(path),
                            metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        totals.append(engine.counts.copy())
        print("288 saved", len(entries), "/", len(prior["cases"]),
              "required numeric failures", sum(not r["passes"] for r in checks)
              + sum(r["error"] > r["bound"] for r in fd), flush=True)
        del engine, taped, triangular, finite, data, forecast, p
        gc.collect()
        release(model)
    save_json(out / "status.json", dict(
        status="complete", cases=entries, raw_bytes=raw_bytes, ended=stamp(),
        counts={n: sum(r[n] for r in totals) for n in totals[0]}, outcomes_not_yet_audited=True,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-285", type=Path, required=True)
    p.add_argument("--previous-287", type=Path, required=True)
    p.add_argument("--qualification-287b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.reference_285.resolve(), a.previous_287.resolve(),
         a.qualification_287b.resolve(), a.out.resolve())
