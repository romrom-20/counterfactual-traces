"""Sealed retrospective source geometry: qualify native replay before interventions."""

import argparse
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .geometry302 import ARMS, BOUNDS, CRITERIA, ROOT, SOURCES, components, execute


def load_parents(parent, case):
    key = case["key"]
    paths = {
        n: parent / n / (key + ".npz")
        for n in ("forecasts", "observations", "witnesses-forecast", "witnesses-native")
    }
    inputs, refs = {}, {}
    for folder in ("witnesses-forecast", "witnesses-native"):
        with np.load(paths[folder]) as z:
            inputs.update(
                {n: z[n] for n in z.files if n.startswith(("A_F_0_", "B_F_0_", "A_T_0_", "B_T_0_"))}
            )
    for folder in ("forecasts", "observations"):
        with np.load(paths[folder]) as z:
            for arm in ("A_F", "B_F", "A_T", "B_T"):
                for suffix in ("trace", "scores", "emit1"):
                    n = arm + "_" + suffix
                    if n in z:
                        refs[n] = z[n]
    return inputs, refs, paths


def free(run):
    assert shutil.disk_usage(run).free >= 8 * 1024**3, "Eight GiB reserve required"


def save_case(run, phase, case, data, checks, counts):
    path = run / phase / (case["key"] + ".npz")
    assert not path.exists()
    np.savez_compressed(path, **data)
    meta = dict(
        key=case["key"],
        sha256=sha(path),
        bytes=path.stat().st_size,
        checks=checks,
        counts=counts,
        passed=all(r["passes"] for r in checks),
    )
    save_json(path.with_suffix(".json"), meta)
    return dict(
        key=case["key"],
        sha256=meta["sha256"],
        bytes=meta["bytes"],
        metadata_sha256=sha(path.with_suffix(".json")),
        passed=meta["passed"],
    )


def calibrate(run, parent):
    assert not run.exists()
    assert read(parent / "audit.json")["status"] == "passed"
    old = read(parent / "design.json")
    assert old["experiment"] == 301
    cases = [c for c in old["cases"] if c["group"]["reverse"] == c["group"]["target"] == 0]
    assert len(cases) == 8
    for n, h in old["source_hashes"].items():
        assert sha(ROOT / n) == h, n
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    cpu = read(ROOT.parent / "302-cpu/qualification.json")
    assert cpu["passed"] and cpu["source_hashes"] == hashes
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for n, h in hashes.items():
        import hashlib

        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        ), n
    _, checkpoint = locate(ROOT)
    assert checkpoint == old["checkpoint"]
    assert installed_sources() == old["installed_sources"]
    run.mkdir()
    for folder in ("calibration", "interventions", "components", "sources"):
        (run / folder).mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, run / "sources" / Path(n).name)
    parent_hashes = {}
    entries = {
        phase: {
            r["key"]: r
            for r in read(
                parent / ("forecast-status.json" if phase == "forecasts" else "native-status.json")
            )["cases"]
        }
        for phase in ("forecasts", "observations")
    }
    for case in cases:
        _, _, paths = load_parents(parent, case)
        for folder, path in paths.items():
            phase = "forecasts" if folder in ("forecasts", "witnesses-forecast") else "observations"
            field = "witness_sha256" if folder.startswith("witnesses") else "arrays_sha256"
            assert sha(path) == entries[phase][case["key"]][field]
            parent_hashes[str(path.relative_to(parent))] = sha(path)
    save_json(
        run / "design.json",
        dict(
            experiment=302,
            source_commit=commit,
            source_hashes=hashes,
            cpu_qualification_sha256=sha(ROOT.parent / "302-cpu/qualification.json"),
            checkpoint=checkpoint,
            installed_sources=installed_sources(),
            cases=cases,
            parent_design_sha256=sha(parent / "design.json"),
            parent_files=parent_hashes,
            parent_raw_failures_retained=96,
            parent_raw_total=576,
            criteria=CRITERIA,
            bounds=BOUNDS,
            arms=ARMS,
            retrospective=True,
            native_parents_supplied=True,
            False_only_forecast=False,
            guards=dict(free_GiB=8, raw_MiB=128),
            expected_full_forwards=384,
            expected_intervention_attention_calls=352,
            expected_witness_QK=384,
            boundary="One first-source row only, all later finite operations intact. Conditional KV-minus-native targets, not the full alias effect or a learned circuit.",
        ),
    )
    model = load_model(True)
    inv = array(model.model.rotary_emb.inv_freq).astype(float)
    np.savez_compressed(run / "rotary.npz", inv=inv)
    save_json(
        run / "runtime.json",
        dict(
            device=str(model.device),
            dtype=str(next(model.parameters()).dtype),
            layers=len(model.model.layers),
            training=model.training,
            rotary_sha256=sha(run / "rotary.npz"),
        ),
    )
    counts, saved, total = Counter(), [], 0
    for case in cases:
        free(run)
        inputs, refs, _ = load_parents(parent, case)
        comp = components(case, inputs, inv)
        cp = run / "components" / (case["key"] + ".npz")
        np.savez_compressed(cp, **comp)
        data, checks, used = execute(model, case, inputs, refs, comp, ("N",))
        entry = save_case(run, "calibration", case, data, checks, used)
        entry["component_sha256"] = sha(cp)
        saved.append(entry)
        counts.update(used)
        total += entry["bytes"]
        save_json(
            run / f"calibration-progress-{len(saved):04}.json",
            dict(cases=len(saved), counts=dict(counts), raw_bytes=total),
        )
        print("302 calibration", len(saved), "/8", flush=True)
        if not entry["passed"]:
            break
    status = dict(
        cases=saved,
        counts=dict(counts),
        raw_bytes=total,
        passed=len(saved) == 8 and all(r["passed"] for r in saved),
    )
    save_json(run / "calibration-status.json", status)
    assert status["passed"], "Native calibration failed; no experimental interventions run"
    assert counts == dict(full_forwards=32, witness_full_QK=32)


def intervene(run, parent):
    design = read(run / "design.json")
    assert read(run / "calibration-audit.json")["status"] == "passed"
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == h
    for n, h in design["parent_files"].items():
        assert sha(parent / n) == h
    _, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    assert installed_sources() == design["installed_sources"]
    assert not list((run / "interventions").glob("*.npz"))
    entries = {e["key"]: e for e in read(run / "calibration-status.json")["cases"]}
    model = load_model(True)
    with np.load(run / "rotary.npz") as z:
        assert np.array_equal(z["inv"], array(model.model.rotary_emb.inv_freq).astype(float))
    counts, saved, total = Counter(), [], 0
    for case in design["cases"]:
        free(run)
        key = case["key"]
        for folder, h in [
            ("calibration", entries[key]["sha256"]),
            ("components", entries[key]["component_sha256"]),
        ]:
            assert sha(run / folder / (key + ".npz")) == h
        with np.load(run / "components" / (key + ".npz")) as z:
            comp = {n: z[n] for n in z.files}
        with np.load(run / "calibration" / (key + ".npz")) as z:
            native = {n: z[n] for n in z.files}
        inputs, refs, _ = load_parents(parent, case)
        data, checks, used = execute(model, case, inputs, refs, comp, ARMS[1:], native)
        entry = save_case(run, "interventions", case, data, checks, used)
        saved.append(entry)
        counts.update(used)
        total += entry["bytes"]
        assert total + read(run / "calibration-status.json")["raw_bytes"] <= 128 * 1024**2
        save_json(
            run / f"intervention-progress-{len(saved):04}.json",
            dict(cases=len(saved), counts=dict(counts), raw_bytes=total),
        )
        print("302 intervention", len(saved), "/8", flush=True)
        if not entry["passed"]:
            break
    status = dict(
        cases=saved,
        counts=dict(counts),
        raw_bytes=total,
        passed=len(saved) == 8 and all(r["passed"] for r in saved),
    )
    save_json(run / "intervention-status.json", status)
    assert status["passed"], (
        "Required execution control failed; retain artifacts and withhold outcomes"
    )
    assert counts == dict(full_forwards=352, witness_full_QK=352, extra_full_QK_softmax_PV_each=352)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["calibrate", "intervene"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    (calibrate if args.phase == "calibrate" else intervene)(
        args.run.resolve(), args.parent.resolve()
    )
