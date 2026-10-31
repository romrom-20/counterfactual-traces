"""Frozen retrospective native-parent diagnostic; no prospective forecast claim."""

import argparse
import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import load_model
from .interaction300 import BOUNDS, CRITERIA, ROOT, SOURCES, CalibrationFailure, execute
from .run296 import runtime, stamp


def main(run, parent_run):
    assert not run.exists()
    assert shutil.disk_usage(run.parent).free >= 8 * 1024**3
    parents = read(ROOT / "research/boolean_coupling/parents300.json")
    assert len(parents["cases"]) == 8 and parents["selected_block"] == 15
    d = read(parent_run / "design.json")
    for path, field in (
        ("results/query-generators-299/pre-native/design.json", "parent_public_design_sha256"),
        (
            "results/query-generators-299/native-phase/native-status.json",
            "parent_native_status_sha256",
        ),
        (
            "results/query-generators-299/native-phase/release-manifest.json",
            "parent_release_sha256",
        ),
    ):
        assert sha(ROOT / path) == parents[field]
    assert sha(parent_run / "design.json") == parents["parent_public_design_sha256"]
    assert sha(parent_run / "query-weights.npz") == parents["parent_query_weights_sha256"]
    assert sha(parent_run / "head.npz") == parents["parent_head_sha256"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    assert len(hashes) == len(SOURCES) and len({Path(n).name for n in SOURCES}) == len(SOURCES)
    for n, h in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        ), n
    cpu = read(ROOT.parent / "300-cpu/qualification.json")
    assert cpu["passed"] and cpu["source_hashes"] == hashes
    _, checkpoint = locate(ROOT)
    assert checkpoint == d["checkpoint"]
    run.mkdir()
    (run / "sources").mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, run / "sources" / Path(n).name)
    with np.load(parent_run / "head.npz") as z:
        weights = dict(z)
    with np.load(parent_run / "query-weights.npz") as z:
        weights["output"] = z["14_output"]
    np.savez_compressed(run / "weights.npz", **weights)
    save_json(
        run / "design.json",
        dict(
            experiment=300,
            retrospective=True,
            native_parents_supplied=True,
            false_only_forecast=False,
            parents=parents,
            criteria=CRITERIA,
            bounds=BOUNDS,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=checkpoint,
            weights_sha256=sha(run / "weights.npz"),
            cpu_qualification_sha256=sha(ROOT.parent / "300-cpu/qualification.json"),
            raw_cap_bytes=64 * 1024**2,
            created_utc=stamp(),
        ),
    )
    model = load_model(True)
    assert runtime(model) == read(parent_run / "runtime.json")
    save_json(run / "runtime.json", runtime(model))
    counts, entries, total = Counter(), [], 0
    for index, item in enumerate(parents["cases"]):
        assert shutil.disk_usage(run).free >= 8 * 1024**3
        case = item["case"]
        key = case["key"]
        pp = parent_run / "witnesses-native" / (key + ".npz")
        rp = parent_run / "observations" / (key + ".npz")
        assert (
            sha(pp) == item["witness_sha256"]
            and sha(rp) == item["native_arrays_sha256"]
            and sha(rp.with_suffix(".json")) == item["native_metadata_sha256"]
        )
        with np.load(pp) as z:
            parent = dict(z)
        with np.load(rp) as z:
            reference = dict(z)
        try:
            data, checks, used, meta = execute(
                model, case, parent, reference, read(rp.with_suffix(".json"))
            )
        except CalibrationFailure as error:
            np.savez_compressed(run / (key + "-partial.npz"), **error.data)
            save_json(
                run / "calibration-failure.json",
                dict(key=key, error=str(error), counts=error.counts, checks=error.checks),
            )
            raise
        counts.update(used)
        p = run / (key + ".npz")
        np.savez_compressed(p, **data)
        total += p.stat().st_size
        assert total <= 64 * 1024**2
        save_json(
            p.with_suffix(".json"),
            dict(
                key=key,
                group=case["group"],
                arrays_sha256=sha(p),
                checks=checks,
                counts=used,
                **meta,
            ),
        )
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
                bytes=p.stat().st_size,
            )
        )
        save_json(
            run / f"progress-{index + 1:04}.json",
            dict(cases=index + 1, counts=dict(counts), raw_bytes=total),
        )
        print("300 retained native diagnostic", index + 1, "/8", flush=True)
    assert counts == dict(
        full_forwards=48,
        extra_full_QK=8,
        extra_full_softmax=8,
        extra_full_PV=8,
        extra_full_output_projection=32,
    ), dict(counts)
    save_json(
        run / "status.json",
        dict(
            status="complete",
            retrospective=True,
            cases=entries,
            counts=dict(counts),
            raw_bytes=total,
            ordinary_native_calibration_repetitions=8,
            conditional_intervention_forwards=40,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve(), args.parent.resolve())
