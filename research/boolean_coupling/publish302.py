"""Publish qualified 302 records; model weights, activation arrays and drafts stay local."""

import argparse
import statistics
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/first-source-geometry-302"


def publish(run, cpu):
    assert not DEST.exists()
    assert (
        read(run / "audit.json")["status"]
        == read(run / "calibration-audit.json")["status"]
        == "passed"
    )
    names = [
        "design.json",
        "runtime.json",
        "calibration-status.json",
        "intervention-status.json",
        "calibration-audit.json",
        "audit.json",
        "calibration-checks.csv",
        "calibration-raw.csv",
        "checks.csv",
        "raw.csv",
        "effects.csv",
        "phase.csv",
        "ledger.csv",
        "vector-records.json",
    ]
    for name in names:
        copy(run / name, DEST / name)
    copy(cpu / "qualification.json", DEST / "cpu-qualification.json")
    inventory, scores = [], []
    for case in read(run / "design.json")["cases"]:
        key = case["key"]
        for folder in ["calibration", "interventions", "components"]:
            path = run / folder / (key + ".npz")
            with np.load(path) as z:
                spec = {n: dict(shape=list(z[n].shape), dtype=str(z[n].dtype)) for n in z.files}
                if folder != "components":
                    scores.append(
                        dict(
                            key=key,
                            phase=folder,
                            scores={n: z[n].tolist() for n in z.files if n.endswith("_scores")},
                        )
                    )
            inventory.append(
                dict(
                    path=folder + "/" + path.name,
                    sha256=sha(path),
                    bytes=path.stat().st_size,
                    arrays=spec,
                )
            )
            if folder != "components":
                copy(
                    path.with_suffix(".json"),
                    DEST / "cases" / folder / path.with_suffix(".json").name,
                )
    write(DEST / "omitted-arrays.json", dict(arrays=inventory, activation_arrays_published=False))
    write(DEST / "scores.json", dict(cases=scores))
    for n in ["joint.csv", "provenance.json"]:
        copy(run / "posthoc-joint" / n, DEST / "posthoc-joint" / n)
    import csv

    with (run / "effects.csv").open() as f:
        rows = list(csv.DictReader(f))
    summary = {}
    for arm in ["K", "V", "H", "S", "RH", "RK", "RV"]:
        rr = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            name: statistics.median(max(float(r["A_" + name]), float(r["B_" + name])) for r in rr)
            for name in [
                "source_relative",
                "emit_relative",
                "receiving_relative",
                "verdict_relative",
                "margin_error",
            ]
        }
    write(
        DEST / "descriptive-summary.json",
        dict(posthoc=True, all_eight_cases=True, medians_of_worse_reference=summary),
    )
    write(
        DEST / "cost.json",
        dict(
            full_checkpoint_forwards=384,
            native_calibration_forwards=32,
            intervention_forwards=352,
            extra_full_QK=736,
            extra_full_softmax=352,
            extra_full_PV=352,
            host_audits_are_additional=True,
            raw_array_bytes=sum(r["bytes"] for r in inventory),
            model_copy=False,
            wallclock_speedup_claimed=False,
            not_a_compact_forecast=True,
        ),
    )


def release():
    design = read(DEST / "design.json")
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in design["source_hashes"])
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in [
            "publish302.py",
            "joint302.py",
            "verify302.py",
            "describe302.py",
            "reproduce302.py",
        ]
    )
    files.add(ROOT / "docs/analysis/302-first-source-geometry.md")
    files.update((ROOT / "docs/figures/first-source-geometry-302").glob("*"))
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=302,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Record integrity and signed arithmetic, not independent activation/model/circuit replication.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--release-only", action="store_true")
    a = p.parse_args()
    release() if a.release_only else publish(a.run.resolve(), a.cpu.resolve())
