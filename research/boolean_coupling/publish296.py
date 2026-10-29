"""Publish pre-native receipts and later outcomes without weights/activation arrays."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-geometry-296"


def inventory(run, phase, status):
    folder = "forecasts" if phase == "forecast" else "observations"
    rows = []
    for e in status["cases"]:
        for name in [folder, "witnesses-" + phase]:
            p = run / name / (e["key"] + ".npz")
            if not p.exists():
                continue
            with np.load(p) as z:
                arrays = {n: dict(shape=list(z[n].shape), dtype=str(z[n].dtype)) for n in z.files}
            rows.append(
                dict(path=name + "/" + p.name, sha256=sha(p), bytes=p.stat().st_size, arrays=arrays)
            )
    return rows


def release(phase):
    dest = DEST / phase
    d = read(DEST / "pre-native/design.json")
    proof = read(DEST / "pre-native/execution-qualification.json")
    files = {p for p in dest.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in d["source_hashes"])
    files.update(ROOT / n for n in proof["source_hashes"])
    if phase == "native-phase":
        files.add(DEST / "pre-native/release-manifest.json")
        files.add(ROOT / "docs/analysis/296-query-geometry.md")
        files.update((ROOT / "docs/figures/query-geometry-296").glob("*"))
        files.update(
            ROOT / "research/boolean_coupling" / n
            for n in ["publish296.py", "describe296.py", "verify296.py"]
        )
    write(
        dest / "release-manifest.json",
        dict(
            experiment=296,
            phase=phase,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Record integrity/arithmetic and source ancestry, not independent checkpoint or omitted-array replication.",
        ),
    )


def publish(run, cpu, old, phase):
    dest = DEST / phase
    assert not dest.exists()
    if phase == "pre-native":
        a = read(run / "pre-audit.json")
        assert a["status"] == "passed" and not list((run / "observations").glob("*"))
        names = [
            "design.json",
            "runtime.json",
            "forecast-status.json",
            "pre-audit.json",
            "forecast-seal.json",
            "pre-checks.csv",
            "pre-raw-logits.csv",
            "execution-qualification.json",
            "exact-original-replay.json",
        ]
        folder, ph, status = "forecasts", "forecast", read(run / "forecast-status.json")
        for n in names:
            copy(run / n, dest / n)
        copy(cpu / "qualification.json", dest / "cpu-qualification.json")
        copy(old / "design.json", dest / "original-stop/design.json")
        original = []
        for c in read(old / "design.json")["cases"][:2]:
            p = old / "forecasts" / (c["key"] + ".npz")
            wp = old / "witnesses-forecast" / p.name
            original.append(
                dict(
                    key=c["key"],
                    arrays_sha256=sha(p),
                    metadata_sha256=sha(p.with_suffix(".json")),
                    witness_sha256=sha(wp) if wp.exists() else None,
                )
            )
        write(
            dest / "original-stop/summary.json",
            dict(
                error="Immutable shared progress basename rejected second payload",
                raw_failure_sha256=sha(old / "failure-forecast.json"),
                saved_cases=2,
                ordinary_true=0,
                completed_full_forwards=16,
                cases=original,
            ),
        )
    else:
        assert phase == "native-phase" and read(run / "audit.json")["status"] == "passed"
        for n in [
            "native-status.json",
            "measurement-proof.json",
            "audit.json",
            "checks.csv",
            "raw-logits.csv",
            "native.csv",
            "vector-records.json",
        ]:
            copy(run / n, dest / n)
        folder, ph, status = "observations", "native", read(run / "native-status.json")
    scores = []
    for e in status["cases"]:
        p = run / folder / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        copy(p.with_suffix(".json"), dest / "cases" / p.with_suffix(".json").name)
        with np.load(p) as z:
            scores.append(
                dict(
                    key=e["key"],
                    scores={
                        n: z[n].astype(float).tolist() for n in z.files if n.endswith("_scores")
                    },
                )
            )
    write(dest / "scores.json", dict(cases=scores))
    write(
        dest / "omitted-array-inventory.json",
        dict(
            raw_bytes=status["raw_bytes"],
            arrays=inventory(run, ph, status),
            boundary="Local weights and arrays omitted; hashes bind regeneration, not independent vector replication.",
        ),
    )
    if phase == "native-phase":
        fs = read(run / "forecast-status.json")
        ns = read(run / "native-status.json")
        write(
            dest / "cost.json",
            dict(
                original_stopped_full_forwards=16,
                completed_forecast_counts=fs["counts"],
                native_counts=ns["counts"],
                total_checkpoint_full_forwards=16
                + fs["counts"]["full_forwards"]
                + ns["counts"]["full_forwards"],
                ordinary_native_true_calls=32,
                independent_model_replays=0,
                dense_current_queries_evaluated=True,
                wallclock_speedup_claimed=False,
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["publish", "seal"])
    p.add_argument("--phase", choices=["pre-native", "native-phase"], required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--original", type=Path)
    a = p.parse_args()
    if a.action == "seal":
        release(a.phase)
    else:
        publish(
            a.run.resolve(),
            None if a.cpu is None else a.cpu.resolve(),
            None if a.original is None else a.original.resolve(),
            a.phase,
        )
