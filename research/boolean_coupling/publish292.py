"""Publish sealed scalar evidence, never raw arrays, model weights or articles."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/joint-gate-292"


def cases(run, phase, status, dest):
    inventory, scores = [], []
    for e in status["cases"]:
        p = run / phase / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        copy(p.with_suffix(".json"), dest / "cases" / p.with_suffix(".json").name)
        with np.load(p) as z:
            inventory.append(
                dict(
                    path="cases/" + p.name,
                    sha256=sha(p),
                    bytes=p.stat().st_size,
                    arrays={
                        n: dict(shape=list(z[n].shape), dtype=str(z[n].dtype)) for n in z.files
                    },
                )
            )
            scores.append(
                dict(
                    key=e["key"], scores={n: z[n].tolist() for n in z.files if n.endswith("scores")}
                )
            )
    write(
        dest / "omitted-array-inventory.json",
        dict(
            arrays=inventory,
            case_bytes=status["raw_bytes"],
            boundary="Local arrays are omitted; hashes/shapes/dtypes bind regeneration, not independent model replication.",
        ),
    )
    write(dest / "scores.json", dict(cases=scores))


def pre(run, cpu, replay, stop):
    assert not DEST.exists() and not (run / "native-started.json").exists()
    d, st, a = (read(run / (n + ".json")) for n in ("design", "forecast-status", "forecast-audit"))
    assert (
        d["experiment"] == 292
        and not d["synthetic"]
        and st["ordinary_true"] == a["ordinary_true"] == 0
    )
    assert a["status"] == "passed"
    pre = DEST / "pre-native"
    for n in (
        "design.json",
        "runtime.json",
        "head.json",
        "forecast-status.json",
        "forecast-audit.json",
        "forecast-seal.json",
        "continuation.json",
        "resident-awaiting-receipt.json",
        *a["csv_sha256"],
    ):
        copy(run / n, pre / n)
    cases(run, "forecasts", st, pre)
    for name in ("design.json", "runtime.json", "failure-forecast.json", "head.json"):
        copy(stop / name, DEST / "original-stop" / name)
    old = list((stop / "forecasts").glob("*.npz"))
    assert len(old) == 1 and sha(old[0]) == read(old[0].with_suffix(".json"))["arrays_sha256"]
    copy(old[0].with_suffix(".json"), DEST / "original-stop/first-case.json")
    write(
        DEST / "original-stop/omitted-array.json",
        dict(
            sha256=sha(old[0]),
            bytes=old[0].stat().st_size,
            boundary="Original first archive remains local and unchanged. The full continuation replays all42arrays exactly beforecase2.",
        ),
    )
    for name in ("preflight-summary.json",):
        copy(cpu / name, DEST / "cpu-qualification" / name)
    for name in ("design.json", "forecast-audit.json", "native-audit.json"):
        copy(cpu / "complete" / name, DEST / "cpu-qualification" / name)
    copy(replay / "preflight-summary.json", DEST / "cpu-qualification/replay-qualification.json")
    helpers = (
        "research/boolean_coupling/continue292.py",
        "research/boolean_coupling/preflight292b.py",
        "research/boolean_coupling/resident292.py",
        "docs/experiments/292b-resource-continuation.md",
        "docs/experiments/292c-resident-execution.md",
    )
    files = {p for p in DEST.rglob("*") if p.is_file()}
    files.update(ROOT / n for n in (*d["source_hashes"], *helpers))
    write(
        DEST / "pre-manifest.json",
        dict(
            experiment=292,
            ordinary_true=0,
            scientific_sources=d["source_hashes"],
            execution_sources={n: sha(ROOT / n) for n in helpers},
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Globally qualified predictions before ordinary checkpoint True; CPU native tests are explicitly synthetic.",
        ),
    )


def post(run):
    assert (DEST / "pre-manifest.json").exists() and not (DEST / "native-phase").exists()
    st, a = read(run / "native-status.json"), read(run / "native-audit.json")
    assert a["status"] == "passed"
    out = DEST / "native-phase"
    for n in (
        "native-started.json",
        "native-status.json",
        "native-audit.json",
        "resident-complete.json",
        *a["csv_sha256"],
    ):
        copy(run / n, out / n)
    cases(run, "native", st, out)


def seal():
    d = read(DEST / "pre-native/design.json")
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in d["source_hashes"])
    files.update(ROOT / n for n in read(DEST / "pre-manifest.json")["execution_sources"])
    files.add(ROOT / "docs/analysis/292-joint-gate-forecast.md")
    files.update((ROOT / "docs/figures/joint-gate-292").glob("*"))
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in ("publish292.py", "verify292.py", "describe292.py", "review292.py")
    )
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=292,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Source/record verification, not independent checkpoint or omitted-array replication.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("pre", "post", "seal"))
    for flag in ("run", "cpu", "replay-cpu", "original-stop"):
        p.add_argument("--" + flag, type=Path)
    a = p.parse_args()
    if a.phase == "pre":
        pre(a.run.resolve(), a.cpu.resolve(), a.replay_cpu.resolve(), a.original_stop.resolve())
    elif a.phase == "post":
        post(a.run.resolve())
    else:
        seal()
