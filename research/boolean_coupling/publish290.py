"""Publish scalar290 records and source seals without model weights or activations."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/receiving-nonlinearity-290"


def main(run, cpu):
    assert not DEST.exists()
    design, status, audit = (read(run / (n + ".json")) for n in ("design", "status", "audit"))
    assert design["experiment"] == 290 and not design["synthetic"]
    assert audit["status"] == "passed" and status["status"] == "complete"
    for n in ("design.json", "status.json", "audit.json", "runtime.json", *audit["csv_sha256"]):
        copy(run / n, DEST / n)
    arrays, scores = [], []
    for entry in status["cases"]:
        p = run / "cases" / (entry["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"] and p.stat().st_size == entry["bytes"]
        assert sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        copy(p.with_suffix(".json"), DEST / "cases" / p.with_suffix(".json").name)
        with np.load(p) as a:
            arrays.append(
                dict(
                    path="cases/" + p.name,
                    sha256=sha(p),
                    bytes=p.stat().st_size,
                    arrays={
                        n: dict(shape=list(a[n].shape), dtype=str(a[n].dtype)) for n in a.files
                    },
                )
            )
            scores.append(
                dict(
                    key=entry["key"],
                    scores={n: a[n].tolist() for n in a.files if n.endswith("_scores")},
                )
            )
    write(
        DEST / "scores.json",
        dict(
            experiment=290,
            cases=scores,
            boundary="Actual finite candidate logits from full-vocabulary heads; no vectors.",
        ),
    )
    write(
        DEST / "omitted-array-inventory.json",
        dict(
            experiment=290,
            arrays=arrays,
            case_bytes=status["raw_bytes"],
            boundary="Parent/adjoint and gradient vectors verified locally; public CI checks records and seals, not omitted arrays.",
        ),
    )
    assert sum(e["bytes"] for e in arrays) == status["raw_bytes"]
    assert read(cpu / "preflight-summary.json")["passed"]
    for n in ("preflight-summary.json", "all-coordinates.json"):
        copy(cpu / n, DEST / "cpu-qualification" / n)
    ca = read(cpu / "complete/audit.json")
    for n in ("design.json", "status.json", "audit.json", *ca["csv_sha256"]):
        if n in ("hypotheses.csv", "receiving_vectors.csv", "native_scalar.csv", "ledger.csv"):
            continue
        copy(cpu / "complete" / n, DEST / "cpu-qualification/complete" / n)


def seal():
    sources = read(DEST / "design.json")["source_hashes"]
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in sources)
    files.add(ROOT / "docs/analysis/290-receiving-nonlinearity.md")
    files.update((ROOT / "docs/figures/receiving-nonlinearity-290").glob("*"))
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in ("publish290.py", "describe290.py", "verify290.py")
    )
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=290,
            retrospective=True,
            scientific_sources=sources,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Source-sealed records/software, not independent checkpoint or omitted-vector replication.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--seal", action="store_true")
    a = p.parse_args()
    if a.seal:
        assert a.run is None and a.cpu is None
        seal()
    else:
        assert a.run is not None and a.cpu is not None
        main(a.run.resolve(), a.cpu.resolve())
