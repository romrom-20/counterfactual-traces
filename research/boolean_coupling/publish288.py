"""Export audited scalar 288 evidence; weights, vectors and private prose stay local."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/tangent-routing-288"


def main(run, cpu):
    assert not DEST.exists()
    d, status, audit = (read(run / (n + ".json")) for n in ("design", "status", "audit"))
    assert d["experiment"] == 288 and not d["synthetic"]
    assert status["status"] == "complete" and audit["status"] == "passed"
    for name in ("design.json", "status.json", "audit.json", "runtime.json", *audit["csv_sha256"]):
        copy(run / name, DEST / name)
    arrays, scores = [], []
    for entry in status["cases"]:
        p = run / "cases" / (entry["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"] and p.stat().st_size == entry["bytes"]
        assert sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        copy(p.with_suffix(".json"), DEST / "cases" / p.with_suffix(".json").name)
        with np.load(p) as a:
            arrays.append(dict(path="cases/" + p.name, sha256=sha(p), bytes=p.stat().st_size,
                               arrays={n: dict(shape=list(a[n].shape), dtype=str(a[n].dtype)) for n in a.files}))
            scores.append(dict(key=entry["key"], scores={
                n: a[n].tolist() for n in a.files if n.endswith("_scores")
            }))
    assert sum(r["bytes"] for r in arrays) == status["raw_bytes"]
    write(DEST / "scores.json", dict(
        experiment=288, cases=scores,
        boundary="Candidate logits from finite full-vocabulary heads; no activation vectors or weights.",
    ))
    write(DEST / "omitted-array-inventory.json", dict(
        experiment=288, arrays=arrays, case_bytes=status["raw_bytes"],
        head_reference="Experiment287's omitted head.npz; no new head array copied.",
        boundary="Arrays verified at export; public CI checks records, not omitted arrays or checkpoint replication.",
    ))
    assert read(cpu / "preflight-summary.json")["passed"]
    for name in ("preflight-summary.json", "all-coordinates.json"):
        copy(cpu / name, DEST / "cpu-qualification" / name)
    for name in ("design.json", "status.json", "audit.json", "checks.csv", "derivatives.csv"):
        copy(cpu / "complete" / name, DEST / "cpu-qualification" / "complete" / name)


def seal():
    sources = read(DEST / "design.json")["source_hashes"]
    extra = [ROOT / "docs/analysis/288-source-receiver-tangent-routing.md"]
    extra += list((ROOT / "docs/figures/tangent-routing-288").glob("*"))
    extra += [ROOT / "research/boolean_coupling" / n
              for n in ("publish288.py", "describe288.py", "verify288.py")]
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in sources)
    files.update(extra)
    write(DEST / "release-manifest.json", dict(
        experiment=288, retrospective=True, scientific_sources=sources,
        files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
        boundary="Public records/software verification, not independent primitive AD or omitted-vector replication.",
    ))


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
