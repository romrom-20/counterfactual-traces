"""Publish corrected293c comparisons with invalid original records retained."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/response-subspace-293"
EXEC = (
    "research/boolean_coupling/resource293b.py",
    "research/boolean_coupling/preflight293b.py",
    "docs/experiments/293b-resource-continuation.md",
)


def main(run, correction, stop, cpu, resource_cpu, namespace_cpu):
    assert not DEST.exists()
    d, st, old = (read(run / (n + ".json")) for n in ("design", "status", "audit"))
    a = read(correction / "audit.json")
    assert (
        old["status"] == a["status"] == "passed"
        and not a["original_scientific_decisions_valid"]
        and not d["synthetic"]
    )
    for n in (
        "design.json",
        "runtime.json",
        "head.json",
        "status.json",
        "resource-qualification.json",
        "checks.csv",
        "local.csv",
    ):
        copy(run / n, DEST / n)
    for n in ("audit.json", "native.csv", "pairs.csv"):
        copy(run / n, DEST / "invalid-original" / n)
    for n in ("native.csv", "pairs.csv", "audit.json", "scores.json", "vector-records.json"):
        copy(correction / n, DEST / n)
    for n in ("design.json", "baseline-proof.json", "checks.csv"):
        copy(correction / n, DEST / "namespace-qualification" / n)
    inventory = []
    counts = {}
    for e in st["cases"]:
        p = run / "cases" / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        copy(p.with_suffix(".json"), DEST / "cases" / p.with_suffix(".json").name)
        m = read(p.with_suffix(".json"))
        for meta in m["arms"].values():
            for k, v in meta["counts"].items():
                counts[k] = counts.get(k, 0) + v
        with np.load(p) as z:
            inventory.append(
                dict(
                    path="cases/" + p.name,
                    bytes=p.stat().st_size,
                    sha256=sha(p),
                    arrays={
                        n: dict(shape=list(z[n].shape), dtype=str(z[n].dtype)) for n in z.files
                    },
                )
            )
    write(
        DEST / "omitted-array-inventory.json",
        dict(
            arrays=inventory,
            case_bytes=st["raw_bytes"],
            boundary="Original237-array candidate archives retained. B0 restored from292False; no candidate array rewritten. Omitted arrays not independently replicated by CI.",
        ),
    )
    write(
        DEST / "cost.json",
        dict(
            completed_producer_full_F=8,
            completed_candidate_and_zero_forwards=counts.pop("full_forwards"),
            original_stopped_forwards=77,
            additional_component_calls=counts,
            ordinary_checkpoint_true=0,
            namespace_correction_model_calls=0,
            wallclock_speedup_claimed=False,
        ),
    )
    for n in ("design.json", "runtime.json", "failure.json"):
        copy(stop / n, DEST / "original-stop" / n)
    oldcases = [
        dict(
            key=p.stem,
            arrays_sha256=sha(p),
            metadata_sha256=sha(p.with_suffix(".json")),
            bytes=p.stat().st_size,
        )
        for p in sorted((stop / "cases").glob("*.npz"))
    ]
    write(
        DEST / "original-stop/inventory.json",
        dict(cases=oldcases, complete_status=False, ordinary_checkpoint_true=0),
    )
    for folder, src in [
        ("cpu", cpu),
        ("resource-cpu", resource_cpu),
        ("namespace-cpu", namespace_cpu),
    ]:
        copy(src / "preflight-summary.json", DEST / folder / "preflight-summary.json")
        for n in ("design.json", "audit.json"):
            copy(src / "complete" / n, DEST / folder / n)
    copy(cpu / "rejected-basis/audit.json", DEST / "cpu/expected-corruption-audit.json")


def seal():
    d = read(DEST / "design.json")
    c = read(DEST / "namespace-qualification/design.json")
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in (*d["source_hashes"], *c["source_hashes"], *EXEC))
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in ("publish293.py", "verify293.py", "describe293.py")
    )
    files.add(ROOT / "docs/analysis/293-response-subspace.md")
    files.update((ROOT / "docs/figures/response-subspace-293").glob("*"))
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=293,
            retrospective=True,
            scientific_sources=d["source_hashes"],
            namespace_sources=c["source_hashes"],
            execution_sources={n: sha(ROOT / n) for n in EXEC},
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Corrected293c comparisons only. Original science tables invalid; primitive qualifications retained. No independent omitted-array/checkpoint replication.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("publish", "seal"))
    for n in ("run", "correction", "stop", "cpu", "resource-cpu", "namespace-cpu"):
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.phase == "seal":
        seal()
    else:
        main(
            a.run.resolve(),
            a.correction.resolve(),
            a.stop.resolve(),
            a.cpu.resolve(),
            a.resource_cpu.resolve(),
            a.namespace_cpu.resolve(),
        )
