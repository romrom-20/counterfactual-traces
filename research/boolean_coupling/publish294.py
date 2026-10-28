"""Publish qualified294 scalar evidence and omitted-array identities."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/response-change-294"


def main(run, cpu):
    assert not DEST.exists()
    d, st, a = (read(run / (n + ".json")) for n in ("design", "status", "audit"))
    assert (
        a["status"] == "passed"
        and not a["scientific_outcomes_withheld"]
        and a["failed_checks"] == 0
    )
    for n in ("design.json", "runtime.json", "status.json", "audit.json", *a["csv_sha256"]):
        copy(run / n, DEST / n)
    inv = []
    for e in st["cases"]:
        p = run / "cases" / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        copy(p.with_suffix(".json"), DEST / "cases" / p.with_suffix(".json").name)
        with np.load(p) as z:
            inv.append(
                dict(
                    path="cases/" + p.name,
                    sha256=sha(p),
                    bytes=p.stat().st_size,
                    arrays={
                        n: dict(shape=list(z[n].shape), dtype=str(z[n].dtype)) for n in z.files
                    },
                )
            )
    write(
        DEST / "omitted-array-inventory.json",
        dict(
            arrays=inv,
            case_bytes=st["raw_bytes"],
            boundary="Local matrices/vectors and weights omitted. Hashes bind regeneration, not independent replication.",
        ),
    )
    copy(cpu / "qualification.json", DEST / "cpu-qualification.json")
    write(
        DEST / "cost.json",
        dict(
            model_instantiations=0,
            model_forwards=0,
            native_true_calls=0,
            checkpoint_MLP_weight_sets=3,
            conditional_points=48,
            rank=32,
            power_steps=2,
            learned_basis_batches=72,
            learned_basis_row_matrices=144,
            random_basis_batches=24,
            random_basis_row_matrices=48,
            producer_operator_applications=360,
            producer_operator_row_applications=720,
            producer_QR_batches=384,
            producer_QR_row_matrices=768,
            independent_operator_applications=528,
            independent_operator_row_applications=1056,
            independent_QR_batches=384,
            independent_QR_row_matrices=768,
            calculated_from_frozen_loop_counts=True,
            independent_operator_and_QR_checks_additional=True,
            weight_reads_and_host_arithmetic_count=True,
            speedup_claimed=False,
        ),
    )


def seal():
    d = read(DEST / "design.json")
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in d["source_hashes"])
    files.add(ROOT / "docs/analysis/294-response-change.md")
    files.update((ROOT / "docs/figures/response-change-294").glob("*"))
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in ("publish294.py", "describe294.py", "verify294.py")
    )
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=294,
            retrospective=True,
            conditional_P_inputs=True,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Record integrity and arithmetic; no independent omitted-array or checkpoint replication.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("publish", "seal"))
    p.add_argument("--run", type=Path)
    p.add_argument("--cpu", type=Path)
    a = p.parse_args()
    if a.phase == "seal":
        seal()
    else:
        main(a.run.resolve(), a.cpu.resolve())
