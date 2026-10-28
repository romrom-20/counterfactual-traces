"""Publish qualified295 scalar evidence and omitted-array identities."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/finite-drive-295"


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
            checkpoint_down_weight_sets=3,
            conditional_points=48,
            quadrature_nodes=[16, 32],
            partition_intervals=12,
            quadrature_product_kernel_batches=24 * 12 * (16 + 32),
            producer_down_batches=24 * 7,
            independent_down_batches=24 * 6,
            calculated_from_frozen_loop_counts=True,
            speedup_claimed=False,
            boundary="Dense host rational/product/down work and separate numerical audit; target supplied; not a useful compute comparison.",
        ),
    )


def seal():
    d = read(DEST / "design.json")
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in d["source_hashes"])
    files.add(ROOT / "docs/analysis/295-finite-drive.md")
    files.add(ROOT / "docs/analysis/research-direction-review-295.md")
    files.update((ROOT / "docs/figures/finite-drive-295").glob("*"))
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in ("publish295.py", "describe295.py", "verify295.py")
    )
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=295,
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
