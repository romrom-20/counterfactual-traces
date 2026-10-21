"""Publish forecast receipts or complete251 evidence, never activation arrays."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def copy_evidence(run, dest):
    inventory = []
    for p in sorted(run.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run)
        if p.suffix == ".npz":
            inventory.append(dict(path=str(rel), size_bytes=p.stat().st_size, sha256=sha(p)))
            continue
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    write_json(dest/"local-array-inventory.json", dict(arrays=inventory, published=False,
        note="Weights, False/True probability tables, trajectories and primitive witnesses retained locally; reproduction generates new arrays."))


def main(run, out, phase, failed, logs):
    out.mkdir(parents=True, exist_ok=True)
    if phase == "pre":
        assert not (out/"pre-native").exists()
        assert (run/"forecast-seal.json").exists() and not (run/"status.json").exists()
        dest = out/"pre-native"
        copy_evidence(run, dest)
        write_json(dest/"receipt-manifest.json", dict(phase="before_native_True", ordinary_true=0,
            files={str(p.relative_to(ROOT)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}))
        print("Prepared source-bound pre-native forecast receipts; no activation arrays.")
        return
    assert not (out/"manifest.json").exists() and json.loads((run/"audit.json").read_text())["status"] == "passed"
    copy_evidence(run, out)
    if failed:
        assert (failed/"failure-forecast.json").exists() and not (failed/"forecast-status.json").exists()
        copy_evidence(failed, out/"preserved-disk-failure")
    for p in logs:
        dest = out/"logs"/p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    extras = [ROOT/p for p in (
        "research/boolean_coupling/reproduce_attention.py", "research/boolean_coupling/describe_attention.py",
        "research/boolean_coupling/package_attention.py", "docs/analysis/251-fixed-receiver-attention-result.md",
        "docs/figures/251/251-fixed-attention.png", "docs/figures/251/251-fixed-attention.svg")]
    design = json.loads((run/"design.json").read_text())
    extras.extend(ROOT/p for p in design["source_hashes"])
    published = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(release="fixed-receiver-attention-251", fresh_literals=True, source_freeze_commit=design["source_freeze_commit"],
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(published+extras))}))
    print("Packaged complete251 evidence, pre-native receipts and retained first-attempt failure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "final"), required=True)
    parser.add_argument("--failed", type=Path)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), args.phase, args.failed.resolve() if args.failed else None, [p.resolve() for p in args.log])
