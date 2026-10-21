"""Publish the retrospective 257 metadata; never activation arrays or weights."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json
from .package_attention import copy_evidence

ROOT = Path(__file__).resolve().parents[2]


def main(run, out, log):
    assert not (out/"manifest.json").exists() and json.loads((run/"audit.json").read_text())["status"] == "passed"
    out.mkdir(parents=True, exist_ok=True)
    copy_evidence(run, out)
    sources = ("research/boolean_coupling/source_radius257.py", "docs/experiments/257-source-receiver-radius.md")
    for name in sources:
        destination = out/"sources"/Path(name).name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    destination = out/"logs"/log.name
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(log, destination)
    extras = [ROOT/p for p in (*sources, "research/boolean_coupling/package_source_radius.py",
        "research/boolean_coupling/reproduce_source_radius.py", "docs/analysis/257-source-receiver-radius-result.md")]
    files = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(experiment=257, retrospective=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files+extras))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), args.log.resolve())
