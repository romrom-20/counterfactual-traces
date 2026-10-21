"""Publish 255/256 complete metadata and sources; never arrays, weights or drafts."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json
from .package_attention import copy_evidence

ROOT = Path(__file__).resolve().parents[2]


def main(run, out, number, logs):
    assert not (out/"manifest.json").exists() and json.loads((run/"audit.json").read_text())["status"] == "passed"
    out.mkdir(parents=True, exist_ok=True)
    copy_evidence(run, out)
    for path in logs:
        destination = out/"logs"/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
    extras = [ROOT/p for p in ("research/boolean_coupling/package_local_query.py", "research/boolean_coupling/reproduce_local_query.py")]
    if number == 255:
        sources = ("research/boolean_coupling/curvature255.py", "docs/experiments/255-local-query-curvature.md")
        for name in sources:
            source = ROOT/name
            destination = out/"sources"/source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source, destination)
        extras.extend(ROOT/p for p in (*sources, "docs/analysis/255-local-query-curvature-result.md"))
    else:
        design = json.loads((run/"design.json").read_text())
        extras.extend(ROOT/p for p in design["source_hashes"])
        extras.extend(ROOT/p for p in ("research/boolean_coupling/describe_local_query.py", "docs/analysis/256-local-query-curvature-result.md",
            "research/boolean_coupling/radius256.py",
            "docs/figures/256/256-local-query-curvature.png", "docs/figures/256/256-local-query-curvature.svg"))
    files = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(experiment=number, release=out.name,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files+extras))}))
    print("Published complete local-query metadata; arrays, weights and unpublished drafts stay local.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", type=int, choices=(255, 256), required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), args.experiment, [p.resolve() for p in args.log])
