"""Publish 258/258b complete metadata and retained failure; never arrays or drafts."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json
from .package_attention import copy_evidence

ROOT = Path(__file__).resolve().parents[2]


def main(run, out, logs):
    assert not (out/"manifest.json").exists() and json.loads((run/"audit.json").read_text())["status"] == "passed"
    out.mkdir(parents=True, exist_ok=True)
    copy_evidence(run, out)
    for path in logs:
        destination = out/"logs"/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
    design = json.loads((run/"design.json").read_text())
    qualification = json.loads((run/"qualification258b.json").read_text())
    extras = [ROOT/p for p in (*design["source_hashes"], *qualification["source_hashes"],
        "research/boolean_coupling/reproduce_receiver_mlp.py", "research/boolean_coupling/describe_receiver_mlp.py",
        "research/boolean_coupling/package_receiver_mlp.py", "docs/analysis/258-receiver-mlp-result.md",
        "docs/figures/258/258-receiver-mlp.png", "docs/figures/258/258-receiver-mlp.svg",
        "docs/figures/258/258-local-capture.png", "docs/figures/258/258-local-capture.svg")]
    files = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(experiment=258, qualification="258b", release=out.name,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files+extras))}))
    print("Packaged all 258 outcomes and explicit qualification; raw failures retained, arrays/weights/articles local.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), [p.resolve() for p in args.log])
