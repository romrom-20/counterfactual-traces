"""Package259's retrospective evidence without arrays, weights or private writing."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def main(run, out, logs):
    assert not (out/"manifest.json").exists()
    audit = json.loads((run/"audit.json").read_text())
    assert audit["status"] == "passed" and audit["model_calls"] == 0
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(run.iterdir()):
        assert path.is_file() and path.suffix in (".json", ".csv")
        shutil.copy2(path, out/path.name)
    for path in logs:
        destination = out/"logs"/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
    extras = [ROOT/p for p in (
        "docs/experiments/259-silu-score-factors.md", "docs/analysis/259-silu-score-factors-result.md",
        "research/boolean_coupling/score259.py", "research/boolean_coupling/describe_score_factors.py",
        "research/boolean_coupling/package_score_factors.py", "research/boolean_coupling/reproduce_receiver_mlp.py",
        "docs/figures/259/259-score-factors.png", "docs/figures/259/259-score-factors.svg")]
    files = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(experiment=259, retrospective=True, release=out.name,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files+extras))}))
    print("Packaged all259 fixed-input score records and checks; no native forecasts or private files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), [p.resolve() for p in args.log])
