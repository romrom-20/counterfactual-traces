"""Bind the explicit pre-native numerical boundary without modifying forecasts."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def main(run):
    out = run/"qualification258b.json"
    assert not out.exists() and not (run/"status.json").exists() and not (run/"forecast-seal.json").exists()
    assert not list((run/"observations").glob("*"))
    design = json.loads((run/"design.json").read_text())
    assert json.loads((run/"forecast-status.json").read_text())["ordinary_true"] == 0
    sources = ("research/boolean_coupling/audit258b.py", "research/boolean_coupling/qualify258b.py",
               "research/boolean_coupling/replay258.py", "docs/experiments/258b-matched-projection-qualification.md")
    write_json(out, dict(created_utc=datetime.now(timezone.utc).isoformat(), ordinary_true=0,
        design_sha256=sha(run/"design.json"), forecast_status_sha256=sha(run/"forecast-status.json"),
        source_hashes={p:sha(ROOT/p) for p in sources},
        qualification_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        original_scorer_sha256=design["source_hashes"]["research/boolean_coupling/audit258.py"],
        original_failure_sha256=sha(run/"failure-audit-pre.json") if (run/"failure-audit-pre.json").exists() else None,
        investigation_sha256=sha(run/"projection-investigation.json") if (run/"projection-investigation.json").exists() else None,
        note="Original absolute whole-primal check failed; raw failures retained, exact full-shape Float32 replay required. Forecasts/criteria unchanged."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve())
