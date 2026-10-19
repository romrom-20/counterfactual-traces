"""Post-run integrity check of240's raw prerequisites against prior release inventories."""

import argparse
import json
from pathlib import Path

from .core225b import sha, write_json


def main(run):
    assert not (run / "input-recheck.json").exists()
    root = Path(__file__).resolve().parents[2]
    design = json.loads((run / "design.json").read_text())
    status = json.loads((run / "status.json").read_text())
    records = []
    for tag, release, phases in (("228", "false-only-alignment-228", ("forecasts", "observations")),
        ("238", "upstream-remainder-238", ("cases",)), ("239", "generator-curvature-239", ("cases",))):
        folder = root / "results" / release
        inventory = json.loads((folder / "local-array-inventory.json").read_text())
        hashes = {r["path"]: r["sha256"] for r in inventory["arrays"]}
        for ent in status["cases"]:
            for phase in phases:
                name = phase + "/" + ent["key"] + ".npz"
                actual = sha(Path(design["input" + tag]) / name)
                records.append(dict(release=release, path=name, sha256=actual,
                    expected=hashes[name], passed=actual == hashes[name]))
    result = dict(status="passed" if all(r["passed"] for r in records) else "failed", retrospective=True,
        model_calls=0, source_sha256=sha(__file__), checks=records)
    write_json(run / "input-recheck.json", result)
    assert result["status"] == "passed", "Preserve the discrepancy; do not silently substitute inputs."
    print("Raw prerequisite hash comparisons:", len(records), result["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
