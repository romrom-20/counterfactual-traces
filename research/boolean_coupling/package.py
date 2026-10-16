"""Collect release metadata; large activation arrays remain hash-inventoried locally."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"runs/baseline-interaction-225b"
DEST = ROOT/"results/baseline-interaction-225b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = json.loads((RUN/"audit.json").read_text())
    assert result["qualified"]
    design = json.loads((RUN/"design.json").read_text())
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    seal = json.loads((RUN/"qualification-seal.json").read_text())
    assert all(sha(RUN/p) == h for p, h in seal["files"].items())
    for frozen in ("frozen-225", "frozen-225b"):
        hashes = json.loads((ROOT/".context"/frozen/"source-hashes.json").read_text())
        assert all(sha(ROOT/p) == h for p, h in hashes.items())
    assert not (DEST/"manifest.json").exists(), "Preserve a completed release manifest."
    DEST.mkdir(parents=True, exist_ok=True)
    for path in RUN.rglob("*.json"):
        target = DEST/path.relative_to(RUN)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    raw = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in RUN.rglob("*.npz")]
    assert len(raw) == 65
    (DEST/"local-raw-inventory.json").write_text(json.dumps(dict(arrays=raw, scope="Not uploaded. Regenerate from published token sequences and pinned checkpoint."), indent=2)+"\n")
    logs = DEST/"logs"
    logs.mkdir(exist_ok=True)
    for name in ("225-qualification-model.log", "225b-qualification-model.log", "225b-qualification-audit.log", "225b-intervention-model.log", "225b-intervention-audit.log", "225b-presentation.log", "225b-public-plan.log"):
        shutil.copyfile(ROOT/".context"/name, logs/name)
    for frozen in ("frozen-225", "frozen-225b"):
        shutil.copytree(ROOT/".context"/frozen, DEST/frozen, dirs_exist_ok=True)
    failed = DEST/"failed-225"
    failed.mkdir(exist_ok=True)
    for path in (ROOT/"runs/baseline-interaction-225").glob("*.json"):
        shutil.copyfile(path, failed/path.name)
    rows = result["rows"]
    fields = ("site", "block", "schema", "reverse", "assignment", "target", "E00", "E10", "E01", "E11", "K_account", "B_account", "K_only_reverses", "B_only_reverses", "context")
    with (DEST/"cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {**row, **row["group"]}
            writer.writerow({k: flat[k] for k in fields})
    summary = json.loads((RUN/"summary.json").read_text())
    with (DEST/"behavior.csv").open("w", newline="") as handle:
        fields = ("context", "schema", "false_margin", "true_margin", "shift", "same_boolean_choice")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["behavior"]:
            flat = {**row, **row["group"]}
            writer.writerow({k: flat[k] for k in fields})
    print(f"Collected metadata, original logs and65 local raw-array hashes in {DEST}.")


if __name__ == "__main__":
    main()
