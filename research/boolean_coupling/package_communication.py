"""Publish250b receipts and scientific code; retain arrays and article drafts locally."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def main(run, out, logs):
    assert not out.exists() and json.loads((run/"audit.json").read_text())["status"] == "passed"
    out.mkdir(parents=True)
    arrays = []
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run)
        if path.suffix == ".npz":
            arrays.append(dict(path=str(rel), size_bytes=path.stat().st_size, sha256=sha(path)))
            continue
        dest = out/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(out/"local-array-inventory.json", dict(arrays=arrays, published=False,
        note="All32 intervention arrays and all32 expanded primitive witnesses remain local. Reproduction does not require this private archive."))
    for path in logs:
        dest = out/"logs"/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (out/"README.md").write_text("""# Native source communication — qualified250b

All32 old contexts, all11 ordinary/intervened arms, a separate finite suffix
identity control, three matched random directions and a saved-array auditor.
The primary continued-communication necessity criterion passes32/32.

Removing source values has much greater effects than removing source keys.
Direct verdict reads matter more than intermediate reads, but both contribute
and their effects interact. This is retrospective causal evidence under the
defined clamps, not a learned circuit, fresh prediction or solved Question B.

The original250 saved witnesses for32 contexts instead of four and stopped on
its final count assertion after all model work. Original failure/progress/logs
and sources are preserved. `completion250b.json` binds the narrowly scoped
count-only correction; no arrays, cases, tolerances or outcome criteria changed.

- `design.json`, `completion250b.json`, `status.json`, `seal.json`: scientific
  freeze, retained original failure, corrected accounting and final data seal.
- `cases/*.json`: every outcome-array hash and all original numerical checks.
- `outcomes.csv`: all signed verdict/full-receiver endpoints and margin effects.
- `signed-contrasts.csv`: every layer's K/V and direct/relay finite contrasts,
  with both interaction-allocation and raw-second-difference sign conventions.
- `primary.csv`, `audit.json`: every primary result and all3,776 audit checks.
- `description.json`: posthoc summaries bound to the audited CSVs.
- `sources/`, `logs/`, `runtime.json`: exact executed scientific code and records.
- `local-array-inventory.json`: SHA256 and size of every retained local array;
  no model weights, activation arrays or unpublished article drafts are uploaded.

See the [report](../../docs/analysis/250b-native-source-communication-result.md),
[original protocol](../../docs/experiments/250-native-source-communication.md),
[count-only correction](../../docs/experiments/250b-witness-completion-correction.md),
and [reproduction guide](../../research/README.md).

352 full forwards;608 H suffix blocks;32 H full-vocabulary heads;5,472 extra
QK/softmax/PV recomputations each;1,056 extra witness QK products. Numerical
auditing is additional. The bounded independent primitive audit covers all32
contexts at blocks10,19,28, not every full forward or every suffix block.
""")
    extras = [ROOT/p for p in (
        "research/boolean_coupling/reproduce_communication.py", "research/boolean_coupling/describe_communication.py",
        "research/boolean_coupling/package_communication.py", "docs/analysis/250b-native-source-communication-result.md",
        "docs/figures/250b/250-source-communication.png", "docs/figures/250b/250-source-communication.svg")]
    design = json.loads((run/"design.json").read_text())
    completion = json.loads((run/"completion250b.json").read_text())
    extras.extend(ROOT/p for p in {*design["source_hashes"], *completion["source_hashes"]})
    published = [p for p in out.rglob("*") if p.is_file()]
    write_json(out/"manifest.json", dict(release="native-source-communication-250b", retrospective=True,
        original_source_freeze_commit=design["source_freeze_commit"], correction_source_freeze_commit=completion["source_freeze_commit"],
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(published+extras))}))
    print("Packaged complete250b receipts without arrays, weights or articles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve(), [p.resolve() for p in args.log])
