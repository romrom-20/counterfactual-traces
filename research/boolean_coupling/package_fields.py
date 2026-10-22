"""Posthoc publication of267, including its storage-aborted unchanged-source attempt."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/"results/correction-field-feedback-267"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+"\n")


def main(runs, seal):
    if not seal:
        assert not DEST.exists()
        run = runs/"field-state-267-retry"
        assert read(run/"audit.json")["status"] == "passed"
        DEST.mkdir(parents=True)
        inventory = []
        for name in ("field-state-267", "field-state-267-retry"):
            folder = runs/name
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or "sources" in path.relative_to(folder).parts:
                    continue
                rel = path.relative_to(folder)
                if path.suffix in (".json", ".csv"):
                    target = DEST/name/rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                else:
                    assert path.suffix == ".npz"
                    inventory.append(dict(path=str(Path(name)/rel), sha256=sha(path), bytes=path.stat().st_size))
        write(DEST/"raw-array-inventory.json", dict(omitted=True, arrays=inventory,
            regeneration="research/README.md#finite-correction-field-factorial-267"))
        write(DEST/"release-notes.json", dict(retrospective=True, fresh_examples=0,
            planned_retained_forwards=80, additional_unretained_forwards=20,
            original_attempt="Storage guard stopped after first context before its case arrays were written. The 20 forward completions were observed in tool output; no numerical outcome from that attempt is claimed.",
            retry="Same frozen source and protocol, new output directory after exact copy-on-write storage sharing of duplicate tracked checkouts. No activation arrays were deleted or modified.",
            general_question_B_solved=False, learned_circuit_identified=False, private_articles_included=False))
        print("Packaged267's complete numerical evidence and original storage failure; arrays inventoried.")
        return
    assert not (DEST/"manifest.json").exists()
    design = read(DEST/"field-state-267-retry/design.json")
    scientific = design["source_hashes"]
    assert scientific == read(DEST/"field-state-267/design.json")["source_hashes"]
    for name, value in scientific.items():
        assert sha(ROOT/name) == value
    extras = [ROOT/p for p in ("docs/analysis/267-correction-field-feedback.md",
        "research/boolean_coupling/package_fields.py", "research/boolean_coupling/describe_fields.py",
        "research/boolean_coupling/verify_fields.py")]
    extras += [ROOT/name for name in scientific]
    extras += [p for p in (ROOT/"docs/figures/fields").iterdir() if p.is_file()]
    files = [p for p in DEST.rglob("*") if p.is_file()]+extras
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in files)
    write(DEST/"manifest.json", dict(release=DEST.name, retrospective=True,
        scientific_freeze_commit=design["source_commit"], scientific_sources=scientific,
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        limitation="Record/source integrity and all-case accounting, not independent model or tensor replication."))
    print("Sealed267 public evidence and posthoc presentation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.runs.resolve(), args.seal)
