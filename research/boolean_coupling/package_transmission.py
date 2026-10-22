"""Publish265/266 scientific records; keep raw tensors and private writing local."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("local-mixing-265", "local-mixing-continuation-265", "finite-silu-interaction-266")


def read(path):
    return json.loads(path.read_text())


def main(runs, out, seal):
    assert out.is_relative_to(ROOT/"results")
    if not seal:
        assert not out.exists()
        out.mkdir(parents=True)
        inventory = []
        for name in RUNS:
            folder = runs/name
            assert read(folder/"audit.json")["status"] == "passed"
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or "sources" in path.relative_to(folder).parts:
                    continue
                relative = path.relative_to(folder)
                if path.suffix in (".json", ".csv"):
                    target = out/name/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                else:
                    assert path.suffix == ".npz"
                    inventory.append(dict(path=str(Path(name)/relative), sha256=sha(path), bytes=path.stat().st_size))
        write_json(out/"raw-array-inventory.json", dict(omitted=True, arrays=inventory,
            regeneration="research/README.md#local-product-mixing-and-finite-activation-control"))
        write_json(out/"release-notes.json", dict(retrospective=True, fresh_examples=0, full_forwards=28,
            all_primary_hypotheses_pass=False, general_question_B_solved=False, private_articles_included=False,
            numerical_scope="Exact matched-shape publicFloat32 controls and checked smooth RMS components; all original CPU64 failures retained, no whole-model derivative certificate."))
        print("Copied265/266 public scientific records; raw arrays inventoried, private files excluded.")
        return
    assert not (out/"manifest.json").exists()
    sources = {}
    for name in RUNS:
        for path, digest in read(out/name/"design.json")["source_hashes"].items():
            assert sha(ROOT/path) == digest
            assert path not in sources or sources[path] == digest
            sources[path] = digest
    extras = [ROOT/p for p in ("docs/analysis/265-266-local-mixing-and-state-feedback.md",
        "research/boolean_coupling/package_transmission.py", "research/boolean_coupling/describe_transmission.py",
        "research/boolean_coupling/verify_transmission.py")]
    extras += [ROOT/name for name in sources]
    extras += [p for p in (ROOT/"docs/figures/transmission").iterdir() if p.is_file()]
    files = [p for p in out.rglob("*") if p.is_file()]+extras
    assert all(p.is_file() and "drafts" not in p.parts and p.suffix != ".npz" for p in files)
    write_json(out/"manifest.json", dict(release=out.name, retrospective=True, scientific_sources=sources,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
        limitation="Source/record integrity and all-case coverage, not independent tensor regeneration or model replication."))
    print("Sealed265/266 scientific evidence and presentation provenance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.runs.resolve(), args.out.resolve(), args.seal)
