"""Release261–264 scientific metadata; exclude activations, weights and private writing."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("mlp-input-origin-261", "mlp-input-origin-261b", "mlp-input-replay-261c",
    "mlp-input-qualified-261c", "mlp-input-bounded-261d", "gate-curvature-factors-262",
    "normalization-branches-energy-263", "normalization-component-interaction-264",
    "normalization-component-interaction-264b")


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
            assert folder.is_dir()
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or "sources" in path.relative_to(folder).parts:
                    continue
                relative = path.relative_to(folder)
                if path.suffix in (".json", ".csv"):
                    target = out/name/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                else:
                    assert path.suffix == ".npz", path
                    inventory.append(dict(path=str(Path(name)/relative), bytes=path.stat().st_size, sha256=sha(path)))
        write_json(out/"raw-array-inventory.json", dict(omitted=True, arrays=inventory,
            regeneration="See docs/analysis/261-264-mlp-normalization-origin.md and research/README.md; never run the preserved invalid input261.py."))
        write_json(out/"release-notes.json", dict(retrospective=True, fresh_examples=0,
            model_forwards=dict(original_invalid_261=28, corrected_261b=28, intervention_263=44, analyses_262_264=0),
            invalid_attempts=dict(original_261="Wrong metadata destination and invalid auditor import; exact originals restored, corrected run independently replayed.",
                original_264="Final contrasts correct; layerwise records omit block1 and mislabel blocks2–28 as1–27. Use264b."),
            numerical_scope="Qualified publicFloat32 operations and checked smooth RMS components; raw CPU64 failures remain failures; older whole-block derivative disputes unresolved.",
            private_files_included=False, general_question_B_solved=False))
        print("Copied public scientific records and inventoried omitted arrays; seal after figures and report.")
        return
    assert (out/"raw-array-inventory.json").exists() and not (out/"manifest.json").exists()
    extras = [ROOT/"docs/analysis/261-264-mlp-normalization-origin.md"]
    extras += list((ROOT/"docs/experiments").glob("26[1-4]*.md"))
    extras += [ROOT/"research/boolean_coupling"/name for name in (
        "input261.py", "input261b.py", "audit261.py", "audit261b.py", "audit261c.py", "audit261d.py",
        "replay261c.py", "factors262.py", "input263.py", "audit263.py", "interaction264.py", "interaction264b.py",
        "package_normalization.py", "normalization_figures.py", "verify_normalization.py", "reproduce_normalization.py")]
    extras += [p for p in (ROOT/"docs/figures/normalization").iterdir() if p.is_file()]
    files = [p for p in out.rglob("*") if p.is_file()]+extras
    assert all(p.is_file() and "drafts" not in p.parts and p.suffix != ".npz" for p in files)
    sources = {}
    for name in ("mlp-input-origin-261", "mlp-input-origin-261b", "normalization-branches-energy-263"):
        for path, digest in read(out/name/"design.json")["source_hashes"].items():
            assert sha(ROOT/path) == digest
            assert path not in sources or sources[path] == digest
            sources[path] = digest
    write_json(out/"manifest.json", dict(release=out.name, retrospective=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}, scientific_sources=sources,
        limitation="Integrity of the published evidence, not independent activation regeneration or model replication."))
    print("Sealed complete normalization release; no arrays, checkpoint or private writing included.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.runs.resolve(), args.out.resolve(), args.seal)
