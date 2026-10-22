"""Posthoc268/269 evidence packaging; preserve the pre-native receipt unchanged."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/"results/amplitude-direction-268-269"
FRESH = ROOT/"results/signed-amplitude-forecast-269"
OLD_RUNS = ("amplitude-direction-268", "directional-geometry-268")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+"\n")


def main(runs, seal):
    if not seal:
        assert not DEST.exists() or {p.name for p in DEST.iterdir()} == {"README.md"}
        assert read(runs/"amplitude-direction-268/audit.json")["status"] == "passed"
        assert read(runs/"signed-amplitude-forecast-269/audit.json")["status"] == "passed"
        DEST.mkdir(parents=True, exist_ok=True)
        inventory = []
        for name in (*OLD_RUNS, "signed-amplitude-forecast-269"):
            folder = runs/name
            for path in sorted(folder.rglob("*")):
                relative = path.relative_to(folder)
                if not path.is_file() or "sources" in relative.parts:
                    continue
                if path.suffix == ".npz":
                    inventory.append(dict(path=str(Path(name)/relative), sha256=sha(path), bytes=path.stat().st_size))
                    continue
                assert path.suffix in (".json", ".csv")
                if name == "signed-amplitude-forecast-269":
                    existing = FRESH/"pre-native"/relative
                    if existing.is_file():
                        assert sha(existing) == sha(path)
                        continue
                    target = FRESH/relative
                else:
                    target = DEST/name/relative
                assert not target.exists(), target
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        write(DEST/"raw-array-inventory.json", dict(omitted=True, arrays=inventory,
            regeneration="research/README.md#signed-amplitudes-and-directions-268269"))
        a268 = read(runs/"amplitude-direction-268/audit.json")
        a269 = read(runs/"signed-amplitude-forecast-269/audit.json")
        write(DEST/"release-notes.json", dict(old_contexts=4, fresh_literal_pairs=32,
            complete_forwards=400, geometry_model_calls=0,
            primary_268_pass=a268["primary_amplitudes_sufficient"], primary_269_pass=a269["primary_pass"],
            claims_separate=True, general_question_B_solved=False, learned_circuit_identified=False,
            private_articles_included=False,
            scope="268 interaction sufficiency and269 fresh native prediction are different claims; extra L forecast and substantial finite primitives remain."))
        print("Packaged268/269 all-case records; pre-native bytes unchanged, activation arrays inventoried.")
        return
    assert not (DEST/"manifest.json").exists()
    assert "DATA_PENDING" not in (ROOT/"docs/analysis/268-269-amplitudes-directions-and-prediction.md").read_text()
    scientific = {}
    for design in (DEST/"amplitude-direction-268/design.json", FRESH/"pre-native/design.json"):
        for name, value in read(design)["source_hashes"].items():
            assert sha(ROOT/name) == value and (name not in scientific or scientific[name] == value)
            scientific[name] = value
    extras = [ROOT/name for name in scientific]
    extras += [ROOT/p for p in ("docs/analysis/268-269-amplitudes-directions-and-prediction.md",
        "research/boolean_coupling/geometry268.py", "research/boolean_coupling/package_amplitude.py",
        "research/boolean_coupling/describe_amplitude.py", "research/boolean_coupling/verify_amplitude.py",
        "research/boolean_coupling/reproduce_amplitude.py")]
    extras += [p for p in (ROOT/"docs/figures/amplitude").iterdir() if p.is_file()]
    files = [p for folder in (DEST, FRESH) for p in folder.rglob("*") if p.is_file()]+extras
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in files)
    write(DEST/"manifest.json", dict(release=DEST.name, scientific_sources=scientific,
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        limitation="Integrity and all-case decisions, not an independent model replication or complete Question B solution."))
    print("Sealed scientific sources, old/fresh records and posthoc presentation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.runs.resolve(), args.seal)
