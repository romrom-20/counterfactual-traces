"""Posthoc publication for278, excluding activations, weights and private writing."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-mlp-primitives-278"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(run, logs, seal):
    if not seal:
        assert not DEST.exists() and read(run / "audit.json")["status"] == "passed"
        DEST.mkdir(parents=True)
        arrays = []
        for path in sorted(run.rglob("*")):
            if not path.is_file() or "sources" in path.relative_to(run).parts:
                continue
            rel = path.relative_to(run)
            if path.suffix == ".npz":
                arrays.append(dict(path=str(rel), sha256=sha(path), bytes=path.stat().st_size))
                continue
            assert path.suffix in (".json", ".csv")
            target = DEST / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        provenance, timings = [], {}
        for stage in ("producer", "audit"):
            original = logs / ("278-" + stage + ".log")
            content = original.read_text()
            assert "Traceback" not in content
            selected = [line for line in content.splitlines() if line.startswith(("278 ", "{'native':", "real ", "user ", "sys "))]
            target = DEST / (stage + ".log")
            target.write_text("\n".join(selected) + "\n")
            provenance.append(dict(stage=stage, raw_sha256=sha(original), raw_bytes=original.stat().st_size,
                published_sha256=sha(target), representation="Scientific progress/results/timing. Local-path environment warnings omitted; original log preserved locally."))
            timings[stage] = {line.split()[0]: float(line.split()[1]) for line in content.splitlines()
                              if line.startswith(("real ", "user ", "sys "))}
        write(DEST / "log-provenance.json", dict(logs=provenance))
        write(DEST / "raw-array-inventory.json", dict(omitted=True, arrays=arrays,
            regeneration="research/README.md#source-mlp-primitives-278"))
        write(DEST / "release-notes.json", dict(experiment=278, retrospective=True, old_cases=32,
            fresh_confirmation=False, private_articles_included=False, general_question_B_solved=False,
            learned_circuit_identified=False, novelty_priority_claimed=False, efficiency_claimed=False,
            counts=read(run / "status.json")["counts"], runtime_seconds=timings,
            boundary="One first-source MLP primitive factorial at native/carrier branch inputs. Native outer residual and all receiving/readout work finite. Signed factorial is not a unique learned-circuit attribution."))
        return
    assert not (DEST / "manifest.json").exists()
    design = read(DEST / "design.json")
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == h
    paths = [p for p in DEST.rglob("*") if p.is_file()]
    paths += [ROOT / name for name in design["source_hashes"]]
    paths += [ROOT / name for name in ("docs/analysis/278-source-mlp-primitives.md",
        "research/boolean_coupling/package_source_primitives.py", "research/boolean_coupling/describe_source_primitives.py",
        "research/boolean_coupling/verify_source_primitives.py")]
    paths += [p for p in (ROOT / "docs/figures/source-primitives").iterdir() if p.is_file()]
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in paths)
    write(DEST / "manifest.json", dict(release=DEST.name, scientific_freeze=design["source_commit"],
        scientific_sources=design["source_hashes"], files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        scope="Source/all-case/signed-ledger/publication integrity; not an independent model replication."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.run.resolve(), args.logs.resolve(), args.seal)
