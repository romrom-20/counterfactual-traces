"""Posthoc publication of275, excluding every activation array and private draft."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-attention-reading-275"


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
            else:
                assert path.suffix in (".json", ".csv")
                target = DEST / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        provenance, timings = [], {}
        for stage in ("producer", "audit"):
            original = logs / ("275-" + stage + ".log")
            content = original.read_text()
            assert "Traceback" not in content
            selected = [line for line in content.splitlines() if line.startswith(
                ("275 ", "{'native':", "real ", "user ", "sys "))]
            target = DEST / (stage + ".log")
            target.write_text("\n".join(selected) + "\n")
            provenance.append(dict(stage=stage, raw_sha256=sha(original), raw_bytes=original.stat().st_size,
                published_sha256=sha(target), representation="Progress, results and timing; environment warnings containing local paths omitted. Raw log preserved locally."))
            timings[stage] = {line.split()[0]: float(line.split()[1]) for line in content.splitlines()
                              if line.startswith(("real ", "user ", "sys "))}
        write(DEST / "log-provenance.json", dict(logs=provenance))
        write(DEST / "raw-array-inventory.json", dict(omitted=True, arrays=arrays,
            regeneration="research/README.md#source-attention-reading-275"))
        write(DEST / "release-notes.json", dict(retrospective=True, old_contexts=32,
            fresh_confirmation=False, private_articles_included=False,
            learned_circuit_identified=False, general_question_B_solved=False,
            novelty_priority_claimed=False, speedup_claimed=False,
            runtime_seconds=timings, counts=read(run / "status.json")["counts"],
            primary_scope="Old-context native source-query attention probability/self-value factorial. Full finite MLPs/receivers; extra native-K value-only arms. Probability includes Q and self-K changes. Not a fresh/cheap predictor or learned circuit."))
        return
    assert not (DEST / "manifest.json").exists()
    design = read(DEST / "design.json")
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == value
    extras = [ROOT / name for name in design["source_hashes"]]
    extras += [ROOT / name for name in (
        "docs/analysis/275-source-attention-reading.md",
        "research/boolean_coupling/package_source_reading.py",
        "research/boolean_coupling/describe_source_reading.py",
        "research/boolean_coupling/describe_first_source.py",
        "research/boolean_coupling/verify_source_reading.py")]
    extras += [p for p in (ROOT / "docs/figures/source-reading").iterdir() if p.is_file()]
    paths = [p for p in DEST.rglob("*") if p.is_file()] + extras
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in paths)
    write(DEST / "manifest.json", dict(release=DEST.name, scientific_freeze=design["source_commit"],
        scientific_sources=design["source_hashes"],
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        scope="Source/evidence/all-case integrity; not a fresh test or independent whole-model replication."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.run.resolve(), args.logs.resolve(), args.seal)
