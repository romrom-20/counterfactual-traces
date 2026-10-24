"""Publish279 source activation forecasts, preserving its receipt and every failure."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-activation-forecast-279"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(run, logs, seal):
    if not seal:
        assert read(run / "audit.json")["status"] == "passed"
        assert not (DEST / "audit.json").exists() and (DEST / "pre-native/forecast-seal.json").exists()
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
            assert not target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        provenance, timings = [], {}
        for stage in ("producer", "forecast-audit", "native", "native-audit"):
            original = logs / ("279-" + stage + ".log")
            content = original.read_text()
            assert "Traceback" not in content
            selected = [line for line in content.splitlines() if line.startswith(("279 ", "{'native':", "real ", "user ", "sys "))]
            target = DEST / (stage + ".log")
            target.write_text("\n".join(selected) + "\n")
            provenance.append(dict(stage=stage, raw_sha256=sha(original), raw_bytes=original.stat().st_size,
                published_sha256=sha(target), representation="Scientific progress/results/timing; local-path warnings omitted. Original log retained locally."))
            timings[stage] = {line.split()[0]: float(line.split()[1]) for line in content.splitlines() if line.startswith(("real ", "user ", "sys "))}
        write(DEST / "log-provenance.json", dict(logs=provenance))
        write(DEST / "raw-array-inventory.json", dict(omitted=True, arrays=arrays, regeneration="research/README.md#source-activation-forecast-279"))
        write(DEST / "release-notes.json", dict(experiment=279, prospective=True, fresh_literals=True,
            independent_templates=False, private_articles_included=False, general_question_B_solved=False,
            learned_circuit_identified=False, novelty_priority_claimed=False, efficiency_claimed=False,
            counts=read(run / "forecast-status.json")["counts"], runtime_seconds=timings,
            boundary="One source activation approximated from False coefficients, native other operations finite.29/32 joint includes three materiality failures even exact reference; sparse correction necessity fails."))
        return
    assert not (DEST / "manifest.json").exists()
    design = read(DEST / "design.json")
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == h
    paths = [p for p in DEST.rglob("*") if p.is_file()]
    paths += [ROOT / name for name in design["source_hashes"]]
    paths += [ROOT / name for name in ("docs/analysis/279-source-activation-forecast.md",
        "research/boolean_coupling/package_source_activation.py", "research/boolean_coupling/describe_source_activation.py",
        "research/boolean_coupling/verify_source_activation.py", "research/boolean_coupling/repeat_source_activation.py")]
    paths += [p for p in (ROOT / "docs/figures/source-activation").iterdir() if p.is_file()]
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in paths)
    write(DEST / "manifest.json", dict(release=DEST.name, scientific_freeze=design["source_commit"], scientific_sources=design["source_hashes"],
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}, scope="All-case forecast/receipt/source integrity, not an independent model replication."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.run.resolve(), args.logs.resolve(), args.seal)
