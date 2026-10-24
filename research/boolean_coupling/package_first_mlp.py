"""Publish276/277 all-case metadata, receipts and plots; never activations or articles."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {276: "first-source-mlp-gain-276", 277: "first-source-mlp-forecast-277"}
REPORT = "docs/analysis/276-277-first-source-mlp.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(run, logs, number, seal):
    dest = ROOT / "results" / NAMES[number]
    if not seal:
        assert read(run / "audit.json")["status"] == "passed"
        assert not (dest / "audit.json").exists()
        if number == 276:
            assert not dest.exists()
        else:
            assert (dest / "pre-native/forecast-seal.json").exists()
        dest.mkdir(parents=True, exist_ok=True)
        arrays = []
        for path in sorted(run.rglob("*")):
            if not path.is_file() or "sources" in path.relative_to(run).parts:
                continue
            rel = path.relative_to(run)
            if path.suffix == ".npz":
                arrays.append(dict(path=str(rel), sha256=sha(path), bytes=path.stat().st_size))
                continue
            assert path.suffix in (".json", ".csv")
            target = dest / rel
            assert not target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        stages = ("producer", "audit") if number == 276 else ("producer", "forecast-audit", "native", "native-audit")
        provenance, timings = [], {}
        for stage in stages:
            original = logs / f"{number}-{stage}.log"
            content = original.read_text()
            assert "Traceback" not in content
            lines = [line for line in content.splitlines() if line.startswith((f"{number} ", "{'native':", "real ", "user ", "sys "))]
            target = dest / (stage + ".log")
            target.write_text("\n".join(lines) + "\n")
            provenance.append(dict(stage=stage, raw_sha256=sha(original), raw_bytes=original.stat().st_size,
                published_sha256=sha(target), representation="Progress, results and timing. Local environment-path warnings omitted; full log retained locally."))
            timings[stage] = {line.split()[0]: float(line.split()[1]) for line in content.splitlines()
                              if line.startswith(("real ", "user ", "sys "))}
        write(dest / "log-provenance.json", dict(logs=provenance))
        write(dest / "raw-array-inventory.json", dict(omitted=True, arrays=arrays,
            regeneration="research/README.md#first-source-mlp-276277"))
        write(dest / "release-notes.json", dict(experiment=number, retrospective=number == 276,
            fresh_literals=number == 277, independent_templates=False, private_articles_included=False,
            general_question_B_solved=False, learned_circuit_identified=False, novelty_priority_claimed=False,
            efficiency_claimed=False, runtime_seconds=timings,
            counts=read(run / ("status.json" if number == 276 else "forecast-status.json"))["counts"],
            boundary="Only source MLP1 is approximated/intervened. All other model computation finite. Predictive and gain-explanation decisions separate; failed276 explanation retained."))
        return
    assert not (dest / "manifest.json").exists()
    design = read(dest / "design.json")
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == value
    paths = [p for p in dest.rglob("*") if p.is_file()]
    paths += [ROOT / name for name in design["source_hashes"]]
    paths += [ROOT / name for name in (REPORT, "research/boolean_coupling/package_first_mlp.py",
        "research/boolean_coupling/describe_first_mlp.py", "research/boolean_coupling/verify_first_mlp.py",
        "research/boolean_coupling/repeat_first_mlp.py")]
    paths += [p for p in (ROOT / "docs/figures/first-mlp").iterdir() if p.is_file()]
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in paths)
    write(dest / "manifest.json", dict(release=dest.name, scientific_freeze=design["source_commit"],
        scientific_sources=design["source_hashes"], files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        scope="All-case evidence/source/receipt integrity; does not itself replicate the model."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--experiment", type=int, choices=(276, 277), required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.run.resolve(), args.logs.resolve(), args.experiment, args.seal)
