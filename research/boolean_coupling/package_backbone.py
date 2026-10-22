"""Posthoc public evidence package for backbone and controller diagnostics270/271."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/backbone-feedback-270-271"
RUNS = {270: "backbone-curvature-feedback-270", 271: "amplitude-query-mediation-271"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(runs, logs, seal):
    if not seal:
        assert not DEST.exists()
        DEST.mkdir(parents=True)
        inventory, times, log_inventory = [], {}, []
        for number, name in RUNS.items():
            folder = runs / name
            assert read(folder / "audit.json")["status"] == "passed"
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or "sources" in path.relative_to(folder).parts:
                    continue
                rel = path.relative_to(folder)
                if path.suffix in (".json", ".csv"):
                    target = DEST / name / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                else:
                    assert path.suffix == ".npz"
                    inventory.append(dict(path=str(Path(name) / rel), sha256=sha(path),
                                          bytes=path.stat().st_size))
            times[str(number)] = {}
            for stage in ("producer", "audit"):
                source = logs / f"{number}-{stage}.log"
                content = source.read_text()
                assert "Traceback" not in content
                selected = [line for line in content.splitlines() if line.startswith(
                    (str(number) + " ", "{\'native\':", "real ", "user ", "sys "))]
                target = DEST / name / (stage + ".log")
                target.write_text("\n".join(selected) + "\n")
                log_inventory.append(dict(experiment=number, stage=stage,
                    raw_sha256=sha(source), raw_bytes=source.stat().st_size,
                    published_sha256=sha(target),
                    representation="Scientific progress/result/timing lines only; import warnings with local paths omitted. Original log preserved locally."))
                times[str(number)][stage] = {line.split()[0]: float(line.split()[1])
                    for line in content.splitlines() if line.startswith(("real ", "user ", "sys "))}
        extra = runs / "controller-clamp-checks-271"
        shutil.copytree(extra, DEST / extra.name)
        write(DEST / "log-provenance.json", dict(logs=log_inventory))
        write(DEST / "raw-array-inventory.json", dict(omitted=True, arrays=inventory,
            regeneration="research/README.md#backbone-curvature-and-feedback-270271"))
        write(DEST / "release-notes.json", dict(retrospective=True, old_contexts=4,
            fresh_examples=0, complete_forwards=76, runtime_seconds=times,
            general_question_B_solved=False, learned_circuit_identified=False,
            private_articles_included=False, speedup_claimed=False,
            distinction="270 removes finite backbone gate/cross terms;271 clamps B-forecast coefficients/queries, with B's extra work explicitly supplied. Neither is fresh confirmation or a cheap learned generator."))
        print("Packaged complete270/271 records; raw arrays inventoried, not published.")
        return
    assert not (DEST / "manifest.json").exists()
    scientific = {}
    freezes = {}
    for number, name in RUNS.items():
        design = read(DEST / name / "design.json")
        freezes[str(number)] = design["source_commit"]
        for path, value in design["source_hashes"].items():
            assert sha(ROOT / path) == value
            assert path not in scientific or scientific[path] == value
            scientific[path] = value
    extra = [ROOT / p for p in (
        "docs/analysis/270-271-backbone-curvature-and-feedback.md",
        "research/boolean_coupling/package_backbone.py",
        "research/boolean_coupling/describe_backbone.py",
        "research/boolean_coupling/verify_backbone.py",
        "research/boolean_coupling/clamps271.py")]
    extra += [ROOT / p for p in scientific]
    extra += [p for p in (ROOT / "docs/figures/backbone").iterdir() if p.is_file()]
    files = [p for p in DEST.rglob("*") if p.is_file()] + extra
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in files)
    write(DEST / "manifest.json", dict(release=DEST.name, retrospective=True,
        scientific_freezes=freezes, scientific_sources=scientific,
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        limitation="Source/record integrity and all-case decisions; not an independent model replication."))
    print("Sealed270/271 public evidence and posthoc report/figure/code.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.runs.resolve(), args.logs.resolve(), args.seal)
