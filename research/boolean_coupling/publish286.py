"""Publish scalar records and inventories for 286/286b, never activations or drafts."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/false-readout-286")


def records(run, dest, qualifier=False):
    design, status, audit = (
        read(run / name) for name in ("design.json", "status.json", "audit.json")
    )
    assert not design["synthetic"] and status["status"] == "complete"
    assert audit["status"] in ("passed", "failed") if qualifier else audit["status"] == "passed"
    names = ["design.json", "status.json", "audit.json", "runtime.json"]
    names += list(audit["csv_sha256"])
    if not qualifier:
        names.append("head.json")
    for name in names:
        copy(run / name, dest / name)
    arrays, score_records = [], []
    for entry in status["cases"]:
        path = run / "cases" / (entry["key"] + ".npz")
        assert sha(path) == entry["arrays_sha256"] and path.stat().st_size == entry["bytes"]
        assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        copy(path.with_suffix(".json"), dest / "cases" / path.with_suffix(".json").name)
        arrays.append(dict(path="cases/" + path.name, sha256=sha(path), bytes=path.stat().st_size))
        if not qualifier:
            with np.load(path) as data:
                arms = (
                    "F",
                    "M",
                    "C",
                    "T",
                    "parallel_F",
                    "orthogonal_F",
                    "parallel_M",
                    "orthogonal_M",
                    "random0",
                    "random1",
                )
                score_records.append(
                    dict(
                        key=entry["key"],
                        runtime_scores={arm: data[arm + "_scores"].tolist() for arm in arms},
                        reference_scores={
                            arm: data["reference_" + arm + "_scores"].tolist()
                            for arm in ("F", "M", "C", "T")
                        },
                    )
                )
    assert status["raw_bytes"] == sum(row["bytes"] for row in arrays)
    extra = []
    if not qualifier:
        extra.append(
            dict(
                path="head.npz",
                sha256=sha(run / "head.npz"),
                bytes=(run / "head.npz").stat().st_size,
            )
        )
        write(
            dest / "scores.json",
            dict(
                experiment=286,
                cases=score_records,
                representation="Only the two recorded candidate logits from each finite full-vocabulary head; no weights or activation vectors.",
            ),
        )
    write(
        dest / "omitted-array-inventory.json",
        dict(
            experiment=design["experiment"],
            arrays=arrays,
            case_bytes=status["raw_bytes"],
            additional_omitted=extra,
            boundary="Omitted arrays retained locally, byte/hash verified when exporting. Public CI verifies records, not these arrays or a new model replication.",
        ),
    )


def main(run, qualifier, cpu, cpu_qualifier):
    dest = ROOT / DEST
    records(run, dest)
    records(qualifier, dest / "full-shape-qualifier", True)
    for source, name in ((cpu, "286"), (cpu_qualifier, "286b")):
        for p in sorted(source.glob("*.json")):
            copy(p, dest / "cpu-qualification" / name / p.name)


def seal():
    dest = ROOT / DEST
    sources = read(dest / "design.json")["source_hashes"]
    sources.update(read(dest / "full-shape-qualifier/design.json")["source_hashes"])
    artifacts = [ROOT / "docs/analysis/286-state-dependent-readout.md"]
    artifacts += list((ROOT / "docs/figures/false-readout-286").glob("*"))
    artifacts += [
        ROOT / "research/boolean_coupling" / name
        for name in ("publish286.py", "verify286.py", "describe286.py")
    ]
    paths = {p for p in dest.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    paths.update(ROOT / name for name in sources)
    paths.update(artifacts)
    write(
        dest / "release-manifest.json",
        dict(
            experiment=286,
            retrospective=True,
            scientific_sources=sources,
            files={safe_relative(path, ROOT): sha(path) for path in sorted(paths)},
            boundary="Scientific freezes remain immutable; final records, qualifier, presentation and posthoc delivery helpers are bound here. No prospective claim.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--qualifier", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--cpu-qualifier", type=Path)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    else:
        assert all((args.run, args.qualifier, args.cpu, args.cpu_qualifier))
        main(
            args.run.resolve(),
            args.qualifier.resolve(),
            args.cpu.resolve(),
            args.cpu_qualifier.resolve(),
        )
