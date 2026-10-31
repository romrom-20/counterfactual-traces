"""Publish retrospective scalar records; keep native arrays, weights and drafts local."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-interaction-300"


def publish(run, cpu):
    assert not DEST.exists()
    audit = read(run / "audit.json")
    assert audit["status"] == "passed" and not audit["scientific_outcomes_withheld"]
    for n in (
        "design.json",
        "runtime.json",
        "status.json",
        "audit.json",
        "checks.csv",
        "raw-coordinates.csv",
        "effects.csv",
        "vector-records.json",
        "local-records.json",
    ):
        copy(run / n, DEST / n)
    copy(cpu / "qualification.json", DEST / "cpu-qualification.json")
    status = read(run / "status.json")
    scores, inventory = [], []
    for e in status["cases"]:
        p = run / (e["key"] + ".npz")
        copy(p.with_suffix(".json"), DEST / "cases" / p.with_suffix(".json").name)
        with np.load(p) as z:
            scores.append(
                dict(
                    key=e["key"],
                    scores={n: z[n].tolist() for n in z.files if n.endswith("_scores")},
                )
            )
            arrays = {n: dict(shape=list(z[n].shape), dtype=str(z[n].dtype)) for n in z.files}
        inventory.append(dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size, arrays=arrays))
    write(DEST / "scores.json", dict(cases=scores))
    write(
        DEST / "omitted-array-inventory.json",
        dict(
            arrays=inventory,
            raw_bytes=status["raw_bytes"],
            native_parents_supplied=True,
            retrospective=True,
        ),
    )
    write(
        DEST / "cost.json",
        dict(
            checkpoint_full_forwards=48,
            repeated_native_calibrations=8,
            conditional_forwards=40,
            counts=status["counts"],
            host_audits_additional=True,
            speedup_claimed=False,
            general_question_B_solved=False,
        ),
    )
    (DEST / "README.md").write_text(
        "# Native query-interaction diagnostic (300)\n\n[Readable report](../../docs/analysis/300-query-interaction.md) · [Frozen protocol](../../docs/experiments/300-query-interaction.md)\n\nEight already observed native contexts; block15 only; finite downstream interventions.\nNo False-only forecast, new native population confirmation or learned-circuit claim.\nAll eight cases, including low-materiality failures, remain in the denominators.\nWeights, activation arrays and private drafts are omitted. The record verifier\nchecks source/parent integrity and arithmetic, not independent model replication.\n"
    )


def release():
    d = read(DEST / "design.json")
    files = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in d["source_hashes"])
    files.update(
        ROOT / "research/boolean_coupling" / n
        for n in ("publish300.py", "describe300.py", "verify300.py", "reproduce300.py")
    )
    files.add(ROOT / "docs/analysis/300-query-interaction.md")
    files.update((ROOT / "docs/figures/query-interaction-300").glob("*"))
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=300,
            retrospective=True,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Record/source/parent integrity and arithmetic; not omitted-array, whole-model or independent-team replication.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--release-only", action="store_true")
    args = parser.parse_args()
    if args.release_only:
        release()
    else:
        publish(args.run.resolve(), args.cpu.resolve())
