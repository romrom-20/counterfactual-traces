"""Publish retrospective298 records and conditional diagnostics, never arrays."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .audit272 import table
from .publish285 import copy, read, safe_relative, sha, write
from .subspace293 import statistics

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/context-messages-298"


def publish(run, cpu):
    assert not DEST.exists()
    audit = read(run / "audit.json")
    assert audit["status"] == "passed"
    assert read(run / "numerical-review.json")["passed"]
    for name in [
        "design.json",
        "runtime.json",
        "status.json",
        "audit.json",
        "checks.csv",
        "raw-coordinate.csv",
        "effects.csv",
        "joint.csv",
        "vectors.json",
        "numerical-review.json",
    ]:
        copy(run / name, DEST / name)
    copy(cpu / "qualification.json", DEST / "cpu-qualification.json")
    status = read(run / "status.json")
    design = read(run / "design.json")
    arrays = []
    scores = []
    messages = []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        p = run / "cases" / (entry["key"] + ".npz")
        copy(p.with_suffix(".json"), DEST / "cases" / p.with_suffix(".json").name)
        with np.load(p) as data:
            arrays.append(
                dict(
                    path="cases/" + p.name,
                    sha256=sha(p),
                    bytes=p.stat().st_size,
                    arrays={
                        n: dict(shape=list(data[n].shape), dtype=str(data[n].dtype))
                        for n in data.files
                    },
                )
            )
            scores.append(
                dict(
                    key=entry["key"],
                    scores={n: data[n].tolist() for n in data.files if n.endswith("_scores")},
                )
            )
            for group in [
                "chosen",
                "swapped",
                "union",
                "complement",
                "neighbor",
                "random1",
                "random2",
                "all",
                "zero",
            ]:
                for scope, index in [("receiving", slice(None)), ("verdict", -1)]:
                    messages.append(
                        dict(
                            key=case["key"],
                            schema=case["group"]["schema"],
                            layout=case["group"]["block"],
                            group=group,
                            scope=scope,
                            **statistics(data["delta_" + group][index], data["delta_all"][index]),
                        )
                    )
    write(
        DEST / "omitted-array-inventory.json",
        dict(
            arrays=arrays,
            raw_bytes=status["raw_bytes"],
            boundary="Local archives retained; public hashes/shapes are not independent reconstruction.",
        ),
    )
    write(DEST / "scores.json", dict(cases=scores))
    write(DEST / "message-records.json", dict(posthoc=True, statistics=messages))
    with (run / "effects.csv").open() as stream:
        effects = list(csv.DictReader(stream))
    scalar = []
    for case in design["cases"]:
        for group in [
            "chosen",
            "swapped",
            "union",
            "complement",
            "neighbor",
            "random1",
            "random2",
            "all",
            "zero",
        ]:
            rr = [r for r in effects if r["key"] == case["key"] and r["group"] == group]
            assert len(rr) == 2
            scalar.append(
                dict(
                    key=case["key"],
                    schema=case["group"]["schema"],
                    layout=case["group"]["block"],
                    group=group,
                    passes=all(
                        abs(float(r["native_margin_effect"])) >= 0.02
                        and float(r["margin_error"]) <= float(r["margin_bound"])
                        for r in rr
                    ),
                )
            )
    table(DEST / "scalar-component.csv", scalar)
    write(
        DEST / "cost.json",
        dict(
            checkpoint_full_transformer_forwards=0,
            new_false_or_true_full_passes=0,
            counts=status["counts"],
            supplied_native_parent_cases=8,
            retrospective=True,
            model_components_loaded=[
                "last_MLP",
                "last_MLP_RMSNorm",
                "attention_output_matrix",
                "final_RMSNorm",
                "full_vocabulary_head",
            ],
            raw_bytes=status["raw_bytes"],
            one_shared_checkpoint=True,
            host_projection_and_audit_work_additional=True,
            no_compact_counterfactual_predictor_claim=True,
        ),
    )


def release():
    d = read(DEST / "design.json")
    review = read(DEST / "numerical-review.json")
    paths = {p for p in DEST.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    paths.update(ROOT / n for n in d["source_hashes"])
    paths.update(ROOT / n for n in review["source_hashes"])
    paths.add(ROOT / "docs/analysis/298-context-messages.md")
    paths.update((ROOT / "docs/figures/context-messages-298").glob("*"))
    paths.update(
        ROOT / "research/boolean_coupling" / n
        for n in ["publish298.py", "describe298.py", "verify298.py", "reproduce298.py"]
    )
    write(
        DEST / "release-manifest.json",
        dict(
            experiment=298,
            retrospective=True,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(paths)},
            boundary="Source/record integrity and outcome arithmetic, not fresh prediction or independent checkpoint replay.",
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
