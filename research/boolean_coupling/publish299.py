"""Publish immutable 299 records; activation arrays and private drafts stay local."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-generators-299"


def inventory(run, phase, status):
    arrays = []
    folder = "forecasts" if phase == "pre-native" else "observations"
    tag = "forecast" if phase == "pre-native" else "native"
    for entry in status["cases"]:
        for name in [folder, "witnesses-" + tag]:
            path = run / name / (entry["key"] + ".npz")
            if path.exists():
                with np.load(path) as data:
                    spec = {
                        n: dict(shape=list(data[n].shape), dtype=str(data[n].dtype))
                        for n in data.files
                    }
                arrays.append(
                    dict(
                        path=name + "/" + path.name,
                        sha256=sha(path),
                        bytes=path.stat().st_size,
                        arrays=spec,
                    )
                )
    return arrays


def publish(run, cpu, phase):
    dest = DEST / phase
    assert not dest.exists()
    before = phase == "pre-native"
    audit = read(run / ("pre-audit.json" if before else "audit.json"))
    assert audit["status"] == "passed" and not audit["scientific_outcomes_withheld"]
    if before:
        assert not list((run / "observations").glob("*"))
        names = [
            "design.json",
            "runtime.json",
            "forecast-status.json",
            "pre-audit.json",
            "forecast-seal.json",
            "pre-checks.csv",
            "pre-raw-queries.csv",
        ]
        copy(cpu / "qualification.json", dest / "cpu-qualification.json")
    else:
        names = [
            "native-status.json",
            "measurement-proof.json",
            "audit.json",
            "checks.csv",
            "raw-queries.csv",
            "native.csv",
            "vector-records.json",
            "layer-records.json",
            "local-records.json",
        ]
    for name in names:
        copy(run / name, dest / name)
    status = read(run / ("forecast-status.json" if before else "native-status.json"))
    folder = "forecasts" if before else "observations"
    scores = []
    for entry in status["cases"]:
        path = run / folder / (entry["key"] + ".npz")
        copy(path.with_suffix(".json"), dest / "cases" / path.with_suffix(".json").name)
        with np.load(path) as data:
            scores.append(
                dict(
                    key=entry["key"],
                    scores={n: data[n].tolist() for n in data.files if n.endswith("_scores")},
                )
            )
    write(dest / "scores.json", dict(cases=scores))
    write(
        dest / "omitted-array-inventory.json",
        dict(
            arrays=inventory(run, phase, status),
            raw_bytes=status["raw_bytes"],
            boundary="Hashes/shapes/metadata only; original activation arrays remain local.",
        ),
    )
    if not before:
        forecast = read(run / "forecast-status.json")
        write(
            dest / "cost.json",
            dict(
                total_checkpoint_full_forwards=forecast["counts"]["full_forwards"]
                + status["counts"]["full_forwards"],
                completed_forecast_counts=forecast["counts"],
                native_counts=status["counts"],
                ordinary_native_true_calls=32,
                total_extra_full_RMSNorm=sum(
                    s["counts"]["extra_full_RMSNorm"] for s in [forecast, status]
                ),
                total_witness_query_projection=sum(
                    s["counts"]["witness_extra_full_query_projection"] for s in [forecast, status]
                ),
                total_witness_QK=sum(
                    s["counts"]["witness_extra_full_QK"] for s in [forecast, status]
                ),
                total_witness_softmax=sum(
                    s["counts"]["witness_extra_full_softmax"] for s in [forecast, status]
                ),
                total_witness_PV=sum(
                    s["counts"]["witness_extra_full_PV"] for s in [forecast, status]
                ),
                query_weights_bytes=(run / "query-weights.npz").stat().st_size,
                head_bytes=(run / "head.npz").stat().st_size,
                host_audits_are_additional=True,
                wallclock_speedup_claimed=False,
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
        )
    (dest / "README.md").write_text(
        "# 299 " + phase + "\n\n"
        "Immutable records for previous-block carry versus most-recent MLP query operands.\n"
        "The pre-native release precedes ordinary True measurements. Native outcomes\n"
        "and retrospective common-parent diagnostics are separately sealed afterward.\n"
        "Required output checks and original raw-logit diagnostics remain separate.\n"
        "Metadata verifiers do not reconstruct omitted arrays or independently repeat\n"
        "the checkpoint. All work and original failures remain explicit.\n"
    )


def release(phase):
    dest = DEST / phase
    design = read(DEST / "pre-native/design.json")
    files = {p for p in dest.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / name for name in design["source_hashes"])
    if phase == "native-phase":
        files.add(DEST / "pre-native/release-manifest.json")
        files.add(ROOT / "docs/analysis/299-query-generators.md")
        files.update((ROOT / "docs/figures/query-generators-299").glob("*"))
        files.update(
            ROOT / "research/boolean_coupling" / name
            for name in ["publish299.py", "describe299.py", "verify299.py", "reproduce299.py"]
        )
    write(
        dest / "release-manifest.json",
        dict(
            experiment=299,
            phase=phase,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Record integrity, source ancestry and scalar/vector arithmetic; not independent model or activation-array replication.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--phase", choices=["pre-native", "native-phase"], required=True)
    parser.add_argument("--release-only", action="store_true")
    args = parser.parse_args()
    if not args.release_only:
        publish(args.run.resolve(), args.cpu.resolve(), args.phase)
    if args.phase == "pre-native" or args.release_only:
        release(args.phase)
