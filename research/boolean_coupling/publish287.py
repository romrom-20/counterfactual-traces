"""Export scalar287/287b records, retain the failed gate, exclude arrays/drafts."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import copy, read, safe_relative, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = Path("results/source-receiver-readout-287")


def main(run, qualifier, cpu, cpu_qualifier, stopped):
    dest = ROOT / DEST
    design, status, audit = (read(run / (n + ".json")) for n in ("design", "status", "audit"))
    assert not design["synthetic"] and status["status"] == "complete"
    assert audit["status"] == "failed" and audit["failed_checks"] == 1
    for name in (
        "design.json",
        "status.json",
        "audit.json",
        "runtime.json",
        "head.json",
        *audit["csv_sha256"],
    ):
        copy(run / name, dest / name)
    qa = read(qualifier / "audit.json")
    assert qa["status"] == "qualified_boundary_only" and not qa["original_numeric_gate_passed"]
    for name in ("design.json", "audit.json", *qa["csv_sha256"]):
        copy(qualifier / name, dest / "boundary-qualification" / name)
    arrays, scores = [], []
    for entry in status["cases"]:
        p = run / "cases" / (entry["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"] and p.stat().st_size == entry["bytes"]
        assert sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        copy(p.with_suffix(".json"), dest / "cases" / p.with_suffix(".json").name)
        arrays.append(dict(path="cases/" + p.name, sha256=sha(p), bytes=p.stat().st_size))
        with np.load(p) as a:
            scores.append(
                dict(
                    key=entry["key"],
                    scores={n: a[n].tolist() for n in a.files if n.endswith("_scores")},
                )
            )
    assert sum(r["bytes"] for r in arrays) == status["raw_bytes"]
    write(
        dest / "scores.json",
        dict(
            experiment=287,
            cases=scores,
            boundary="Two candidate logits per finite full-vocabulary head; no activation vectors or weights.",
        ),
    )
    write(
        dest / "omitted-array-inventory.json",
        dict(
            experiment=287,
            arrays=arrays,
            case_bytes=status["raw_bytes"],
            additional_omitted=[
                dict(
                    path="head.npz",
                    sha256=sha(run / "head.npz"),
                    bytes=(run / "head.npz").stat().st_size,
                )
            ],
            boundary="Verified at export; public CI checks records, not omitted arrays or independent model replication.",
        ),
    )
    for source, name in ((cpu, "287"), (cpu_qualifier, "287b")):
        for p in sorted(source.glob("*.json")):
            copy(p, dest / "cpu-qualification" / name / p.name)
    assert not (stopped / "status.json").exists() and not list((stopped / "cases").iterdir())
    for name in ("design.json", "runtime.json", "head.json"):
        copy(stopped / name, dest / "retained-zero-case-stop" / name)
    write(
        dest / "retained-zero-case-stop/outcome.json",
        dict(
            status="resource_stop",
            completed_cases=0,
            ordinary_F_forwards=0,
            scientific_case_calls=0,
            reason="Original512MiB per-case disk guard, after model initialization and before first case.",
            source_or_threshold_changes=False,
            retry="New output directory; same original70 sources and all bounds.",
        ),
    )


def seal():
    dest = ROOT / DEST
    sources = read(dest / "boundary-qualification/design.json")["source_hashes"]
    extra = [ROOT / "docs/analysis/287-source-receiver-readout.md"]
    extra += list((ROOT / "docs/figures/source-receiver-readout-287").glob("*"))
    extra += [
        ROOT / "research/boolean_coupling" / n
        for n in ("publish287.py", "verify287.py", "describe287.py")
    ]
    files = {p for p in dest.rglob("*") if p.is_file() and p.name != "release-manifest.json"}
    files.update(ROOT / n for n in sources)
    files.update(extra)
    write(
        dest / "release-manifest.json",
        dict(
            experiment=287,
            retrospective=True,
            original_numeric_gate_passed=False,
            scientific_sources=sources,
            files={safe_relative(p, ROOT): sha(p) for p in sorted(files)},
            boundary="Original70 sources and posthoc73-source boundary qualification immutable. No article, arrays, fresh forecast, learned circuit or generalQB solution.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--qualifier", type=Path)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--cpu-qualifier", type=Path)
    p.add_argument("--stopped", type=Path)
    p.add_argument("--seal", action="store_true")
    a = p.parse_args()
    if a.seal:
        seal()
    else:
        assert all((a.run, a.qualifier, a.cpu, a.cpu_qualifier, a.stopped))
        main(
            a.run.resolve(),
            a.qualifier.resolve(),
            a.cpu.resolve(),
            a.cpu_qualifier.resolve(),
            a.stopped.resolve(),
        )
