"""Posthoc publication of272; raw arrays and KV tensors remain local."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/causal-prefix-budget-272"


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
        log_records, timings = [], {}
        for stage in ("producer", "audit"):
            original = logs / ("272-" + stage + ".log")
            text = original.read_text()
            assert "Traceback" not in text
            selected = [line for line in text.splitlines() if line.startswith(
                ("272 ", "{'native':", "real ", "user ", "sys "))]
            target = DEST / (stage + ".log")
            target.write_text("\n".join(selected) + "\n")
            log_records.append(dict(stage=stage, raw_sha256=sha(original), raw_bytes=original.stat().st_size,
                                    published_sha256=sha(target), representation="Progress/result/timing lines; import warnings with local paths explicitly omitted. Original log preserved locally."))
            timings[stage] = {line.split()[0]: float(line.split()[1]) for line in text.splitlines()
                              if line.startswith(("real ", "user ", "sys "))}
        write(DEST / "log-provenance.json", dict(logs=log_records))
        write(DEST / "raw-array-inventory.json", dict(omitted=True, arrays=arrays,
            KV_tensors_saved=False, KV_regeneration="One ordinary False forward; prefix sizes/hashes in every case record.",
            regeneration="research/README.md#causal-prefix-budget-baseline-272"))
        write(DEST / "release-notes.json", dict(retrospective=True, old_contexts=32, fresh_confirmation=False,
            profiled_full_forwards=64, profiled_suffix_forwards=96, benchmark_full_forwards=96,
            benchmark_suffix_forwards=96, runtime_seconds=timings,
            general_question_B_solved=False, learned_circuit_identified=False,
            novelty_priority_claimed=False, private_articles_included=False,
            primary_scope="Near-native Float32 prediction and measured dense-work budget for a late single-token edit with one observed False KV cache. Exact reuse in real-arithmetic causal graph; not bit-exact Float32."))
        return
    assert not (DEST / "manifest.json").exists()
    design = read(DEST / "design.json")
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == value
    extras = [ROOT / name for name in design["source_hashes"]]
    extras += [ROOT / name for name in (
        "docs/analysis/272-research-review-and-causal-baseline.md",
        "research/boolean_coupling/review272.py", "research/boolean_coupling/package_cache.py",
        "research/boolean_coupling/describe_cache.py", "research/boolean_coupling/verify_cache.py")]
    extras += [p for p in (ROOT / "docs/figures/cache").iterdir() if p.is_file()]
    paths = [p for p in DEST.rglob("*") if p.is_file()] + extras
    assert all(p.suffix != ".npz" and "drafts" not in p.parts for p in paths)
    write(DEST / "manifest.json", dict(release=DEST.name, retrospective=True,
        scientific_freeze=design["source_commit"], scientific_sources=design["source_hashes"],
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        scope="Source/evidence/all-case integrity, not independent model replication or a learned-circuit certificate."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    main(args.run.resolve(), args.logs.resolve(), args.seal)
