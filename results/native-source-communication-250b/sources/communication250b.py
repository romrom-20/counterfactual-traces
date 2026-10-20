"""Recover only250's complete-data witness-count failure, retaining original bytes."""

import argparse
import json
import subprocess
import traceback
from pathlib import Path

from . import communication250
from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
COUNTS = dict(full_forwards=352, H_suffix_blocks=608, H_finite_heads=32, extra_QK_softmax_PV_each=5472, witness_extra_QK=1056)
CORRECTION_SOURCES = [ROOT/p for p in ("research/boolean_coupling/communication250b.py", "research/boolean_coupling/audit250b.py", "docs/experiments/250b-witness-completion-correction.md")]


def read(path):
    return json.loads(path.read_text())


def complete(previous, out):
    assert not (out/"status.json").exists() and not (out/"seal.json").exists() and not (out/"completion250b.json").exists()
    assert read(out/"failure.json") == dict(error="AssertionError()")
    progress, design = read(out/"progress.json"), read(out/"design.json")
    assert progress == dict(cases=32, counts=COUNTS)
    assert design["experiment"] == 250
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(out/"sources"/Path(p).name) == h
    for p, h in design["inputs"].items():
        assert sha(previous/p) == h
    cases = read(out/"sources/cases242.json")
    assert design["witness_cases"] == [c["key"] for c in cases]
    entries = []
    for case in cases:
        key = case["key"]
        path = out/"cases"/(key+".json")
        meta = read(path)
        assert meta["key"] == key and meta["group"] == case["group"]
        assert sha(path.with_suffix(".npz")) == meta["arrays_sha256"]
        assert sha(out/"witnesses"/(key+".npz")) == meta["witness_sha256"]
        assert meta["prior_forecast_sha256"] == sha(previous/"forecasts"/(key+".npz"))
        assert meta["prior_native_sha256"] == sha(previous/"observations"/(key+".npz"))
        assert all(c["error"] == c["error"] and c["error"] <= c["bound"] for c in meta["checks"])
        entries.append(dict(key=key, arrays_sha256=meta["arrays_sha256"], record_sha256=sha(path), witness_sha256=meta["witness_sha256"]))
    for p in CORRECTION_SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"completion250b.json", dict(experiment="250b", correction="complete-data witness coverage/count only", model_calls=0,
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in CORRECTION_SOURCES},
        original_design_sha256=sha(out/"design.json"), original_failure_sha256=sha(out/"failure.json"), original_progress_sha256=sha(out/"progress.json"),
        witness_contexts=32, witness_kernels=1056, unchanged_primary=True, unchanged_tolerances=True))
    write_json(out/"status.json", dict(status="complete", cases=entries, counts=COUNTS, qualified_completion="250b", completion_sha256=sha(out/"completion250b.json")))
    write_json(out/"seal.json", dict(status="sealed", cases=entries, design_sha256=sha(out/"design.json"), status_sha256=sha(out/"status.json"), completion_sha256=sha(out/"completion250b.json")))
    print("250b: all32 completed cases retained; count-only correction sealed; no model calls in recovery.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    previous, out = args.input_242.resolve(), args.out.resolve()
    if not args.recover:
        assert not out.exists()
        try:
            communication250.main(previous, out)
        except AssertionError as exc:
            if not (out/"progress.json").exists() or read(out/"progress.json") != dict(cases=32, counts=COUNTS):
                raise
            (out/"original-failure-traceback.txt").write_text(traceback.format_exc())
            write_json(out/"failure.json", dict(error=repr(exc)))
        else:
            raise AssertionError("The frozen original producer was expected to stop on its preserved count assertion")
    complete(previous, out)
