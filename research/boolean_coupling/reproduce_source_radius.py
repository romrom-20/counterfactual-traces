"""Verify 257 and prior evidence, or repeat 256 followed by the model-free diagnostic."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(history):
    folder = ROOT/"results/source-receiver-radius-257"
    for name, h in read(folder/"manifest.json")["files"].items():
        assert sha(ROOT/name) == h, name
    audit = read(folder/"audit.json")
    assert audit["status"] == "passed" and audit["retrospective"] and audit["model_calls"] == 0
    assert audit["source_sha256"] == sha(ROOT/"research/boolean_coupling/source_radius257.py")
    assert audit["protocol_sha256"] == sha(ROOT/"docs/experiments/257-source-receiver-radius.md")
    assert audit["input_audit_sha256"] == sha(ROOT/"results/local-query-curvature-256/audit.json")
    assert audit["checkpoint"] == read(ROOT/"results/local-query-curvature-256/design.json")["checkpoint"]
    assert audit["csv_sha256"] == sha(folder/"source-receiver-errors.csv")
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in audit["checks"])
    with (folder/"source-receiver-errors.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1792 and len({r["key"] for r in rows}) == 32
    assert {r["scope"] for r in rows} == {"source", "receivers"}
    first = [r for r in rows if r["scope"] == "source" and r["layer"] == "1"]
    assert len(first) == 32 and len({r["maximum_b"] for r in first}) == 1
    initial = audit["initial_unique_source"]
    assert float(first[0]["maximum_b"]) == initial["maximum_b"]
    assert audit["source_outside_radius_hypothesis"] == (initial["maximum_b"] >= 1)
    for row in rows:
        size = float(row["maximum_b"])
        assert math.isfinite(size) and size >= 0
        if size > 0:
            assert abs(float(row["minimum_series_radius"])-1/math.sqrt(size)) <= 1e-12
        else:
            assert row["minimum_series_radius"] == ""
    if history:
        subprocess.run([sys.executable, str(ROOT/"research/boolean_coupling/reproduce_local_query.py"), "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified257 source/receiver geometry, rejected outside-radius hypothesis and prior evidence. Integrity is not model replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only", "history", "plan", "run", "offline"):
        parser.add_argument("--"+name, action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = (args.out or ROOT/"runs/source-radius-reproduction").resolve()
    commands = [[sys.executable, str(ROOT/"research/boolean_coupling/reproduce_local_query.py"), "--run", "--out", str(out/"forecast256")]]
    if args.offline:
        commands[0].append("--offline")
    commands.append([sys.executable, "-m", "boolean_coupling.source_radius257", "--run", str(out/"forecast256/experiment256"), "--out", str(out/"diagnostic257")])
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(commands=commands, scope="Published-literal replication, then retrospective local algebra; no fresh validation or global theorem."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence")
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    record = dict(status="running", completed=[], wrapper_sha256=sha(Path(__file__)))
    for i, command in enumerate(commands):
        try:
            with (out/f"stage{i+1}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_stage=i+1, error=repr(exc))
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise
        record["completed"].append(i+1)
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    record["status"] = "complete"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
