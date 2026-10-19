"""Verify241's full-vector diagnostic or regenerate its fixed-array pipeline."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded(records):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in records)


def verify(history=False):
    folder = ROOT / "results/before-mlp16-vectors-241"
    design, status, audit, selection, manifest = (read(folder / n) for n in
        ("design.json", "status.json", "audit.json", "selection.json", "manifest.json"))
    for name, h in manifest["files"].items():
        assert sha(ROOT / name) == h, name
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(folder / "sources" / Path(name).name) == h
    assert status["status"] == "complete" and audit["status"] == "passed" and len(status["cases"]) == 32
    assert status["counts"] == dict(JVPs=13920, FD_endpoints=27840) and status["ordinary_model_calls"] == 0
    assert audit["independent_counts"] == dict(directions=13920, finite_endpoints=27840)
    assert audit["selection_sha256"] == sha(folder / "selection.json")
    assert set(selection["order"]) == set(range(2, 31)) and selection["primary_sites"] == selection["order"][:4]
    assert audit["primary"] == selection["decisions"]["4"]
    bounded(audit["checks"])
    inv = {r["path"]: r["sha256"] for r in read(folder / "local-array-inventory.json")["arrays"]}
    old = {}
    for tag, release in (("228", "false-only-alignment-228"), ("238", "upstream-remainder-238"), ("240", "target-blind-silu-240")):
        old[tag] = {r["path"]: r["sha256"] for r in read(ROOT / "results" / release / "local-array-inventory.json")["arrays"]}
    for tag, release in (("238", "upstream-remainder-238"), ("240", "target-blind-silu-240")):
        assert sha(ROOT / "results" / release / "audit.json") == design["input_audits"][tag]
    for ent in status["cases"]:
        key = ent["key"]
        path = folder / "cases" / (key + ".json")
        meta = read(path)
        assert sha(path) == ent["record_sha256"]
        assert meta["arrays_sha256"] == ent["arrays_sha256"] == inv["cases/" + key + ".npz"]
        bounded(meta["checks"])
        assert len(meta["witnesses"]) == 29 and sum(w["directions"] for w in meta["witnesses"]) == 435
        assert all(math.isfinite(w["max_fd_error"]) and w["max_fd_error"] <= 1e-4 for w in meta["witnesses"])
        for label, tag, phase in (("ledger", "238", "ledger"), ("forecasts", "228", "forecasts"),
            ("observations", "228", "observations"), ("mask", "240", "masks")):
            assert meta["inputs"][label] == old[tag][phase + "/" + key + ".npz"]
    for name, source in (("profile-summary.json", "describe241.py"), ("metric-description.json", "metric241.py"),
        ("transport-metric.json", "decompose241.py")):
        desc = read(folder / name)
        assert desc["source_sha256"] == sha(ROOT / "research/boolean_coupling" / source)
        assert desc["audit_sha256"] == sha(folder / "audit.json")
        bounded(desc["checks"])
    command = [sys.executable, str(ROOT / "research/boolean_coupling/reproduce_silu.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified241 complete vector transport, derivative witnesses, all fixed inputs and every concentration outcome; zero model calls. "
        "Integrity checking is not numerical replication or causal attribution.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-238", type=Path)
    parser.add_argument("--input-240", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/early-reproduction")
    args = parser.parse_args()
    if bool(args.input_238) != bool(args.input_240):
        parser.error("Supply both --input-238 and --input-240, or neither.")
    out = args.out.resolve()
    previous = args.input_238.resolve() if args.input_238 else out / "prerequisites/prerequisites/prerequisites/experiment238"
    last = args.input_240.resolve() if args.input_240 else out / "prerequisites/experiment240"
    steps = []
    if not args.input_238:
        steps.append(dict(name="through240", command=[sys.executable, "research/boolean_coupling/reproduce_silu.py", "--run",
            "--out", str(out / "prerequisites"), *(["--offline"] if args.offline else [])]))
    run = out / "experiment241"
    steps += [dict(name="241-vectors", command=[sys.executable, "-m", "boolean_coupling.vector241", "--input-238", str(previous),
        "--input-240", str(last), "--out", str(run)]),
        dict(name="241-audit", command=[sys.executable, "-m", "boolean_coupling.audit241", "--run", str(run)]),
        dict(name="241-profile", command=[sys.executable, "-m", "boolean_coupling.describe241", "--run", str(run)]),
        dict(name="241-Gram-description", command=[sys.executable, "-m", "boolean_coupling.metric241", "--run", str(run)]),
        dict(name="241-transport-description", command=[sys.executable, "-m", "boolean_coupling.decompose241", "--run", str(run)])]
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Retrospective fixed-array remainder transport. Strict archived input hashes; no fresh predictor or primitive causal claim."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    verify()
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", completed=[], steps=steps, wrapper_sha256=sha(__file__))
    for step in steps:
        (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
        try:
            with (out / (step["name"] + ".log")).open("w") as log:
                subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_step=step["name"], failure=repr(exc))
            (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
            raise
        record["completed"].append(step["name"])
    record["status"] = "complete"
    (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Completed fixed-array upstream vector diagnostic:", out)


if __name__ == "__main__":
    main()
