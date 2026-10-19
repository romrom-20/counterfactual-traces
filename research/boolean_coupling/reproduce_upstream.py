"""Verify238 evidence or reproduce the upstream remainder/generator diagnostic."""

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


def bounded(checks):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in checks)


def inventory(folder):
    return {a["path"]:a["sha256"] for a in read(folder/"local-array-inventory.json")["arrays"]}


def verify(history=False):
    folder = ROOT/"results/upstream-remainder-238"
    for name, h in read(folder/"manifest.json")["files"].items():
        assert sha(ROOT/name) == h, name
    design = read(folder/"design.json")
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(folder/"sources"/Path(name).name) == h
    old = ROOT/"results/false-only-alignment-228"
    previous = inventory(old)
    for name, h in design["input_hashes"].items():
        assert (previous[name] if name in previous else sha(old/name)) == h, name
    arrays = inventory(folder)
    seal = read(folder/"selection-seal.json")
    assert seal["before_intervention_outcomes"] and seal["intervention_cases_before_seal"] == 0
    for name, h in seal["files"].items():
        assert (arrays[name] if name in arrays else sha(folder/name)) == h, name
    for phase in ("ledger", "intervene"):
        status = read(folder/("ledger-status.json" if phase == "ledger" else "status.json"))
        audit = read(folder/("ledger-audit.json" if phase == "ledger" else "audit.json"))
        assert status["status"] == "complete" and audit["status"] in ("passed", "passed_declared_domain_with_raw_bridge_failures")
        assert status["counts"] == audit["counts"] and len(status["cases"]) == 32
        assert audit["independent_derivative_FDs"] == status["counts"]["jvps"]
        assert status["selection_sha256"] == sha(folder/"selection.json")
        bounded(audit["checks"])
        fd = read(folder/("numpy-fd-"+phase+".json"))
        assert fd["status"] == "passed" and fd["independent_FD_comparisons"] == status["counts"]["jvps"]
        assert fd["independent_finite_endpoints"] == 2*fd["independent_FD_comparisons"]
        assert sha(ROOT/"research/boolean_coupling/auditfd238.py") == fd["source_sha256"]
        bounded(fd["checks"])
        sub = "ledger" if phase == "ledger" else "cases"
        for e in status["cases"]:
            path = folder/sub/(e["key"]+".json")
            record = read(path)
            assert sha(path) == e["record_sha256"]
            assert record["arrays_sha256"] == e["arrays_sha256"] == arrays[sub+"/"+e["key"]+".npz"]
            bounded(record["checks"])
            bounded(record["fd_witnesses"])
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_late.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified238 source/input seals, native-remainder ledger, pre-outcome selection, all interventions and numerical records; "
        "zero model calls. Integrity verification is not numerical replication.", flush=True)


def plan(out, previous=None, offline=False):
    steps = []
    if previous is None:
        steps.append(dict(name="regenerate-228-through-237", command=[sys.executable,
            "research/boolean_coupling/reproduce_late.py", "--run", "--out", str(out/"prerequisites"),
            *(["--offline"] if offline else [])]))
        previous = out/"prerequisites/experiment237b"
    run = out/"experiment238"
    steps += [dict(name="238-ledger", command=[sys.executable, "-m", "boolean_coupling.transport238c", "--phase", "ledger",
        "--input-237", str(previous), "--out", str(run)]),
        dict(name="238-ledger-audit", command=[sys.executable, "-m", "boolean_coupling.audit238", "--phase", "ledger", "--run", str(run)]),
        dict(name="238-ledger-numpy-fd", command=[sys.executable, "-m", "boolean_coupling.auditfd238", "--phase", "ledger", "--run", str(run)]),
        dict(name="238-selection-seal", seal=True),
        dict(name="238-intervene", command=[sys.executable, "-m", "boolean_coupling.transport238c", "--phase", "intervene", "--out", str(run)]),
        dict(name="238-intervention-audit", command=[sys.executable, "-m", "boolean_coupling.audit238", "--phase", "intervene", "--run", str(run)]),
        dict(name="238-intervention-numpy-fd", command=[sys.executable, "-m", "boolean_coupling.auditfd238", "--phase", "intervene", "--run", str(run)])]
    return steps


def seal_selection(run):
    from datetime import datetime, timezone
    assert not any((run/"cases").iterdir()) and not (run/"selection-seal.json").exists()
    assert read(run/"ledger-audit.json")["status"] == read(run/"numpy-fd-ledger.json")["status"] == "passed"
    paths = [run/name for name in ("design.json", "runtime.json", "selection.json", "ledger-status.json", "ledger-audit.json", "numpy-fd-ledger.json")]
    paths += [p for p in (run/"ledger").iterdir() if p.is_file()]
    record = dict(retrospective=True, created_utc=datetime.now(timezone.utc).isoformat(), before_intervention_outcomes=True,
        intervention_cases_before_seal=0, files={str(p.relative_to(run)):sha(p) for p in paths})
    (run/"selection-seal.json").write_text(json.dumps(record, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-237", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/upstream-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    steps = plan(out, args.input_237.resolve() if args.input_237 else None, args.offline)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Retrospective selected-generator test; source-only computation allowed, native receiving remainders oracle."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; preserve existing evidence.")
    verify()
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", completed=[], steps=steps, source_sha256=sha(__file__))
    for step in steps:
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
        try:
            if step.get("seal"):
                seal_selection(out/"experiment238")
            else:
                with (out/(step["name"]+".log")).open("w") as file:
                    subprocess.run(step["command"], cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_step=step["name"], failure=repr(exc))
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise
        record["completed"].append(step["name"])
    record["status"] = "complete"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    print("Completed fixed-input upstream remainder diagnostic:", out)


if __name__ == "__main__":
    main()
