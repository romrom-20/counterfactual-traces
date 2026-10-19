"""Verify the239 curvature diagnostic or regenerate it with its fixed prerequisites."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inventory(folder):
    return {r["path"]:r["sha256"] for r in read(folder/"local-array-inventory.json")["arrays"]}


def bounded(checks):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in checks)


def verify(history=False):
    folder, previous = ROOT/"results/generator-curvature-239", ROOT/"results/upstream-remainder-238"
    manifest, design, status, audit = (read(folder/name) for name in ("manifest.json", "design.json", "status.json", "audit.json"))
    for name, h in manifest["files"].items():
        assert sha(ROOT/name) == h, name
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(folder/"sources"/Path(name).name) == h
    assert sha(previous/"audit.json") == design["audit238_sha256"] and sha(previous/"selection.json") == design["selection_sha256"]
    assert status["status"] == "complete" and status["ordinary_models"] == audit["ordinary_models"] == 0
    assert audit["status"] in ("passed", "passed_declared_domain_with_raw_bridge_failures")
    assert status["hessian_AD_comparisons"] == 384 and status["finite_second_difference_endpoints"] == 768
    assert len(status["cases"]) == 32
    bounded(audit["checks"])
    arrays, old = inventory(folder), inventory(previous)
    for ent in status["cases"]:
        p = folder/"cases"/(ent["key"]+".json")
        meta = read(p)
        assert sha(p) == ent["record_sha256"]
        assert meta["arrays_sha256"] == ent["arrays_sha256"] == arrays["cases/"+ent["key"]+".npz"]
        assert meta["ledger238_sha256"] == old["ledger/"+ent["key"]+".npz"]
        assert meta["intervention238_sha256"] == old["cases/"+ent["key"]+".npz"]
        bounded(meta["checks"])
    tail = read(folder/"tail-description.json")
    assert tail["posthoc"] and tail["new_model_calls"] == 0 and not tail["causal_interventions"]
    assert sha(ROOT/"research/boolean_coupling/describe239.py") == tail["source_sha256"]
    bounded(tail["checks"])
    for record in [*audit["raw_float64_bridges"], *tail["public_bridge_checks"]]:
        assert math.isfinite(record["error"]) and record["passed"] == (record["error"] <= record["bound"])
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_upstream.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified239 curvature formulas, Hessian/FD records, all site/scope outcomes and explicit posthoc identities; zero model calls. "
        "Integrity checking is not numerical replication or a successful general predictor.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-238", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/curvature-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    steps = []
    previous = args.input_238.resolve() if args.input_238 else out/"prerequisites/experiment238"
    if not args.input_238:
        steps.append(dict(name="regenerate-through-238", command=[sys.executable,
            "research/boolean_coupling/reproduce_upstream.py", "--run", "--out", str(out/"prerequisites"),
            *(["--offline"] if args.offline else [])]))
    run = out/"experiment239"
    steps += [dict(name="239-coefficients", command=[sys.executable, "-m", "boolean_coupling.run239",
        "--input-238", str(previous), "--out", str(run)]),
        dict(name="239-audit", command=[sys.executable, "-m", "boolean_coupling.audit239", "--run", str(run)]),
        dict(name="239-posthoc-tail", command=[sys.executable, "-m", "boolean_coupling.describe239", "--run", str(run)])]
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Old-context local contrast under supplied native/forecast inputs; not a new full-response forecast."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    verify()
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", completed=[], steps=steps, wrapper_sha256=sha(__file__))
    for step in steps:
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
        try:
            with (out/(step["name"]+".log")).open("w") as log:
                subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_step=step["name"], failure=repr(exc))
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise
        record["completed"].append(step["name"])
    record["status"] = "complete"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    print("Completed fixed-input curvature diagnostic:", out)


if __name__ == "__main__":
    main()
