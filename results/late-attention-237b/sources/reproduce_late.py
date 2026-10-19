"""Verify preserved237/237b evidence or reproduce the late-attention omissions."""

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


def inventory(folder):
    return {a["path"]:a["sha256"] for a in read(folder/"local-array-inventory.json")["arrays"]}


def bounded(checks):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in checks)


def verify(history=False):
    failed, folder = (ROOT/"results"/name for name in ("late-attention-237-failed", "late-attention-237b"))
    for release in (failed, folder):
        for name, h in read(release/"manifest.json")["files"].items():
            assert sha(ROOT/name) == h, name
        for name, h in read(release/"design.json")["source_hashes"].items():
            assert sha(ROOT/name) == sha(release/"sources"/Path(name).name) == h
    #237 failed before writing a complete case. The missing outcome is explicit.
    assert not (failed/"status.json").exists() and not (failed/"audit.json").exists()
    assert "0.0001678466796875" in read(failed/"failure.json")["error"]
    assert "0.0001" in read(failed/"failure.json")["error"]
    design, status, audit = (read(folder/n) for n in ("design.json", "status.json", "complete-numerical-diagnostic.json"))
    assert status["status"] == "complete" and audit["status"] == "complete_with_failed_checks" and audit["retrospective"]
    assert not (folder/"audit.json").exists() and (folder/"failure-audit.json").exists()
    assert audit["failed_checks"] == [c for c in audit["checks"] if not c["passed"]]
    for c in audit["checks"]:
        assert math.isfinite(c["error"]) and c["passed"] == (c["error"] <= c["bound"])
    assert sha(ROOT/"research/boolean_coupling/audit237c.py") == audit["diagnostic_source_sha256"]
    replay = read(folder/"public-mlp-replay.json")
    assert replay["status"] == "passed" and replay["same_backend_replay"] and not replay["raw_numpy_failures_reclassified"]
    assert sha(ROOT/"research/boolean_coupling/replay237c.py") == replay["source_sha256"]
    bounded(replay["checks"])
    assert status["counts"] == audit["counts"] == design["expected"] and status["counts"]["ordinary_true"] == 0
    assert len(status["cases"]) == 32
    assert status["primitive_derivatives"] == audit["primitive_derivatives"]
    assert status["primitive_fd_endpoints"] == 2*audit["primitive_derivatives"]
    arrays = inventory(folder)
    assert status["weights_sha256"] == arrays["weights.npz"]
    prior, old = ROOT/"results/upstream-factorial-236", ROOT/"results/false-only-alignment-228"
    assert sha(prior/"audit.json") == design["audit236_sha256"]
    assert design["weights235_sha256"] == inventory(ROOT/"results/global-error-budget-235")["weights.npz"]
    before = inventory(old)
    for name, h in design["input_hashes"].items():
        assert (before[name] if name in before else sha(old/name)) == h, name
    old236 = inventory(prior)
    for entry in status["cases"]:
        path = folder/"cases"/(entry["key"]+".json")
        record = read(path)
        assert sha(path) == entry["record_sha256"]
        assert record["arrays_sha256"] == entry["arrays_sha256"] == arrays["cases/"+entry["key"]+".npz"]
        assert record["input236_arrays_sha256"] == old236["cases/"+entry["key"]+".npz"]
        bounded(record["checks"])
        for d in record["derivatives"]:
            assert all(math.isfinite(d[k]) for k in ("ad_error", "fd_error", "primal_error"))
            assert d["ad_error"] <= 1e-9 and d["fd_error"] <= 1e-4 and d["primal_error"] <= 1e-9
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_failure.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified failed237 and partially qualified237b, all raw-array inventories/source seals/numerical records; zero model calls. "
          "This verifies retained evidence, not numerical replication.", flush=True)


def plan(out, previous=None, offline=False, include_failed=False):
    steps = []
    if previous is None:
        steps.append(dict(name="regenerate-228-235-236", command=[sys.executable,
            "research/boolean_coupling/reproduce_failure.py", "--run", "--out", str(out/"prerequisites"),
            *(["--offline"] if offline else [])]))
        previous = out/"prerequisites/experiment236"
    if include_failed:
        steps.append(dict(name="237-original-failed-attempt", expect_failure=True, command=[sys.executable,
            "-m", "boolean_coupling.late237", "--input-236", str(previous), "--out", str(out/"experiment237")]))
    steps += [dict(name="237b-omissions", command=[sys.executable, "-m", "boolean_coupling.late237b",
        "--input-236", str(previous), "--out", str(out/"experiment237b")]),
        dict(name="237b-original-audit", expect_audit_failure=True, command=[sys.executable, "-m", "boolean_coupling.audit237b", "--run", str(out/"experiment237b")]),
        dict(name="237c-complete-checks", command=[sys.executable, "-m", "boolean_coupling.audit237c", "--run", str(out/"experiment237b")]),
        dict(name="237c-public-replay", command=[sys.executable, "-m", "boolean_coupling.replay237c", "--run", str(out/"experiment237b")]),
        dict(name="237c-posthoc-description", command=[sys.executable, "-m", "boolean_coupling.describe_late", "--run", str(out/"experiment237b")])]
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--include-failed-237", action="store_true")
    parser.add_argument("--input-236", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/late-attention-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    steps = plan(out, args.input_236.resolve() if args.input_236 else None, args.offline, args.include_failed_237)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Retrospective omission test using finite public attention and oracle source rows; not a False-only predictor."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; preserve existing evidence.")
    verify()
    from huggingface_hub import snapshot_download
    spec = read(ROOT/"research/boolean_coupling/settings.json")["model"]
    snapshot_download(repo_id=spec["model"], revision=spec["revision"], local_files_only=args.offline)
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", steps=steps, completed=[], wrapper_sha256=sha(__file__))
    for step in steps:
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
        with (out/(step["name"]+".log")).open("w") as log:
            result = subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if step.get("expect_failure"):
            # Preserve this as failure, not a successful science run. A different
            # failure or a surprising pass must stop for investigation.
            error = read(out/"experiment237/failure.json")["error"] if (out/"experiment237/failure.json").exists() else ""
            valid = result.returncode != 0 and "full_graph_public_final" in error and "0.0001" in error
            record["original237"] = dict(returncode=result.returncode, preserved_failure=error)
        elif step.get("expect_audit_failure"):
            file = out/"experiment237b/failure-audit.json"
            error = read(file)["error"] if file.exists() else ""
            valid = result.returncode != 0 and "numpy_finite_mlp" in error and "0.0001" in error
            record["original237b_audit"] = dict(returncode=result.returncode, preserved_failure=error)
        else:
            valid = result.returncode == 0
        if not valid:
            record.update(status="failed", failed_step=step["name"], returncode=result.returncode)
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise RuntimeError(f"Stopped at {step['name']}; original logs/failure retained.")
        record["completed"].append(step["name"])
    record["status"] = "complete_with_preserved_numerical_failures"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    print("Completed fixed-input late-attention diagnostic with original failures retained:", out)


if __name__ == "__main__":
    main()
