"""Verify communication250 evidence without models, or run its finite factorial."""

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


def verify(history):
    run = ROOT/"results/native-source-communication-250b"
    design, status, seal, audit, manifest = (read(run/n) for n in ("design.json", "status.json", "seal.json", "audit.json", "manifest.json"))
    for p, h in manifest["files"].items():
        assert sha(ROOT/p) == h, p
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(run/"sources"/Path(p).name) == h
    assert design["experiment"] == 250 and design["retrospective"] and audit["status"] == "passed"
    assert status["status"] == "complete" and seal["status"] == "sealed" and len(status["cases"]) == 32
    assert status["cases"] == seal["cases"] and seal["status_sha256"] == sha(run/"status.json")
    assert seal["design_sha256"] == audit["design_sha256"] == sha(run/"design.json")
    assert audit["seal_sha256"] == sha(run/"seal.json") and audit["scorer_sha256"] == sha(ROOT/"research/boolean_coupling/audit250b.py")
    completion = read(run/"completion250b.json")
    assert completion["original_design_sha256"] == sha(run/"design.json")
    assert completion["original_failure_sha256"] == sha(run/"failure.json")
    assert completion["original_progress_sha256"] == sha(run/"progress.json")
    assert audit["completion_sha256"] == status["completion_sha256"] == seal["completion_sha256"] == sha(run/"completion250b.json")
    for p, h in completion["source_hashes"].items():
        assert sha(ROOT/p) == sha(run/"sources"/Path(p).name) == h
    assert audit["witness_coverage"] == dict(contexts=32, blocks=[10, 19, 28], arms=11, kernels=1056, full_forward_reconstruction=False)
    previous = ROOT/"results/uniform-primitive-242b"
    old_inventory = {e["path"]:e["sha256"] for e in read(previous/"local-array-inventory.json")["arrays"]}
    for p, h in design["inputs"].items():
        assert (old_inventory[p] if p == "weights.npz" else sha(previous/p)) == h
    inventory = {e["path"]:e["sha256"] for e in read(run/"local-array-inventory.json")["arrays"]}
    for c in audit["checks"]:
        assert math.isfinite(c["error"]) and c["error"] <= c["bound"]
    for e in status["cases"]:
        p = run/"cases"/(e["key"]+".json")
        meta = read(p)
        assert sha(p) == e["record_sha256"] and meta["arrays_sha256"] == e["arrays_sha256"] == inventory["cases/"+e["key"]+".npz"]
        for c in meta["checks"]:
            assert math.isfinite(c["error"]) and c["error"] <= c["bound"]
        assert meta["witness_sha256"] == e["witness_sha256"]
        if e["witness_sha256"]:
            assert e["key"] in design["witness_cases"] and inventory["witnesses/"+e["key"]+".npz"] == e["witness_sha256"]
    for name, h in audit["csv_hashes"].items():
        assert sha(run/(name+".csv")) == h
    assert status["counts"] == dict(full_forwards=352, H_suffix_blocks=608, H_finite_heads=32, extra_QK_softmax_PV_each=5472, witness_extra_QK=1056)
    if history:
        subprocess.run([sys.executable, str(ROOT/"research/boolean_coupling/reproduce_native.py"), "--verify-only", "--history"], check=True, cwd=ROOT)
    print("Verified qualified250b complete outcomes, finite witness coverage and seals; integrity is not model replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only", "history", "plan", "run", "offline"):
        parser.add_argument("--"+name, action="store_true")
    for name in ("input-242", "out"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    out = (args.out or ROOT/"runs/communication-reproduction").resolve()
    previous = args.input_242.resolve() if args.input_242 else out/"prerequisite242b"
    result = out/"experiment250"
    steps = []
    if not args.input_242:
        steps.append(dict(name="through242b", command=[sys.executable, "research/boolean_coupling/reproduce_uniform.py", "--run", "--out", str(previous), *(["--offline"] if args.offline else [])]))
    for name, module in (("communication250", "communication250b"), ("audit250", "audit250b")):
        steps.append(dict(name=name, command=[sys.executable, "-m", "boolean_coupling."+module, "--input-242", str(previous), "--out" if name == "communication250" else "--run", str(result)]))
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Retrospective native source communication clamps; no False-only prediction or fresh confirmation."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence")
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", steps=steps, completed=[], wrapper_sha256=sha(__file__))
    for step in steps:
        try:
            with (out/(step["name"]+".log")).open("x") as log:
                subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_step=step["name"], error=repr(exc))
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise
        record["completed"].append(step["name"])
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    record["status"] = "complete"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
