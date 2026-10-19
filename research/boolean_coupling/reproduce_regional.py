"""Verify243's evidence without model imports or rerun its regional factorial."""

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


def bounded(rows):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in rows)


def verify(history=False):
    folder = ROOT / "results/regional-silu-243"
    manifest, design, status, seal, audit = (read(folder / n) for n in
        ("manifest.json", "design.json", "status.json", "intervention-seal.json", "audit.json"))
    for name, h in manifest["files"].items():
        assert sha(ROOT / name) == h, name
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(folder / "sources" / Path(name).name) == h
    assert design["experiment"] == 243 and design["retrospective"]
    assert status["status"] == "complete" and seal["status"] == "sealed" and audit["status"] == "passed"
    assert audit["scorer_sha256"] == sha(ROOT / "research/boolean_coupling/audit243b.py")
    assert audit["original_scorer_sha256"] == sha(ROOT / "research/boolean_coupling/audit243.py")
    assert "unstable_forwards" in read(folder / "failure-audit.json")["error"]
    assert status["ordinary_true_calls"] == seal["ordinary_true_calls"] == 0
    assert not seal["native_arrays_opened_by_runner"] and seal["retrospective"]
    assert seal["design_sha256"] == audit["design_sha256"] == sha(folder / "design.json")
    assert seal["status_sha256"] == sha(folder / "status.json") and audit["seal_sha256"] == sha(folder / "intervention-seal.json")
    assert seal["cases"] == status["cases"] and len(status["cases"]) == 32
    assert status["counts"]["zero_edit_controls"] == 32 and status["counts"]["intervention_forwards"] == 128
    assert status["counts"]["complete_forwards"] + status["counts"].get("unstable_forwards", 0) == 160
    bounded(audit["checks"])
    for name, h in audit["csv_hashes"].items():
        assert sha(folder / (name + ".csv")) == h
    previous = ROOT / "results/uniform-primitive-242b"
    for tag, name in (("audit", "audit.json"), ("design", "design.json"), ("forecast_status", "forecast-status.json"), ("native_status", "status.json")):
        assert design["input_" + tag + "_sha256"] == sha(previous / name)
    old = {c["key"]: c for c in read(previous / "forecast-status.json")["cases"]}
    old_inventory = {r["path"]: r["sha256"] for r in read(previous / "local-array-inventory.json")["arrays"]}
    assert old_inventory["weights.npz"] == design["input_weights_sha256"]
    inventory = {r["path"]: r["sha256"] for r in read(folder / "local-array-inventory.json")["arrays"]}
    for ent in status["cases"]:
        key = ent["key"]
        p = folder / "cases" / (key + ".json")
        meta = read(p)
        assert sha(p) == ent["record_sha256"]
        assert meta["arrays_sha256"] == ent["arrays_sha256"] == inventory["cases/" + key + ".npz"]
        assert meta["prior_forecast_sha256"] == old[key]["arrays_sha256"]
        assert meta["prior_record_sha256"] == old[key]["record_sha256"]
        bounded(meta["checks"])
        for arm in meta["arms"].values():
            bounded(arm["reference_checks"])
            assert len(arm["witnesses"]) == arm["qualification_counts"]["distinct_directions"]
    desc = read(folder / "description.json")
    assert desc["posthoc"] and desc["model_calls"] == 0
    assert desc["source_sha256"] == sha(ROOT / "research/boolean_coupling/describe243.py")
    assert desc["audit_sha256"] == sha(folder / "audit.json")
    if history:
        subprocess.run([sys.executable, str(ROOT / "research/boolean_coupling/reproduce_uniform.py"),
            "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified243 source/input/output seals, every regional intervention and numerical record; no model imports. "
        "Evidence integrity is not numerical replication or a prospective prediction.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-242", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/regional-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    previous = args.input_242.resolve() if args.input_242 else out / "prerequisite242b"
    run, steps = out / "experiment243", []
    if not args.input_242:
        steps.append(dict(name="through242b", command=[sys.executable, "research/boolean_coupling/reproduce_uniform.py", "--run",
            "--out", str(previous), *(["--offline"] if args.offline else [])]))
    steps += [dict(name="intervene", command=[sys.executable, "-m", "boolean_coupling.regional243", "--input-242", str(previous), "--out", str(run)]),
        dict(name="audit", command=[sys.executable, "-m", "boolean_coupling.audit243b", "--input-242", str(previous), "--run", str(run)])]
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Retrospective regional SiLU factorial; no learned-circuit or fresh-confirmation claim."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", completed=[], steps=steps, wrapper_sha256=sha(__file__))
    for step in steps:
        try:
            with (out / (step["name"] + ".log")).open("x") as log:
                subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_step=step["name"], error=repr(exc))
            (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
            raise
        record["completed"].append(step["name"])
        (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
    record["status"] = "complete"
    (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Completed regional diagnostic:", run)


if __name__ == "__main__":
    main()
