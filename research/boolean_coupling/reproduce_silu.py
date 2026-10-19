"""Verify240's sealed primitive test or regenerate it with fixed prerequisites."""

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


def bounded(rows):
    assert all(math.isfinite(r["error"]) and r["error"] <= r["bound"] for r in rows)


def verify(history=False):
    folder = ROOT / "results/target-blind-silu-240"
    manifest, design, seal, status, audit = (read(folder / n) for n in
        ("manifest.json", "design.json", "mask-seal.json", "status.json", "audit.json"))
    for name, h in manifest["files"].items():
        assert sha(ROOT / name) == h, name
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(folder / "sources" / Path(name).name) == h
    assert seal["status"] == "sealed" and not seal["native_inputs_opened"] and not seal["intervention_outcomes_measured"]
    assert seal["counts"] == dict(ordinary_false=32, source_blocks=32, mask_mlps=32)
    assert status["status"] == "complete" and len(status["cases"]) == len(seal["cases"]) == 32
    assert status["counts"] == dict(intervention_mlps=384, suffix_blocks=4608, heads=384)
    assert status["ordinary_true_calls"] == 0 and status["mask_seal_sha256"] == sha(folder / "mask-seal.json")
    assert audit["status"] in ("passed", "passed_with_retained_Float64_bridge_failures")
    bounded(audit["checks"])
    arrays = {r["path"]: r["sha256"] for r in read(folder / "local-array-inventory.json")["arrays"]}
    prior = {}
    for key, release in (("228", "false-only-alignment-228"), ("238", "upstream-remainder-238")):
        prior[key] = {r["path"]: r["sha256"] for r in read(ROOT / "results" / release / "local-array-inventory.json")["arrays"]}
    assert arrays["weights.npz"] == seal["weights_sha256"]
    for ent, pent in zip(status["cases"], seal["cases"], strict=True):
        assert ent["key"] == pent["key"]
        name = ent["key"] + ".json"
        cm, pm = read(folder / "cases" / name), read(folder / "masks" / name)
        assert sha(folder / "cases" / name) == ent["record_sha256"]
        assert sha(folder / "masks" / name) == pent["record_sha256"]
        assert cm["arrays_sha256"] == ent["arrays_sha256"] == arrays["cases/" + ent["key"] + ".npz"]
        assert pm["arrays_sha256"] == pent["arrays_sha256"] == cm["mask_arrays_sha256"] == arrays["masks/" + ent["key"] + ".npz"]
        assert pm["forecast228_sha256"] == prior["228"]["forecasts/" + ent["key"] + ".npz"]
        assert cm["observation228_sha256"] == prior["228"]["observations/" + ent["key"] + ".npz"]
        assert pm["upstream238_sha256"] == prior["238"]["cases/" + ent["key"] + ".npz"]
        bounded(cm["checks"] + pm["checks"])
    assert sha(ROOT / "results/generator-curvature-239/audit.json") == design["audit239_sha256"]
    assert sha(ROOT / "results/upstream-remainder-238/audit.json") == design["audit238_sha256"]
    for row in audit["raw_float64_bridges"]:
        assert math.isfinite(row["error"]) and row["passed"] == (row["error"] <= row["bound"])
    recheck = read(folder / "input-recheck.json")
    assert recheck["status"] == "passed" and recheck["model_calls"] == 0 and len(recheck["checks"]) == 128
    assert sha(ROOT / "research/boolean_coupling/provenance240.py") == recheck["source_sha256"]
    for row in recheck["checks"]:
        expected = {r["path"]: r["sha256"] for r in read(ROOT / "results" / row["release"] / "local-array-inventory.json")["arrays"]}
        assert row["passed"] and row["sha256"] == row["expected"] == expected[row["path"]]
    command = [sys.executable, str(ROOT / "research/boolean_coupling/reproduce_curvature.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified240 source/mask seals, all cases, primitive audit records and finite final outcomes; zero model calls. "
        "Integrity verification is not numerical replication or a full-response prediction.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-239", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/silu-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    previous = args.input_239.resolve() if args.input_239 else out / "prerequisites/experiment239"
    steps = []
    if not args.input_239:
        steps.append(dict(name="through239", command=[sys.executable, "research/boolean_coupling/reproduce_curvature.py",
            "--run", "--out", str(out / "prerequisites"), *(["--offline"] if args.offline else [])]))
    run = out / "experiment240"
    steps += [dict(name="240-seal-masks", command=[sys.executable, "-m", "boolean_coupling.run240", "--phase", "prepare",
        "--input-239", str(previous), "--out", str(run)]),
        dict(name="240-intervene", command=[sys.executable, "-m", "boolean_coupling.run240", "--phase", "intervene", "--out", str(run)]),
        dict(name="240-audit", command=[sys.executable, "-m", "boolean_coupling.audit240", "--run", str(run)])]
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Old-context, target-blind mask selection and finite primitive intervention; supplied native input is oracle."), indent=2))
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
    print("Completed sealed-mask primitive diagnostic:", out)


if __name__ == "__main__":
    main()
