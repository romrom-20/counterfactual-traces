"""Verify247–249 without model imports, or reproduce native context/carry/source tests."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded(checks):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in checks)


def verify(history):
    previous = ROOT/"results/uniform-primitive-242b"
    old_weights = next(r["sha256"] for r in read(previous/"local-array-inventory.json")["arrays"] if r["path"] == "weights.npz")
    first = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    native = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    earlier = {}
    for num, name in ((247, "native-verdict-context-247"), (248, "native-carry-read-248"), (249, "source-regeneration-249")):
        folder = ROOT/"results"/name
        manifest, design, status, seal, audit = (read(folder/n) for n in ("manifest.json", "design.json", "status.json", "seal.json", "audit.json"))
        for path, h in manifest["files"].items():
            assert sha(ROOT/path) == h, path
        for path, h in design["source_hashes"].items():
            assert sha(ROOT/path) == sha(folder/"sources"/Path(path).name) == h
        assert design["experiment"] == num and design["retrospective"] and audit["status"] == "passed"
        assert status["status"] == "complete" and seal["status"] == "sealed" and len(status["cases"]) == 32 and seal["cases"] == status["cases"]
        assert seal["status_sha256"] == sha(folder/"status.json") and audit["seal_sha256"] == sha(folder/"seal.json")
        assert audit["design_sha256"] == sha(folder/"design.json") and audit["scorer_sha256"] == sha(ROOT/f"research/boolean_coupling/audit{num}.py")
        bounded(audit["checks"])
        for csv, h in audit["csv_hashes"].items():
            assert sha(folder/(csv+".csv")) == h
        refs = design["inputs"] if num == 247 else design["input242"]
        for filename, h in refs.items():
            assert (old_weights if filename == "weights.npz" else sha(previous/filename)) == h
        inventory = {r["path"]:r["sha256"] for r in read(folder/"local-array-inventory.json")["arrays"]}
        if num == 249:
            fs, fseal, cd = (read(folder/n) for n in ("forecast-status.json", "forecast-seal.json", "control-design.json"))
            assert not fs["native_True_arrays_opened"] and not fseal["native_True_arrays_opened"]
            assert fseal["status_sha256"] == sha(folder/"forecast-status.json") and fseal["design_sha256"] == sha(folder/"design.json")
            assert fs["cases"] == fseal["cases"] and len(fs["cases"]) == 32
            assert cd["forecast_seal_sha256"] == seal["forecast_seal_sha256"] == audit["forecast_seal_sha256"] == sha(folder/"forecast-seal.json")
            assert seal["control_design_sha256"] == sha(folder/"control-design.json")
            assert datetime.fromisoformat(cd["started_utc"]) > datetime.fromisoformat(fseal["created_utc"])
            batches = (("forecasts", fs["cases"]), ("controls", status["cases"]))
            for n, label in ((247, "input247"), (248, "input248")):
                for filename, h in cd[label].items():
                    assert sha(earlier[n]/filename) == h
        else:
            assert seal["design_sha256"] == sha(folder/"design.json")
            batches = (("cases", status["cases"]),)
        for sub, entries in batches:
            for ent in entries:
                key = ent["key"]
                meta_path = folder/sub/(key+".json")
                meta = read(meta_path)
                assert sha(meta_path) == ent["record_sha256"]
                assert meta["arrays_sha256"] == ent["arrays_sha256"] == inventory[sub+"/"+key+".npz"]
                bounded(meta["checks"])
                if "prior_forecast_sha256" in meta:
                    assert meta["prior_forecast_sha256"] == first[key]["arrays_sha256"]
                if "prior_native_sha256" in meta:
                    assert meta["prior_native_sha256"] == native[key]["arrays_sha256"]
        if num == 247:
            assert status["counts"]["ordinary_false"] == status["counts"]["ordinary_true"] == 32 and status["counts"]["primitive_intervention_forwards"] == 320
        if num == 248:
            assert status["reused_True_inputs"] and status["ordinary_true"] == 0
            for filename, h in design["input247"].items():
                assert sha(earlier[247]/filename) == h
            assert status["counts"]["suffix_blocks"] == 2432
        if num == 249:
            assert fs["counts"] == dict(ordinary_false=32, source_dense_blocks=288, suffix_blocks=608, finite_heads=32)
            assert status["counts"] == dict(suffix_blocks=608, finite_heads=32)
        earlier[num] = folder
    if history:
        subprocess.run([sys.executable, str(ROOT/"research/boolean_coupling/reproduce_mixing.py"), "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified247–249 source seals, complete case evidence and forecast/control ordering; no model imports. Integrity is not replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only", "history", "plan", "run", "offline"):
        parser.add_argument("--"+name, action="store_true")
    for name in ("input-242", "input-247", "input-248", "out"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    if args.input_247 and not args.input_242:
        parser.error("--input-247 requires its matching --input-242")
    if args.input_248 and not args.input_247:
        parser.error("--input-248 requires matching --input-247 and --input-242")
    out = (args.out or ROOT/"runs/native-response-reproduction").resolve()
    prev = args.input_242.resolve() if args.input_242 else out/"prerequisite242b"
    local = args.input_247.resolve() if args.input_247 else out/"experiment247"
    carried = args.input_248.resolve() if args.input_248 else out/"experiment248"
    source = out/"experiment249"
    steps = []
    def step(name, module, *arguments):
        steps.append(dict(name=name, command=[sys.executable, "-m", "boolean_coupling."+module, *map(str, arguments)]))
    if not args.input_242:
        steps.append(dict(name="through242b", command=[sys.executable, "research/boolean_coupling/reproduce_uniform.py", "--run", "--out", str(prev), *(["--offline"] if args.offline else [])]))
    if not args.input_247:
        step("native247", "verdict247", "--input-242", prev, "--out", local)
        step("audit247", "audit247", "--input-242", prev, "--run", local)
    if not args.input_248:
        step("carry248", "suffix248", "--input-242", prev, "--input-247", local, "--out", carried)
        step("audit248", "audit248", "--input-242", prev, "--input-247", local, "--run", carried)
    step("source249", "source249", "--phase", "forecast", "--input-242", prev, "--out", source)
    step("control249", "source249", "--phase", "controls", "--input-242", prev, "--input-247", local, "--input-248", carried, "--out", source)
    step("audit249", "audit249", "--input-242", prev, "--input-247", local, "--input-248", carried, "--run", source)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Known-context native-input diagnostics;249 forecasts sealed before oracle controls. No fresh confirmation."), indent=2))
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
    record = dict(status="running", completed=[], steps=steps, wrapper_sha256=sha(__file__))
    for entry in steps:
        try:
            with (out/(entry["name"]+".log")).open("x") as log:
                subprocess.run(entry["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            record.update(status="failed", failed_step=entry["name"], error=repr(exc))
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise
        record["completed"].append(entry["name"])
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    record["status"] = "complete"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
