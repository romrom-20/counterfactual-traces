"""Safe reproduction plan for qualified261b/d,262,263,264b; never run invalid261."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stages(inputs, out):
    run = out/"mlp-input-origin-261b"
    replay = out/"mlp-input-replay-261c"
    strict = out/"mlp-input-qualified-261c"
    bounded = out/"mlp-input-bounded-261d"
    factors = out/"gate-curvature-factors-262"
    branch = out/"normalization-branches-energy-263"
    return (
        ("input261b", ["--original", inputs, "--out", run], None, None),
        ("audit261b", ["--run", run], run/"failure-audit.json", "public_up_projection"),
        ("replay261c", ["--run", run, "--out", replay], None, None),
        ("audit261c", ["--run", run, "--replay", replay, "--out", strict], strict/"failure-audit.json", "gate_up_public_displacement_bridge"),
        ("audit261d", ["--run", run, "--replay", replay, "--out", bounded], None, None),
        ("factors262", ["--run", run, "--qualified", bounded, "--out", factors], None, None),
        ("input263", ["--original", inputs, "--reference", run, "--factors", factors, "--out", branch], None, None),
        ("audit263", ["--run", branch], None, None),
        ("interaction264b", ["--run", branch, "--out", out/"normalization-component-interaction-264b"], None, None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        subprocess.run([sys.executable, "-S", str(ROOT/"research/boolean_coupling/verify_normalization.py")], check=True, cwd=ROOT)
        return
    if args.inputs is None or args.out is None:
        parser.error("--inputs (regenerated258 directory) and --out (NEW parent) are required")
    inputs, out = args.inputs.resolve(), args.out.resolve()
    if out.exists() or out == inputs or out.is_relative_to(inputs) or inputs.is_relative_to(out):
        parser.error("output parent must be new and disjoint from input evidence")
    pipeline = stages(inputs, out)
    for name, argv, failure, reason in pipeline:
        command = [sys.executable, "-m", "boolean_coupling."+name, *map(str, argv)]
        print(shlex.join(command), flush=True)
        if failure:
            print("  expected preserved numerical failure:", reason, flush=True)
    if not args.execute:
        print("Plan only; no file writes or model calls. Includes72 forwards plus audits, not invalid attempts.")
        return
    if not (inputs/"audit.json").is_file():
        parser.error("regenerate258's qualified final inputs before executing")
    out.mkdir(parents=True)
    (out/"logs").mkdir()
    env = {**os.environ, "PYTHONPATH":str(ROOT/"research"), "OMP_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1"}
    for name, argv, failure, reason in pipeline:
        command = [sys.executable, "-m", "boolean_coupling."+name, *map(str, argv)]
        with (out/"logs"/(name+".log")).open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if failure:
            assert result.returncode != 0 and failure.is_file(), (name, result.returncode)
            error = json.loads(failure.read_text())["error"]
            assert reason in error and "0.0001" in error, (name, error)
            print(name, "preserved expected failure", flush=True)
        else:
            result.check_returncode()
            print(name, "complete", flush=True)
    print("Qualified pipeline complete. All failures remain failures; contexts are old and Question B is open.")


if __name__ == "__main__":
    main()
