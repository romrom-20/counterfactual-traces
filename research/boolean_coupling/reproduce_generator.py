"""Verify 229 without dependencies or reproduce it from public inputs/checkpoint."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results/first-attention-generator-229"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(history=False):
    manifest = json.loads((RESULTS/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha(ROOT/name) == digest, name
    design = json.loads((RESULTS/"design.json").read_text())
    for name, digest in design["source_hashes"].items():
        assert sha(ROOT/name) == digest, name
    audit = json.loads((RESULTS/"audit.json").read_text())
    status = json.loads((RESULTS/"status.json").read_text())
    assert audit["status"] == "passed" and design["retrospective"] and audit["retrospective"]
    assert audit["case_count"] == 32 and audit["outcome_count"] == 480
    assert status["counts"] == design["expected"] == audit["counts"]
    assert audit["design_sha256"] == sha(RESULTS/"design.json")
    assert all(c["error"] <= c["bound"] for c in audit["checks"])
    inventory = {r["path"]:r["sha256"] for r in json.loads((RESULTS/"local-array-inventory.json").read_text())["arrays"]}
    assert inventory["weights.npz"] == status["weights_sha256"]
    original = ROOT/"results/false-only-alignment-228"
    old_arrays = {r["path"]:r["sha256"] for r in json.loads((original/"local-array-inventory.json").read_text())["arrays"]}
    for path, digest in design["inputs"].items():
        if path in old_arrays:
            assert old_arrays[path] == digest
        elif (original/path).exists():
            assert sha(original/path) == digest
        else:
            raise AssertionError("Unbound prerequisite metadata: "+path)
    for entry in status["cases"]:
        path = RESULTS/"cases"/(entry["key"]+".json")
        record = json.loads(path.read_text())
        assert sha(path) == entry["record_sha256"]
        assert record["arrays_sha256"] == entry["arrays_sha256"] == inventory["cases/"+entry["key"]+".npz"]
        assert all(c["error"] <= c["bound"] for c in record["checks"])
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_alignment.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Verified {len(manifest['files'])} first-attention release files; zero model calls. "
          "Numerical replication requires regenerated local arrays.", flush=True)


def plan(out, previous=None):
    steps = []
    if previous is None:
        steps.append(dict(name="regenerate-228-inputs", command=[sys.executable,
            str(ROOT/"research/boolean_coupling/reproduce_alignment.py"), "--run", "--out", str(out/"input-228")]))
        previous = out/"input-228/experiment"
    for name, module, arguments in (
        ("generator", "generator229b", ["--input", str(previous), "--out", str(out/"experiment")]),
        ("audit", "audit229", ["--run", str(out/"experiment")]),
    ):
        steps.append(dict(name=name, command=[sys.executable, "-m", "boolean_coupling."+module, *arguments]))
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-228", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/generator-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    previous = args.input_228.resolve() if args.input_228 else None
    steps = plan(out, previous)
    if args.offline:
        for step in steps:
            if step["name"] in ("regenerate-228-inputs", "generator"):
                step["command"].append("--offline")
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, retrospective=True,
            note="Fixed previously observed contexts; not fresh confirmation or a prospective predictor."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    verify()
    # Runtime import only: --verify-only and --plan require the standard library alone.
    from huggingface_hub import snapshot_download
    spec = json.loads((ROOT/"research/boolean_coupling/settings.json").read_text())["model"]
    snapshot_download(repo_id=spec["model"], revision=spec["revision"], local_files_only=args.offline)
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(wrapper_sha256=sha(__file__), steps=steps, completed=[], status="running")
    for step in steps:
        (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
        try:
            with (out/(step["name"]+".log")).open("w") as log:
                subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            record.update(status="failed", failed_step=step["name"])
            (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
            raise
        record["completed"].append(step["name"])
    record["status"] = "complete"
    (out/"pipeline.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"Completed: {out}")


if __name__ == "__main__":
    main()
