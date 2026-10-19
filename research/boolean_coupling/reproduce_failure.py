"""Verify global failure diagnostics or regenerate them from fixed public inputs."""

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
    return {r["path"]:r["sha256"] for r in read(folder/"local-array-inventory.json")["arrays"]}


def checked(records):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in records)


def verify(history=False):
    folders = [ROOT/"results"/name for name in ("global-error-budget-235", "upstream-factorial-236")]
    old = ROOT/"results/false-only-alignment-228"
    before = inventory(old)
    for number, folder in zip((235, 236), folders):
        manifest, design, status, audit = (read(folder/n) for n in ("manifest.json", "design.json", "status.json", "audit.json"))
        for name, h in manifest["files"].items():
            assert sha(ROOT/name) == h, name
        for name, h in design["source_hashes"].items():
            assert sha(ROOT/name) == sha(folder/"sources"/Path(name).name) == h
        for name, h in design["inputs" if number == 235 else "input_hashes"].items():
            assert (before[name] if name in before else sha(old/name)) == h, name
        assert status["status"] == "complete" and audit["status"] == "passed" and audit["retrospective"]
        assert status["counts"] == audit["counts"] and status["counts"]["ordinary_true"] == 0
        assert len(status["cases"]) == 32
        checked(audit["checks"])
        arrays = inventory(folder)
        for entry in status["cases"]:
            path = folder/"cases"/(entry["key"]+".json")
            record = read(path)
            assert sha(path) == entry["record_sha256"]
            assert entry["arrays_sha256"] == record["arrays_sha256"] == arrays["cases/"+entry["key"]+".npz"]
            checked(record["checks"])
        if number == 235:
            ledger, selection = read(folder/"ledger-status.json"), read(folder/"selection.json")
            assert ledger["status"] == "complete" and ledger["ordinary_models"] == 0
            assert ledger["selection_sha256"] == sha(folder/"selection.json")
            assert selection["selected"] == [45, 43, 41, 39]
            assert status["weights_sha256"] == arrays["weights.npz"]
            checked(ledger["checks"])
            for entry in ledger["cases"]:
                path = folder/"ledger"/(entry["key"]+".json")
                assert sha(path) == entry["record_sha256"]
                assert read(path)["arrays_sha256"] == entry["arrays_sha256"] == arrays["ledger/"+entry["key"]+".npz"]
        else:
            assert design["audit235_sha256"] == sha(folders[0]/"audit.json")
            assert design["weights235_sha256"] == inventory(folders[0])["weights.npz"]
            assert audit["fd_count"] == status["counts"]["jvps"] == 1664
            assert audit["finite_mlp_reconstructions"] == 1024
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_transport.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified 235/236: complete old-input provenance, all ledgers/branches and numerical records; zero model calls. "
          "Omitted arrays must be regenerated for numerical replication.", flush=True)


def plan(out, study, input228=None, input235=None, offline=False):
    steps = []
    previous235 = input235 or out/"experiment235"
    if not input235:
        previous228 = input228 or out/"input228"/"experiment"
        if not input228:
            steps.append(dict(name="regenerate-fixed-228", command=[sys.executable,
                "research/boolean_coupling/reproduce_alignment.py", "--run", "--out", str(out/"input228"),
                *(["--offline"] if offline else [])]))
        steps += [dict(name="235-ledger", command=[sys.executable, "-m", "boolean_coupling.ledger235",
                    "--input-228", str(previous228), "--out", str(previous235)]),
                  dict(name="235-finite", command=[sys.executable, "-m", "boolean_coupling.finite235", "--run", str(previous235)]),
                  dict(name="235-audit", command=[sys.executable, "-m", "boolean_coupling.audit235", "--run", str(previous235)])]
    if study != "235":
        steps += [dict(name="236-factorial", command=[sys.executable, "-m", "boolean_coupling.continue236",
                    "--input-235", str(previous235), "--out", str(out/"experiment236")]),
                  dict(name="236-audit", command=[sys.executable, "-m", "boolean_coupling.audit236", "--run", str(out/"experiment236")])]
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--study", choices=("235", "236", "both"), default="both")
    parser.add_argument("--input-228", type=Path)
    parser.add_argument("--input-235", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/global-failure-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.input_235 and (args.input_228 or args.study == "235"):
        parser.error("Use --input-235 only to reuse a qualified prerequisite for 236.")
    steps = plan(out, args.study, args.input_228.resolve() if args.input_228 else None,
        args.input_235.resolve() if args.input_235 else None, args.offline)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Retrospective fixed inputs; oracle controls are not prospective predictors."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    verify()
    from huggingface_hub import snapshot_download
    spec = read(ROOT/"research/boolean_coupling/settings.json")["model"]
    snapshot_download(repo_id=spec["model"], revision=spec["revision"], local_files_only=args.offline)
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", completed=[], steps=steps, wrapper_sha256=sha(__file__))
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
    print("Completed fixed-input diagnostics:", out)


if __name__ == "__main__":
    main()
