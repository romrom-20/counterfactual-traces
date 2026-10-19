"""Verify242b seals without model imports, or replicate its uniform-rule test."""

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
    assert all(math.isfinite(r["error"]) and r["error"] <= r["bound"] for r in rows)


def verify(history=False):
    folder = ROOT / "results/uniform-primitive-242b"
    manifest = read(folder / "manifest.json")
    for name, h in manifest["files"].items():
        assert sha(ROOT / name) == h, name
    for name, h in read(folder / "pre-publication-manifest.json")["files"].items():
        assert sha(ROOT / name) == h, name
    design, forecasts, seal, pre, status, audit, proof = (read(folder / n) for n in
        ("design.json", "forecast-status.json", "forecast-seal.json", "pre-measurement-audit.json",
        "status.json", "audit.json", "measurement-proof.json"))
    assert design["experiment"] == "242b" and design["primary"] == "100" and not design["replication"]
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(folder / "sources" / Path(name).name) == h
    assert forecasts["ordinary_true"] == seal["ordinary_true"] == pre["ordinary_true"] == 0
    assert seal["status"] == "sealed" and seal["design_sha256"] == sha(folder / "design.json")
    assert seal["manifest_sha256"] == sha(folder / "forecast-manifest.json")
    assert seal["pre_audit_sha256"] == sha(folder / "pre-measurement-audit.json")
    assert pre["status"] == audit["status"] == "passed" and status["ordinary_true"] == 32
    assert status["forecast_seal_sha256"] == proof["seal_sha256"] == sha(folder / "forecast-seal.json")
    assert status["proof_sha256"] == sha(folder / "measurement-proof.json")
    assert proof["mode"] == "private_remote" and proof["commit"] == proof["remote_head"]
    assert audit["outcomes_sha256"] == sha(folder / "outcomes.csv")
    description = read(folder / "description.json")
    assert description["posthoc"] and description["model_calls"] == 0
    assert description["audit_sha256"] == sha(folder / "audit.json")
    assert description["source_sha256"] == sha(ROOT / "research/boolean_coupling/describe242b.py")
    bounded(description["checks"])
    assert len(forecasts["cases"]) == len(status["cases"]) == 32
    bounded(pre["checks"] + audit["checks"])
    inventory = {x["path"]: x["sha256"] for x in read(folder / "local-array-inventory.json")["arrays"]}
    assert inventory["weights.npz"] == design["weights_sha256"]
    for f, o in zip(forecasts["cases"], status["cases"], strict=True):
        assert f["key"] == o["key"]
        for phase, ent in (("forecasts", f), ("observations", o)):
            meta_path = folder / phase / (ent["key"] + ".json")
            meta = read(meta_path)
            assert sha(meta_path) == ent["record_sha256"]
            assert meta["arrays_sha256"] == ent["arrays_sha256"] == inventory[phase + "/" + ent["key"] + ".npz"]
            if phase == "forecasts":
                bounded(meta["checks"] + meta["reference_checks"])
                assert meta["ordinary_true"] == 0
            else:
                assert meta["forecast_record_sha256"] == f["record_sha256"]
                assert meta["native_started_utc"] > proof["accepted_utc"] > seal["created_utc"]
    failed = ROOT / "results/uniform-primitive-242-failed"
    fm = read(failed / "manifest.json")
    for name, h in fm["files"].items():
        assert sha(ROOT / name) == h
    for name, h in read(failed / "design.json")["source_hashes"].items():
        assert sha(ROOT / name) == sha(failed / "sources" / Path(name).name) == h
    assert "NotImplementedError" in read(failed / "failure-forecast.json")["error"]
    if history:
        subprocess.run([sys.executable, str(ROOT / "research/boolean_coupling/reproduce_early.py"),
            "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified242b pre-native source/forecast seals, all primitive-check records and outcomes, and242's retained implementation failure. "
        "No model imports or forwards; integrity verification is not numerical replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/uniform-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    steps = [dict(name="forecast", command=[sys.executable, "-m", "boolean_coupling.run242b", "--phase", "forecast",
        "--replication", "--out", str(out)]),
        dict(name="pre-audit", command=[sys.executable, "-m", "boolean_coupling.audit242b", "--phase", "pre", "--run", str(out)]),
        dict(name="seal", command=[sys.executable, "-m", "boolean_coupling.run242b", "--phase", "seal", "--out", str(out)]),
        dict(name="measure", command=[sys.executable, "-m", "boolean_coupling.run242b", "--phase", "measure", "--out", str(out),
            "--proof", str(out / "replication-proof.json")]),
        dict(name="final-audit", command=[sys.executable, "-m", "boolean_coupling.audit242b", "--phase", "final", "--run", str(out)])]
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Replicates already observed242b literals; explicit local seal, no Git/network publication, no fresh-confirmation claim."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for step in steps:
        # The forecast command itself creates the new output directory.
        log_path = out.parent / (out.name + "-" + step["name"] + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if step["name"] == "measure":
            (out / "replication-proof.json").write_text(json.dumps(dict(mode="local_replication",
                seal_sha256=sha(out / "forecast-seal.json"), pre_audit_sha256=sha(out / "pre-measurement-audit.json")), indent=2) + "\n")
        with log_path.open("x") as log:
            subprocess.run(step["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Completed retrospective numerical replication:", out, flush=True)


if __name__ == "__main__":
    main()
