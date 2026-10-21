"""Verify251b/252 without models, or reproduce their source-bound forecasting tests."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {251:"fixed-receiver-attention-251", 252:"source-driven-attention-252"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded(checks):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in checks)


def verify(history):
    for num, name in NAMES.items():
        folder = ROOT/"results"/name
        manifest, design, fs, sealed, pre, status, final, audit, proof = (read(folder/n) for n in (
            "manifest.json", "design.json", "forecast-status.json", "forecast-seal.json", "pre-audit.json", "status.json", "seal.json", "audit.json", "measurement-proof.json"))
        for p, h in manifest["files"].items():
            assert sha(ROOT/p) == h, p
        for p, h in design["source_hashes"].items():
            assert sha(ROOT/p) == sha(folder/"sources"/Path(p).name) == h
        assert design["experiment"] == num and design["fresh_literals"] and design["dependent_templates"] and not design["replication"]
        assert design["primary"] == ("R" if num == 251 else "A")
        assert fs["status"] == status["status"] == "complete" and sealed["status"] == final["status"] == "sealed"
        assert fs["ordinary_true"] == pre["ordinary_true"] == 0 and status["ordinary_true"] == 32
        assert len(fs["cases"]) == len(status["cases"]) == 32 and fs["cases"] == sealed["cases"] and status["cases"] == final["cases"]
        assert sealed["design_sha256"] == sha(folder/"design.json") and sealed["status_sha256"] == sha(folder/"forecast-status.json")
        assert sealed["pre_audit_sha256"] == sha(folder/"pre-audit.json") and final["status_sha256"] == sha(folder/"status.json")
        assert status["forecast_seal_sha256"] == final["forecast_seal_sha256"] == audit["forecast_seal_sha256"] == proof["seal_sha256"] == sha(folder/"forecast-seal.json")
        assert proof["mode"] == "public_remote" and proof["commit"] == proof["remote_head"] and proof["accepted_utc"] > sealed["created_utc"]
        assert status["proof_sha256"] == sha(folder/"measurement-proof.json")
        assert (folder/"pre-native/forecast-seal.json").read_bytes() == (folder/"forecast-seal.json").read_bytes()
        for p, h in read(folder/"pre-native/receipt-manifest.json")["files"].items():
            assert sha(ROOT/p) == h
        inventory = {e["path"]:e["sha256"] for e in read(folder/"local-array-inventory.json")["arrays"]}
        assert inventory["weights.npz"] == design["weights_sha256"]
        for sub, entries in (("forecasts", fs["cases"]), ("observations", status["cases"])):
            for e in entries:
                path = folder/sub/(e["key"]+".json")
                meta = read(path)
                assert sha(path) == e["record_sha256"] and meta["arrays_sha256"] == e["arrays_sha256"] == inventory[sub+"/"+e["key"]+".npz"]
                assert e["witness_sha256"] == meta["witness_sha256"]
                if e["witness_sha256"]:
                    assert inventory[("witnesses-forecast" if sub == "forecasts" else "witnesses-native")+"/"+e["key"]+".npz"] == e["witness_sha256"]
                bounded(meta["checks"])
                if sub == "observations":
                    assert meta["native_started_utc"] > proof["accepted_utc"]
        for result, phase in ((pre, "pre"), (audit, "final")):
            assert result["status"] == "passed" and result["scorer_sha256"] == sha(ROOT/f"research/boolean_coupling/audit{num}{'b' if num == 251 else ''}.py")
            bounded(result["checks"])
            assert result["raw_logit_checks_sha256"] == sha(folder/("raw-logit-checks-"+phase+".csv"))
            assert result["raw_logit_checks"]["failed"] > 0 and result["raw_logit_checks"]["original_bound"] == 1e-4
        for n, h in audit["csv_hashes"].items():
            assert sha(folder/(n+".csv")) == h
        assert audit["seal_sha256"] == sha(folder/"seal.json")
        if num == 251:
            qualification = read(folder/"qualification251b.json")
            assert pre["qualification_sha256"] == audit["qualification_sha256"] == sha(folder/"qualification251b.json")
            assert qualification["original_failure_sha256"] == sha(folder/"failure-audit-pre.json")
            for p, h in qualification["source_hashes"].items():
                assert sha(ROOT/p) == sha(folder/"sources"/Path(p).name) == h
            assert (folder/"preserved-disk-failure/failure-forecast.json").exists()
        count_name = "extra_full_PV" if num == 251 else "extra_full_QK_softmax_PV_each"
        assert fs["counts"] == dict(full_forwards=160, witness_extra_QK=60, **{count_name:3584})
        assert status["counts"] == dict(full_forwards=64, witness_extra_QK=24, **{count_name:896})
    if history:
        subprocess.run([sys.executable, str(ROOT/"research/boolean_coupling/reproduce_communication.py"), "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified251b/252 complete forecasts, public pre-native receipts, numerical boundaries and retained raw-logit failures. Integrity is not numerical replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only", "history", "plan", "run", "offline"):
        parser.add_argument("--"+name, action="store_true")
    parser.add_argument("--experiment", choices=("251", "252", "both"), default="both")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = (args.out or ROOT/"runs/attention-reproduction").resolve()
    selected = (251, 252) if args.experiment == "both" else (int(args.experiment),)
    steps = []
    for num in selected:
        dest = out/f"experiment{num}"
        def step(label, module, *arguments, expected_raw_failure=False):
            steps.append(dict(name=f"{num}-"+label, command=[sys.executable, "-m", "boolean_coupling."+module, *map(str, arguments)], expected_raw_failure=expected_raw_failure))
        step("forecast", f"attention{num}", "--phase", "forecast", "--replication", "--out", dest)
        if num == 251:
            step("original-pre-audit", "audit251", "--phase", "pre", "--run", dest, expected_raw_failure=True)
        step("pre-audit", f"audit{num}"+("b" if num == 251 else ""), "--phase", "pre", "--run", dest)
        step("seal", f"attention{num}", "--phase", "seal", "--out", dest)
        steps.append(dict(name=f"{num}-local-proof", proof_dir=str(dest)))
        step("measure", f"attention{num}", "--phase", "measure", "--out", dest, "--proof", dest/"replication-proof.json")
        step("final-audit", f"audit{num}"+("b" if num == 251 else ""), "--phase", "final", "--run", dest)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Replicates published literals using local seal proof; not a new fresh-context test. Raw-logit failures explicitly retained."), indent=2))
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
    for entry in steps:
        try:
            if "proof_dir" in entry:
                dest = Path(entry["proof_dir"])
                proof = dict(mode="local_replication", seal_sha256=sha(dest/"forecast-seal.json"), accepted_utc=datetime.now(timezone.utc).isoformat())
                (dest/"replication-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
            else:
                with (out/(entry["name"]+".log")).open("x") as log:
                    result = subprocess.run(entry["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if entry["expected_raw_failure"]:
                    dest = Path(entry["command"][-1])
                    failure = read(dest/"failure-audit-pre.json")
                    assert result.returncode != 0 and "CPU64_QK" in failure["error"] and not (dest/"pre-audit.json").exists()
                else:
                    result.check_returncode()
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
