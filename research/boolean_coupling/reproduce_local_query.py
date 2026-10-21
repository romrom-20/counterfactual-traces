"""Verify 255/256 evidence or reproduce the sealed single-pass polynomial forecast."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(history):
    for name in ("local-query-curvature-255", "local-query-curvature-256"):
        folder = ROOT/"results"/name
        for p, h in read(folder/"manifest.json")["files"].items():
            assert sha(ROOT/p) == h, p
    local = ROOT/"results/local-query-curvature-255"
    a = read(local/"audit.json")
    assert a["status"] == "passed" and a["retrospective"] and a["model_calls"] == 0
    assert a["source_sha256"] == sha(ROOT/"research/boolean_coupling/curvature255.py")
    assert a["protocol_sha256"] == sha(ROOT/"docs/experiments/255-local-query-curvature.md")
    assert a["csv_sha256"] == sha(local/"local-errors.csv")
    previous = ROOT/"results/predicted-query-curvature-254"
    assert a["input_audit_sha256"] == sha(previous/"audit.json")
    assert a["input_forecast_seal_sha256"] == sha(previous/"forecast-seal.json")
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in a["checks"])
    folder = ROOT/"results/local-query-curvature-256"
    design, fs, sealed, pre, status, final, audit, proof = (read(folder/n) for n in (
        "design.json", "forecast-status.json", "forecast-seal.json", "pre-audit.json", "status.json", "seal.json", "audit.json", "measurement-proof.json"))
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(folder/"sources"/Path(p).name) == h
    assert design["experiment"] == 256 and design["primary"] == "C" and not design["replication"]
    assert design["fresh_literals"] and design["dependent_templates"]
    assert fs["status"] == status["status"] == "complete" and sealed["status"] == final["status"] == "sealed"
    assert fs["ordinary_true"] == pre["ordinary_true"] == 0 and status["ordinary_true"] == 32
    assert len(fs["cases"]) == len(status["cases"]) == 32
    assert fs["cases"] == sealed["cases"] and status["cases"] == final["cases"]
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
        for entry in entries:
            path = folder/sub/(entry["key"]+".json")
            meta = read(path)
            assert sha(path) == entry["record_sha256"] and meta["arrays_sha256"] == entry["arrays_sha256"] == inventory[sub+"/"+entry["key"]+".npz"]
            assert entry["witness_sha256"] == meta["witness_sha256"]
            if entry["witness_sha256"]:
                assert inventory[("witnesses-forecast" if sub == "forecasts" else "witnesses-native")+"/"+entry["key"]+".npz"] == entry["witness_sha256"]
            assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in meta["checks"])
            if sub == "observations":
                assert meta["native_started_utc"] > proof["accepted_utc"]
    for result, phase in ((pre, "pre"), (audit, "final")):
        assert result["status"] == "passed" and result["scorer_sha256"] == sha(ROOT/"research/boolean_coupling/audit256.py")
        assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in result["checks"])
        assert result["raw_logit_checks_sha256"] == sha(folder/("raw-logit-checks-"+phase+".csv"))
        assert result["raw_logit_checks"]["original_bound"] == 1e-4
    for name, h in audit["csv_hashes"].items():
        assert sha(folder/(name+".csv")) == h
    assert audit["seal_sha256"] == sha(folder/"seal.json")
    assert fs["counts"] == dict(full_forwards=192, CPU64_tangent_maps=4480, CPU64_quadratic_maps=2688, CPU64_cubic_maps=896, extra_full_QK_softmax_PV_each=896, witness_extra_QK=72)
    assert status["counts"] == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=24)
    radii = read(folder/"local-radii.json")
    assert radii["posthoc"] and radii["model_calls"] == 0
    assert radii["source_sha256"] == sha(ROOT/"research/boolean_coupling/radius256.py")
    assert radii["input_audit_sha256"] == sha(folder/"audit.json")
    assert len(radii["cases"]) == 160 and len(radii["checks"]) == 4320
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in radii["checks"])
    with (folder/"outcomes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 128 and {r["arm"] for r in rows} == {"L", "Q", "C", "R"}
    for row in rows:
        passes = all(float(row[s+"_norm"]) >= .001 and float(row[s+"_relative"]) <= .05 and float(row[s+"_cosine"]) >= .99 for s in ("all", "verdict")) and float(row["margin_error"]) <= .10
        assert (row["passes"] == "True") == passes
    if history:
        subprocess.run([sys.executable, str(ROOT/"research/boolean_coupling/reproduce_query.py"), "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified255/256 algebra, source and pre-native seals, all outcomes and retained numerical boundaries. Integrity is not model replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only", "history", "plan", "run", "offline"):
        parser.add_argument("--"+name, action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = (args.out or ROOT/"runs/local-query-reproduction").resolve()
    steps = []
    for phase, module in (("forecast", "attention256"), ("pre", "audit256"), ("seal", "attention256"), ("proof", None), ("measure", "attention256"), ("final", "audit256")):
        if module is None:
            steps.append(dict(name=phase))
            continue
        command = [sys.executable, "-m", "boolean_coupling."+module, "--phase", phase, "--run" if module.startswith("audit") else "--out", str(out/"experiment256")]
        if phase == "forecast":
            command.append("--replication")
        if phase == "measure":
            command += ["--proof", str(out/"experiment256/replication-proof.json")]
        steps.append(dict(name=phase, command=command))
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Published-literal numerical replication with local pre-native proof; not fresh confirmation."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite scientific evidence")
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", completed=[], wrapper_sha256=sha(__file__))
    for step in steps:
        try:
            if step["name"] == "proof":
                folder = out/"experiment256"
                proof = dict(mode="local_replication", seal_sha256=sha(folder/"forecast-seal.json"), accepted_utc=datetime.now(timezone.utc).isoformat())
                (folder/"replication-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
            else:
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
