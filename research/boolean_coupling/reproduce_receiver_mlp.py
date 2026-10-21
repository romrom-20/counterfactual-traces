"""Verify 258 receiver-MLP evidence or reproduce its sealed forecasting pipeline."""

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
from statistics import median

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded(checks):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in checks)


def verify259(input258):
    folder = ROOT/"results/silu-score-factors-259"
    for p, h in read(folder/"manifest.json")["files"].items():
        assert sha(ROOT/p) == h, p
    audit = read(folder/"audit.json")
    assert audit["status"] == "passed" and audit["experiment"] == 259
    assert audit["retrospective"] and audit["model_calls"] == 0
    assert audit["source_sha256"] == sha(ROOT/"research/boolean_coupling/score259.py")
    assert audit["protocol_sha256"] == sha(ROOT/"docs/experiments/259-silu-score-factors.md")
    assert audit["input_audit_sha256"] == sha(input258/"audit.json")
    assert audit["input_forecast_seal_sha256"] == sha(input258/"forecast-seal.json")
    assert audit["csv_sha256"] == sha(folder/"score-factors.csv")
    bounded(audit["checks"])
    assert len(audit["checks"]) == 72 and len(audit["fixed_inputs"]) == 24
    with (folder/"score-factors.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    scores = {"BUW", "BU", "BW", "UW", "B", "U", "W", "DUW", "D"}
    assert len(rows) == 432 and {r["score"] for r in rows} == scores
    identities = [(r["key"], r["arm"], r["layer"], r["score"], r["scope"]) for r in rows]
    assert len(set(identities)) == 432
    for row in rows:
        assert all(math.isfinite(float(row[k])) for k in ("gap_norm", "remaining_relative", "signed_share", "restored_norm", "mask_overlap"))
        assert (row["eligible"] == "True") == (float(row["gap_norm"]) >= 1e-8)
        assert 0 <= float(row["mask_overlap"]) <= 1
        if row["score"] == "BUW":
            assert float(row["mask_overlap"]) == 1
    for arm in ("C", "S"):
        for scope in ("all", "verdict"):
            for score in scores:
                selected = [r for r in rows if r["arm"] == arm and r["scope"] == scope and r["score"] == score and r["eligible"] == "True"]
                assert len(selected) == 12
                expected = audit["summary"][arm+"_"+scope][score]
                assert math.isclose(median(float(r["remaining_relative"]) for r in selected), expected["median_remaining"], rel_tol=1e-12)
                assert math.isclose(median(float(r["mask_overlap"]) for r in selected), expected["median_overlap"], rel_tol=1e-12)
    whole = [r for r in rows if r["scope"] == "all" and r["eligible"] == "True"]
    remaining = {score:median(float(r["remaining_relative"]) for r in whole if r["score"] == score) for score in scores}
    assert math.isclose(remaining["UW"]/remaining["BUW"], audit["dropped_bound_median_ratio"], rel_tol=1e-12)
    assert audit["dropped_bound_meets_2x"] == (remaining["UW"] >= 2*remaining["BUW"])
    within = 0
    for item in audit["fixed_inputs"]:
        selected = [r for r in whole if r["key"] == item["key"] and r["arm"] == item["arm"] and int(r["layer"]) == item["layer"]]
        pair = {r["score"]:float(r["remaining_relative"]) for r in selected}
        within += bool("BU" in pair and "BUW" in pair and pair["BU"]-pair["BUW"] <= .05)
    assert within == audit["dropped_column_norm_within_05"]
    assert audit["dropped_column_norm_meets_18"] == (within >= 18)
    description = read(folder/"description.json")
    assert description["source_sha256"] == sha(ROOT/"research/boolean_coupling/describe_score_factors.py")
    assert description["audit_sha256"] == sha(folder/"audit.json") and description["csv_sha256"] == sha(folder/"score-factors.csv")
    for name, h in description["figure_sha256"].items():
        assert sha(ROOT/"docs/figures/259"/name) == h
    print("Verified259: all432 retrospective score records,72 checks, both registered summaries; no new model measurement.", flush=True)


def verify(history):
    folder = ROOT/"results/receiver-mlp-curvature-258"
    manifest = read(folder/"manifest.json")
    for p, h in manifest["files"].items():
        assert sha(ROOT/p) == h, p
    design, fs, sealed, pre, status, final, audit, proof = (read(folder/n) for n in (
        "design.json", "forecast-status.json", "forecast-seal.json", "pre-audit.json", "status.json", "seal.json", "audit.json", "measurement-proof.json"))
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(folder/"sources"/Path(p).name) == h
    assert design["experiment"] == 258 and design["primary"] == "S" and not design["replication"]
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
            bounded(meta["checks"])
            if sub == "observations":
                assert meta["native_started_utc"] > proof["accepted_utc"]
    for result, phase in ((pre, "pre"), (audit, "final")):
        assert result["status"] == "passed" and result["scorer_sha256"] == sha(ROOT/"research/boolean_coupling/audit258b.py")
        bounded(result["checks"])
        assert result["raw_logit_checks_sha256"] == sha(folder/("raw-logit-checks-"+phase+".csv"))
        assert result["raw_logit_checks"]["original_bound"] == 1e-4
        assert result["raw_down_checks_sha256"] == sha(folder/("raw-down-checks-"+phase+".csv"))
        assert result["raw_down_checks"]["original_bound"] == 1e-4 and result["raw_down_checks"]["failed"] > 0
        assert result["qualification_sha256"] == sha(folder/"qualification258b.json")
    for name, h in audit["csv_hashes"].items():
        assert sha(folder/(name+".csv")) == h
    assert audit["seal_sha256"] == sha(folder/"seal.json")
    qualification = read(folder/"qualification258b.json")
    assert qualification["ordinary_true"] == 0 and qualification["created_utc"] < sealed["created_utc"]
    assert qualification["design_sha256"] == sha(folder/"design.json") and qualification["forecast_status_sha256"] == sha(folder/"forecast-status.json")
    assert qualification["original_scorer_sha256"] == sha(ROOT/"research/boolean_coupling/audit258.py")
    assert qualification["original_failure_sha256"] == sha(folder/"failure-audit-pre.json")
    assert qualification["investigation_sha256"] == sha(folder/"projection-investigation.json")
    for p, h in qualification["source_hashes"].items():
        assert sha(ROOT/p) == sha(folder/"qualification-sources"/Path(p).name) == h
    assert read(folder/"projection-investigation.json")["maximum_full_shape_error"] == 0
    assert read(folder/"projection-investigation.json")["original_CPU64_failures"] > 0
    assert fs["counts"] == dict(full_forwards=192, CPU64_tangent_maps=4480, CPU64_quadratic_maps=4480, CPU64_cubic_maps=4480, CPU64_SiLU_selections=4480, CPU64_SiLU_FD_checks=8960, False_mlp_anchors=896, extra_receiving_down_projections=14336, extra_full_QK_softmax_PV_each=896, witness_extra_QK=72)
    assert status["counts"] == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=24)
    with (folder/"outcomes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 128 and {r["arm"] for r in rows} == {"C", "Q", "S", "R"}
    for row in rows:
        passed = all(float(row[s+"_norm"]) >= .001 and float(row[s+"_relative"]) <= .05 and float(row[s+"_cosine"]) >= .99 for s in ("all", "verdict")) and float(row["margin_error"]) <= .10
        assert (row["passes"] == "True") == passed
    description = read(folder/"description.json")
    assert description["source_sha256"] == sha(ROOT/"research/boolean_coupling/describe_receiver_mlp.py")
    assert description["audit_sha256"] == sha(folder/"audit.json")
    for name, h in description["figure_sha256"].items():
        assert sha(ROOT/"docs/figures/258"/name) == h
    verify259(folder)
    if history:
        subprocess.run([sys.executable, str(ROOT/"research/boolean_coupling/reproduce_source_radius.py"), "--verify-only", "--history"], cwd=ROOT, check=True)
    print("Verified258 sparse receiver-SiLU forecasts, masks, source/public seals, all outcomes and retained numerical boundaries. Integrity is not model replication.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only", "history", "plan", "run", "offline"):
        parser.add_argument("--"+name, action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = (args.out or ROOT/"runs/receiver-mlp-reproduction").resolve()
    steps = []
    for phase, module in (("forecast", "mlp258"), ("qualification", "qualify258b"), ("pre", "audit258b"), ("seal", "mlp258"), ("proof", None), ("measure", "mlp258"), ("final", "audit258b")):
        if module is None:
            steps.append(dict(name=phase))
            continue
        if module == "qualify258b":
            command = [sys.executable, "-m", "boolean_coupling."+module, "--run", str(out/"experiment258")]
        else:
            command = [sys.executable, "-m", "boolean_coupling."+module, "--phase", phase, "--run" if module.startswith("audit") else "--out", str(out/"experiment258")]
        if phase == "forecast":
            command.append("--replication")
        if phase == "measure":
            command += ["--proof", str(out/"experiment258/replication-proof.json")]
        steps.append(dict(name=phase, command=command))
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="Published-literal numerical replication with local pre-native proof; not new confirmation."), indent=2))
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
                folder = out/"experiment258"
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
