"""Verify/reproduce source-conditioned local forecasts; no model imports during verification."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inventory(folder, name="local-array-inventory.json"):
    return {r["path"]:r["sha256"] for r in read(folder/name)["arrays"]}


def checked_records(records):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in records)


def verify(history=False):
    local = ROOT/"results/source-conditioned-local-232"
    manifest, design = read(local/"manifest.json"), read(local/"design.json")
    for name, h in manifest["files"].items():
        assert sha(ROOT/name) == h, name
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(local/"sources"/Path(name).name) == h
    audit, qualification = read(local/"audit.json"), read(local/"forecast-audit.json")
    assert audit["status"] == qualification["status"] == "passed"
    assert audit["cases"] == 32 and audit["ordinary_true"] == 32
    checked_records(audit["checks"])
    checked_records(qualification["checks"])
    before, arrays = inventory(local, "forecast-array-inventory.json"), inventory(local)
    seal = read(local/"forecast-seal.json")
    for name, h in seal["files"].items():
        assert (before[name] if name in before else sha(local/name)) == h, name
    receipt, begin = read(local/"pre-observation-git.json"), read(local/"observation-start.json")
    assert seal["sealed_monotonic_ns"] < receipt["verified_monotonic_ns"] < begin["started_monotonic_ns"]
    assert receipt["remote_ref_verified"] and receipt["forecast_seal_remote_verified"]
    assert receipt["commit"] == manifest["original_forecast_commit"]
    assert begin["seal_sha256"] == sha(local/"forecast-seal.json")
    forecast, status = read(local/"forecast-manifest.json"), read(local/"status.json")
    assert forecast["counts"] == design["expected_forecast"]
    assert forecast["ordinary_true_before_seal"] == forecast["native_both_attention2_before_seal"] == 0
    cases = read(ROOT/"research/boolean_coupling/cases232.json")
    catalog = set(read(ROOT/"research/boolean_coupling/historical-input-hashes-232.json")["hashes"])
    assert len(cases) == len(forecast["entries"]) == len(status["observations"]) == 32
    fresh, derivatives = set(), 0
    for case, f, o in zip(cases, forecast["entries"], status["observations"]):
        assert case["key"] == f["key"] == o["key"]
        for bit in ("false", "true"):
            h = hashlib.sha256(json.dumps(case[bit+"_ids"], separators=(",", ":")).encode()).hexdigest()
            assert h not in catalog and h not in fresh
            fresh.add(h)
        assert len(case["false_ids"]) == len(case["true_ids"])
        changed = [i for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"])) if a != b]
        assert changed == [case["position"]]
        assert case["false_ids"][changed[0]] == 895 and case["true_ids"][changed[0]] == 830
        fp, op = (local/phase/(case["key"]+".json") for phase in ("forecasts", "observations"))
        fm, om = read(fp), read(op)
        assert sha(fp) == f["record_sha256"] and sha(op) == o["record_sha256"]
        assert fm["arrays_sha256"] == f["arrays_sha256"] == before["forecasts/"+case["key"]+".npz"]
        assert om["arrays_sha256"] == o["arrays_sha256"] == arrays["observations/"+case["key"]+".npz"]
        assert om["forecast_sha256"] == fm["arrays_sha256"]
        assert om["seal_sha256"] == sha(local/"forecast-seal.json")
        assert om["observed_monotonic_ns"] > begin["started_monotonic_ns"]
        checked_records(fm["checks"])
        checked_records(om["checks"])
        assert set(fm["derivatives"]) == {"F", "S"}
        for label in ("F", "S"):
            assert all(d["ad_error"] <= 1e-9 and d["fd_error"] <= 1e-4 for d in fm["derivatives"][label])
            derivatives += len(fm["derivatives"][label])
    assert derivatives == forecast["local_affine_derivatives"] == qualification["local_affine_derivatives"]
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_local.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified 232 release, independent anchor records, input freshness and global ordering; zero model calls. "
          "Omitted raw arrays require regeneration for numerical replication.", flush=True)


def plan(out, offline=False):
    return [dict(name=name, command=[sys.executable, "-m", "boolean_coupling."+module,
        "--phase", phase, "--out", str(out/"experiment"), *(["--offline"] if offline and module == "local232" else [])])
        for name, module, phase in (("forecast", "local232", "forecast"),
            ("qualify-and-seal", "audit232", "forecast"), ("observe-native", "local232", "observe"),
            ("score", "audit232", "score"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT/"runs/source-context-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    steps = plan(out, args.offline)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps, scope="All-receiving attention-2 forecast; exact edited first block is declared work.",
            note="Fixed original cases; repetition is not another fresh-input confirmation."), indent=2))
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
    env = dict(os.environ, PYTHONPATH=str(ROOT/"research"), OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    record = dict(status="running", steps=steps, completed=[], wrapper_sha256=sha(__file__))
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
    print("Completed source-context reproduction:", out)


if __name__ == "__main__":
    main()
