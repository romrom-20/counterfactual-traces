"""Verify all evidence or reproduce source-transport studies from fixed public inputs."""

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


def inventory(folder, name="local-array-inventory.json"):
    return {r["path"]:r["sha256"] for r in read(folder/name)["arrays"]}


def checks(records):
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in records)


def release(folder):
    manifest = read(folder/"manifest.json")
    for name, h in manifest["files"].items():
        assert sha(ROOT/name) == h, name
    design = read(folder/"design.json")
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(folder/"sources"/Path(name).name) == h
    return manifest, design


def verify(history=False):
    old = ROOT/"results/source-conditioned-local-232"
    local = ROOT/"results/source-transport-primitives-233"
    _, design = release(local)
    before = inventory(old)
    for name, h in design["inputs"].items():
        assert (before[name] if name in before else sha(old/name)) == h, name
    status, audit = read(local/"status.json"), read(local/"audit.json")
    arrays = inventory(local)
    assert audit["status"] == "passed" and status["status"] == "complete"
    checks(audit["checks"])
    assert len(status["cases"]) == 32 and audit["ordinary_models"] == 0
    qualification = read(local/"qualification-status.json")
    assert len(qualification["cases"]) == 32 and qualification["status"] == "complete"
    for case, q in zip(status["cases"], qualification["cases"]):
        assert case["key"] == q["key"]
        stem = case["key"]
        meta = local/"cases"/(stem+".json")
        record = read(meta)
        assert sha(meta) == case["record_sha256"]
        assert record["arrays_sha256"] == case["arrays_sha256"] == arrays["cases/"+stem+".npz"]
        assert q["arrays_sha256"] == arrays["qualification/"+stem+".npz"]
        assert record["input_forecast_sha256"] == before["forecasts/"+stem+".npz"]
        assert record["input_observation_sha256"] == before["observations/"+stem+".npz"]
        checks(record["checks"])
    assert read(local/"additive-description.json")["explicitly_post_observation"]
    local = ROOT/"results/source-key-value-234"
    manifest, design = release(local)
    before, arrays = inventory(local, "forecast-array-inventory.json"), inventory(local)
    seal, receipt, begin = (read(local/n) for n in ("forecast-seal.json", "pre-observation-git.json", "observation-start.json"))
    for name, h in seal["files"].items():
        assert (before[name] if name in before else sha(local/name)) == h, name
    assert seal["sealed_monotonic_ns"] < receipt["verified_monotonic_ns"] < begin["started_monotonic_ns"]
    assert receipt["commit"] == manifest["original_forecast_commit"]
    assert receipt["remote_ref_verified"] and receipt["forecast_seal_remote_verified"]
    assert begin["seal_sha256"] == sha(local/"forecast-seal.json")
    forecast, audit = read(local/"forecast-manifest.json"), read(local/"forecast-audit.json")
    assert forecast["ordinary_true_before_seal"] == forecast["native_both_attention2_before_seal"] == 0
    assert forecast["counts"] == design["expected_forecast"]
    checks(audit["checks"])
    assert audit["status"] == "passed" and audit["fresh_hashes"] == 64
    status, score = read(local/"status.json"), read(local/"audit.json")
    assert status["status"] == "complete" and score["status"] == "passed" and status["ordinary_true"] == 32
    checks(score["checks"])
    cases = read(ROOT/"research/boolean_coupling/cases234.json")
    catalog = set(read(ROOT/"research/boolean_coupling/historical-input-hashes-234.json")["hashes"])
    fresh, directions = set(), 0
    qualification = read(local/"qualification-status.json")["cases"]
    assert len(cases) == len(status["observations"]) == len(forecast["entries"]) == len(qualification) == 32
    for case, f, o, q in zip(cases, forecast["entries"], status["observations"], qualification):
        key = case["key"]
        assert key == f["key"] == o["key"] == q["key"]
        for bit in ("false", "true"):
            h = hashlib.sha256(json.dumps(case[bit+"_ids"], separators=(",", ":")).encode()).hexdigest()
            assert h not in fresh and h not in catalog
            fresh.add(h)
        assert len(case["false_ids"]) == len(case["true_ids"])
        changed = [i for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"])) if a != b]
        assert changed == [case["position"]]
        assert case["false_ids"][changed[0]] == 895 and case["true_ids"][changed[0]] == 830
        fp, op = (local/phase/(key+".json") for phase in ("forecasts", "observations"))
        fm, om = read(fp), read(op)
        assert sha(fp) == f["record_sha256"] and sha(op) == o["record_sha256"]
        assert fm["arrays_sha256"] == f["arrays_sha256"] == before["forecasts/"+key+".npz"]
        assert om["arrays_sha256"] == o["arrays_sha256"] == arrays["observations/"+key+".npz"]
        assert q["arrays_sha256"] == before["qualification/"+key+".npz"]
        assert om["forecast_sha256"] == fm["arrays_sha256"] and om["seal_sha256"] == sha(local/"forecast-seal.json")
        assert om["observed_monotonic_ns"] > begin["started_monotonic_ns"]
        checks(fm["checks"])
        checks(om["checks"])
        assert set(fm["derivatives"]) == {"F"}
        for d in fm["derivatives"]["F"]:
            assert math.isfinite(d["ad_error"]) and d["ad_error"] <= 1e-9
            assert math.isfinite(d["fd_error"]) and d["fd_error"] <= 1e-4
            directions += 1
    assert directions == forecast["local_affine_derivatives"] == audit["local_affine_derivatives"] == 192
    command = [sys.executable, str(ROOT/"research/boolean_coupling/reproduce_source_context.py"), "--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Verified 233/234: original sources, all cases, qualified forecast ordering and retained failures; "
          "zero model calls. Numerical replication requires regenerated arrays.", flush=True)


def plan(out, study, input232=None, offline=False):
    steps = []
    if study in ("233", "both"):
        previous = input232 or out/"input232"/"experiment"
        if not input232:
            steps.append(dict(name="regenerate-fixed-232", command=[sys.executable,
                "research/boolean_coupling/reproduce_source_context.py", "--run", "--out", str(out/"input232"),
                *(["--offline"] if offline else [])]))
        steps += [dict(name="233-diagnostic", command=[sys.executable, "-m", "boolean_coupling.transport233",
                      "--input-232", str(previous), "--out", str(out/"experiment233")]),
                  dict(name="233-qualification", command=[sys.executable, "-m", "boolean_coupling.audit233",
                      "--run", str(out/"experiment233")]),
                  dict(name="233-posthoc-description", command=[sys.executable, "-m", "boolean_coupling.describe233",
                      "--run", str(out/"experiment233")])]
    if study in ("234", "both"):
        steps += [dict(name="234-"+phase, command=[sys.executable, "-m", "boolean_coupling."+module,
                    "--phase", phase, "--out", str(out/"experiment234"),
                    *(["--offline"] if offline and module == "local234" else [])])
                  for module, phase in (("local234", "forecast"), ("audit234", "forecast"),
                      ("local234", "observe"), ("audit234", "score"))]
        steps[-3]["name"] = "234-qualify-and-seal"
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--study", choices=("233", "234", "both"), default="234")
    parser.add_argument("--input-232", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/transport-reproduction")
    args = parser.parse_args()
    out = args.out.resolve()
    steps = plan(out, args.study, args.input_232.resolve() if args.input_232 else None, args.offline)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(study=args.study, steps=steps,
            scope="Fixed original inputs; repetition is not fresh confirmation. 233 retrospective; 234 local forecast."), indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    verify()
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
    print("Completed fixed-input reproduction:", out)


if __name__ == "__main__":
    main()
