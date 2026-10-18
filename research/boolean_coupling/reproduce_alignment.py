"""Verify the published alignment study or run its sealed six-phase sequence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"results/false-only-alignment-228"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_ordering():
    cases=json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text())
    catalog=set(json.loads((ROOT/"research/boolean_coupling/historical-input-hashes-228.json").read_text())["hashes"])
    hashes=set()
    inventory={p["path"]:p["sha256"] for p in json.loads((RESULTS/"local-array-inventory.json").read_text())["arrays"]}
    seal=json.loads((RESULTS/"forecast-qualification-seal.json").read_text())
    for name,expected in seal["files"].items():
        assert (inventory[name] if name in inventory else sha(RESULTS/name))==expected,name
    begin=json.loads((RESULTS/"observation-start.json").read_text())
    assert begin["started_monotonic_ns"]>seal["sealed_monotonic_ns"]
    forecast=json.loads((RESULTS/"forecast-manifest.json").read_text())
    assert forecast["native_true_full_models_before_seal"]==0 and len(forecast["entries"])==32
    assert json.loads((RESULTS/"forecast-status.json").read_text())["counts"]["ordinary_true"]==0
    for case,entry in zip(cases,forecast["entries"]):
        assert case["key"]==entry["key"]
        for bit in ("false","true"):
            h=hashlib.sha256(json.dumps(case[bit+"_ids"],separators=(",",":")).encode()).hexdigest()
            assert h not in catalog and h not in hashes
            hashes.add(h)
        changed=[i for i,(a,b) in enumerate(zip(case["false_ids"],case["true_ids"])) if a!=b]
        assert len(case["false_ids"])==len(case["true_ids"]) and changed==[case["position"]]
        assert case["false_ids"][changed[0]]==895 and case["true_ids"][changed[0]]==830
        fp=RESULTS/"forecasts"/(case["key"]+".json")
        op=RESULTS/"observations"/(case["key"]+".json")
        f,o=json.loads(fp.read_text()),json.loads(op.read_text())
        assert sha(fp)==entry["record_sha256"] and f["arrays_sha256"]==entry["arrays_sha256"]
        assert inventory["forecasts/"+case["key"]+".npz"]==f["arrays_sha256"]
        assert inventory["observations/"+case["key"]+".npz"]==o["arrays_sha256"]
        assert o["forecast_sha256"]==f["arrays_sha256"] and o["prefix_hashes"]==f["prefix_hashes"]
        assert o["observed_monotonic_ns"]>begin["started_monotonic_ns"]
        assert o["qualification_seal_sha256"]==sha(RESULTS/"forecast-qualification-seal.json")
    assert len(cases)==32 and len(hashes)==64


def verify(history=False):
    manifest=json.loads((RESULTS/"manifest.json").read_text())
    for name,expected in manifest["files"].items():
        assert sha(ROOT/name)==expected,name
    design=json.loads((RESULTS/"design.json").read_text())
    for name,expected in design["source_hashes"].items():
        assert sha(ROOT/name)==expected,name
    correction=json.loads((RESULTS/"score-correction.json").read_text())
    original=ROOT/"research/boolean_coupling/audit228.py"
    corrected=ROOT/"research/boolean_coupling/audit228b.py"
    assert sha(original)==correction["original_source_sha256"]
    assert sha(corrected)==correction["corrected_source_sha256"]
    expected=original.read_text().replace("arm=arm,block=index+1,**gg","arm=arm,layer=index+1,**gg")
    expected=expected.replace("Independent NumPy qualification, sealed-order checks and frozen scoring for228.",
        "228b reporting-only layer-key correction; original228 sources and numerical rules unchanged.")
    assert expected==corrected.read_text(),"Only the declared reporting-key/docstring correction is permitted."
    verify_ordering()
    command=[sys.executable,str(ROOT/"research/boolean_coupling/reproduce_tail.py"),"--verify-only"]
    if history:
        command.append("--history")
    subprocess.run(command,cwd=ROOT,check=True)
    print(f"Verified {len(manifest['files'])} alignment release files and source seals; zero model calls. "
          "Raw numerical audits require regenerated arrays.",flush=True)


def plan(out):
    preflight=out/"preflight"
    experiment=out/"experiment"
    steps=[]
    for name,module,phase,destination,extra in (
        ("preflight","alignment228","preflight",preflight,[]),
        ("preflight-audit","audit228","preflight",preflight,[]),
        ("forecast","alignment228","forecast",experiment,["--preflight",str(preflight)]),
        ("forecast-audit-and-seal","audit228","forecasts",experiment,[]),
        ("observe-native-true","alignment228","observe",experiment,[]),
        ("score","audit228b","score",experiment,[]),
    ):
        steps.append(dict(name=name,command=[sys.executable,"-m","boolean_coupling."+module,
            "--phase",phase,"--out",str(destination),*extra]))
    return steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--history",action="store_true")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--out",type=Path,default=ROOT/"runs/alignment-reproduction")
    args=parser.parse_args()
    out=args.out.resolve()
    steps=plan(out)
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps,full_true_before_global_seal=0,
            note="Replication reuses the original fixed inputs; it is not a second fresh-context demonstration."),indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence.")
    verify()
    from huggingface_hub import snapshot_download
    spec=json.loads((ROOT/"research/boolean_coupling/settings.json").read_text())["model"]
    snapshot_download(repo_id=spec["model"],revision=spec["revision"],local_files_only=args.offline)
    out.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/"research"),OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    record=dict(wrapper_sha256=sha(__file__),steps=steps,completed=[],status="running")
    for step in steps:
        (out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")
        print("Running",step["name"],flush=True)
        try:
            with (out/(step["name"]+".log")).open("w") as log:
                subprocess.run(step["command"],cwd=ROOT,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        except Exception as exc:
            record.update(status="failed",failed_step=step["name"],error=str(exc))
            (out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")
            raise
        record["completed"].append(step["name"])
    record["status"]="complete"
    (out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
