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


def verify(history=False):
    manifest=json.loads((RESULTS/"manifest.json").read_text())
    for name,expected in manifest["files"].items():
        assert sha(ROOT/name)==expected,name
    design=json.loads((RESULTS/"design.json").read_text())
    for name,expected in design["source_hashes"].items():
        assert sha(ROOT/name)==expected,name
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
        ("score","audit228","score",experiment,[]),
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
