"""Verify244/245/246 without model imports, or regenerate their public prerequisites."""

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
    assert all(math.isfinite(c["error"]) and c["error"]<=c["bound"] for c in rows)


def verify(history):
    for number,foldername,sealname in ((244,"native-attention-244","intervention-seal.json"),(245,"first-mixing-245","seal.json"),("246b","block9-origin-246b","seal.json")):
        folder=ROOT/"results"/foldername
        manifest,design,status,seal,audit=(read(folder/n) for n in ("manifest.json","design.json","status.json",sealname,"audit.json"))
        for name,h in manifest["files"].items():
            assert sha(ROOT/name)==h,name
        for name,h in design["source_hashes"].items():
            assert sha(ROOT/name)==sha(folder/"sources"/Path(name).name)==h
        assert design["experiment"]==number and design["retrospective"] and audit["status"]=="passed"
        assert status["status"]=="complete" and seal["status"]=="sealed"
        assert audit["design_sha256"]==seal["design_sha256"]==sha(folder/"design.json")
        assert seal["status_sha256"]==sha(folder/"status.json") and audit["seal_sha256"]==sha(folder/sealname)
        assert seal["cases"]==status["cases"] and len(status["cases"])==32
        bounded(audit["checks"])
        for name,h in audit["csv_hashes"].items():
            assert sha(folder/(name+".csv"))==h
        inventory={r["path"]:r["sha256"] for r in read(folder/"local-array-inventory.json")["arrays"]}
        for ent in status["cases"]:
            meta=read(folder/"cases"/(ent["key"]+".json"))
            assert sha(folder/"cases"/(ent["key"]+".json"))==ent["record_sha256"]
            assert meta["arrays_sha256"]==ent["arrays_sha256"]==inventory["cases/"+ent["key"]+".npz"]
            bounded(meta["checks"])
            first=ROOT/"results/uniform-primitive-242b"
            prior_entries={e["key"]:e for e in read(first/"forecast-status.json")["cases"]}
            assert meta["prior_forecast_sha256"]==prior_entries[ent["key"]]["arrays_sha256"]
            if number!=244:
                references={e["key"]:e for e in read(ROOT/"results/native-attention-244/status.json")["cases"]}
                assert meta["prior_regional_sha256"]==references[ent["key"]]["arrays_sha256"]
            records=meta["arms"].values() if number==244 else meta["qualifications"].values()
            for record in records:
                bounded(record["reference_checks"] if number==244 else record["checks"])
                counts=record["qualification_counts"] if number==244 else record["counts"]
                assert len(record["witnesses"])==counts.get("distinct_directions",0)
        if number==244:
            assert "distinct_directions" in read(folder/"failure-audit.json")["error"]
            assert audit["scorer_sha256"]==sha(ROOT/"research/boolean_coupling/audit244b.py")
            assert audit["original_scorer_sha256"]==sha(ROOT/"research/boolean_coupling/audit244.py")
            assert status["ordinary_true_calls"]==seal["ordinary_true_calls"]==32
            assert not seal["native_arrays_opened_by_runner"]
            assert status["counts"]["complete_forwards"]==160
            prior=ROOT/"results/regional-silu-243"
            for tag,name in (("design","design.json"),("audit","audit.json"),("status","status.json")):
                assert design["input_regional_"+tag+"_sha256"]==sha(prior/name)
        elif number==245:
            assert audit["scorer_sha256"]==sha(ROOT/"research/boolean_coupling/audit245.py")
            assert status["counts"]["primitive_intervention_forwards"]==224
            inputs=[ROOT/"results/native-attention-244"/n for n in ("design.json","status.json","audit.json")]
            inputs += [ROOT/"results/uniform-primitive-242b"/n for n in ("design.json","forecast-status.json")]
            expected=[sha(p) for p in inputs]
            prior_inventory=read(ROOT/"results/uniform-primitive-242b/local-array-inventory.json")["arrays"]
            expected += [next(r["sha256"] for r in prior_inventory if r["path"]=="weights.npz")]
            assert expected==list(design["inputs"].values())
        else:
            assert audit["scorer_sha256"]==sha(ROOT/"research/boolean_coupling/audit246b.py")
            assert status["counts"]["regional_replays"]==128 and status["counts"]["primitive_intervention_forwards"]==320
            assert "float64" in read(folder/"preserved-failure/failure.json")["error"]
            posthoc=read(folder/"position-summary.json")
            assert posthoc["posthoc"] and posthoc["model_calls"]==0
            assert posthoc["source_sha256"]==sha(ROOT/"research/boolean_coupling/position246.py")
            assert posthoc["csv_sha256"]==sha(folder/"position-effects.csv")
            assert design["input244_audit_sha256"]==sha(ROOT/"results/native-attention-244/audit.json")
            assert design["input244_status_sha256"]==sha(ROOT/"results/native-attention-244/status.json")
    if history:
        subprocess.run([sys.executable,str(ROOT/"research/boolean_coupling/reproduce_regional.py"),"--verify-only","--history"],cwd=ROOT,check=True)
    print("Verified244/245/246 complete published evidence and source seals; no model imports. Integrity is not numerical replication.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("verify-only","history","plan","run","offline"):
        parser.add_argument("--"+name,action="store_true")
    parser.add_argument("--input-242",type=Path)
    parser.add_argument("--input-243",type=Path)
    parser.add_argument("--input-244",type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"runs/mixing-reproduction")
    args=parser.parse_args()
    if args.input_244 and not args.input_242:
        parser.error("--input-244 also requires its --input-242 prerequisite")
    out=args.out.resolve()
    previous=args.input_242.resolve() if args.input_242 else out/"prerequisite242b"
    regional=args.input_243.resolve() if args.input_243 else out/"prerequisite243"
    native=args.input_244.resolve() if args.input_244 else out/"experiment244"
    final=out/"experiment245"
    steps=[]
    def step(name,module,*arguments):
        steps.append(dict(name=name,command=[sys.executable,"-m","boolean_coupling."+module,*map(str,arguments)]))
    if not args.input_242:
        steps.append(dict(name="through242b",command=[sys.executable,"research/boolean_coupling/reproduce_uniform.py","--run","--out",str(previous),*(["--offline"] if args.offline else [])]))
    if not args.input_244:
        if not args.input_243:
            step("regional243","regional243","--input-242",previous,"--out",regional)
            step("audit243","audit243b","--input-242",previous,"--run",regional)
        step("native244","native244","--input-242",previous,"--input-243",regional,"--out",native)
        step("audit244","audit244b","--input-242",previous,"--input-243",regional,"--run",native)
    step("mixing245","mixing245","--input-242",previous,"--input-244",native,"--out",final)
    step("audit245","audit245","--input-242",previous,"--input-244",native,"--run",final)
    step("origin246b","origin246b","--input-242",previous,"--input-244",native,"--out",out/"experiment246b")
    step("audit246b","audit246b","--input-242",previous,"--input-244",native,"--run",out/"experiment246b")
    if args.verify_only or args.history:
        verify(args.history)
    if args.plan:
        print(json.dumps(dict(steps=steps,scope="Known-context retrospective replication; FF is a replay control, not a forecast."),indent=2))
    if not args.run:
        if not (args.verify_only or args.history or args.plan):
            parser.print_help()
        return
    if out.exists():
        parser.error("--out must be new; never overwrite evidence")
    out.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/"research"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    record=dict(status="running",wrapper_sha256=sha(__file__),steps=steps,completed=[])
    for s in steps:
        try:
            with (out/(s["name"]+".log")).open("x") as log:
                subprocess.run(s["command"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        except Exception as exc:
            record.update(status="failed",failed_step=s["name"],error=repr(exc))
            (out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")
            raise
        record["completed"].append(s["name"])
        (out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")
    record["status"]="complete"
    (out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    main()
