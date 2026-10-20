"""Separate finite arithmetic and intervention audit at the selected block9."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import checked, sha
from .audit242b import decision, geometry, head
from .checkpoint238 import locate, weights
from .core225b import write_json

ROOT=Path(__file__).resolve().parents[2]
ARMS=("QQ","FQ","QF","FF")
EDITS=("D","P","G","ALL","gate","up","mix","random0","random1","random2")


def read(path):
    return json.loads(path.read_text())


def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(run,regional,previous):
    assert not (run/"audit.json").exists()
    design,status,seal=(read(run/n) for n in ("design.json","status.json","seal.json"))
    assert design["experiment"]==246 and design["retrospective"] and design["primary"]=="D_local_and_final"
    assert design["input244_audit_sha256"]==sha(regional/"audit.json") and design["input244_status_sha256"]==sha(regional/"status.json")
    assert seal["design_sha256"]==sha(run/"design.json") and seal["status_sha256"]==sha(run/"status.json")
    assert status["status"]=="complete" and seal["status"]=="sealed" and len(status["cases"])==32 and seal["cases"]==status["cases"]
    for name,h in design["source_hashes"].items():
        assert sha(ROOT/name)==sha(run/"sources"/Path(name).name)==h
    checkpoint,identity=locate(ROOT)
    assert identity==read(regional/"design.json")["checkpoint"]
    config=read(checkpoint/"config.json")
    with safe_open(checkpoint/"model.safetensors",framework="pt",device="cpu") as handle:
        w=weights(handle,config,8,"mlp")
    headw=dict(np.load(previous/"weights.npz"))
    regional_entries={e["key"]:e for e in read(regional/"status.json")["cases"]}
    first_entries={e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    checks,local,outcomes,primary=[],[],[],[]
    for case,ent in zip(read(run/"sources/cases242.json"),status["cases"],strict=True):
        key=case["key"]
        assert key==ent["key"]
        path=run/"cases"/(key+".npz")
        meta=read(path.with_suffix(".json"))
        assert sha(path)==meta["arrays_sha256"]==ent["arrays_sha256"] and sha(path.with_suffix(".json"))==ent["record_sha256"]
        assert meta["prior_regional_sha256"]==regional_entries[key]["arrays_sha256"]==sha(regional/"cases"/path.name)
        assert meta["prior_forecast_sha256"]==first_entries[key]["arrays_sha256"]==sha(previous/"forecasts"/path.name)
        for q in [dict(checks=meta["checks"]),*meta["qualifications"].values()]:
            for c in q["checks"]:
                assert np.isfinite(c["error"]) and c["error"]<=c["bound"]
        with np.load(path) as d,np.load(regional/"cases"/path.name) as r,np.load(previous/"forecasts"/path.name) as old:
            for a in (*ARMS,"finite0","finiteA","star"):
                g,u,phi,prod,write=(d[a+"_"+n].astype(float) for n in ("gate","up","phi","product","write"))
                checked(checks,"finite_product_bridge",phi*u,prod,1e-4)
                checked(checks,"finite_down_bridge",prod@w["down_weight"].T,write,1e-4)
                if a not in ("QQ","FQ"):
                    checked(checks,"finite_SiLU_primal",g/(1+np.exp(-g)),phi,1e-4)
                if a.startswith("finite") or a=="star":
                    x=d[a+"_mid"].astype(float)
                    z=x/np.sqrt(np.mean(x*x,-1,keepdims=True)+w["eps"])*w["norm"]
                    checked(checks,"finite_RMS_primal",z,d[a+"_normalized"],1e-4)
            prod={a:d[a+"_product"].astype(float) for a in ARMS}
            allmixed=prod["FF"]-prod["FQ"]-prod["QF"]+prod["QQ"]
            terms=dict(P=prod["FF"]-d["star_product"],G=d["star_product"].astype(float)-d["finiteA_product"]-prod["QF"]+d["finite0_product"],
                D=d["finiteA_product"].astype(float)-prod["FQ"]-d["finite0_product"]+prod["QQ"])
            checked(checks,"product_mixed_closure",sum(terms.values()),allmixed,1e-10)
            for name,value in terms.items():
                checked(checks,"component_reconstruction",value,d[name],0.)
            ra=d["finiteA_phi"].astype(float)-d["FQ_phi"]
            r0=d["finite0_phi"].astype(float)-d["QQ_phi"]
            du=d["FQ_up"].astype(float)-d["QQ_up"]
            primitives=dict(gate=(ra-r0)*d["QQ_up"],up=r0*du,mix=(ra-r0)*du)
            for name,value in primitives.items():
                checked(checks,"D_primitive_reconstruction",value,d[name],0.)
            rounding=terms["D"]-sum(primitives.values())
            checked(checks,"D_rounding_reconstruction",rounding,d["D_round"],1e-12)
            checked(checks,"D_rounding_bound",rounding,np.zeros_like(rounding),1e-4)
            def mixed(name):
                return d["FF_"+name].astype(float)-d["FQ_"+name]-d["QF_"+name]+d["QQ_"+name]
            ain,matt,mmlp,bout=(mixed(n) for n in ("input","attention","write","output"))
            blockterms=dict(inherited=ain,attention_P=d["attention_P"],attention_G=d["attention_G"],**{n:v@w["down_weight"].T for n,v in terms.items()})
            checked(checks,"attention_split",d["attention_P"]+d["attention_G"],matt,1e-10)
            blockterms["rounding"]=bout-sum(blockterms.values())
            checked(checks,"whole_block_rounding",blockterms["rounding"],np.zeros_like(bout),1e-4)
            valid=[]
            for space,target,components in (("product",allmixed,{**terms,**primitives}),("block",bout,blockterms)):
                for scope,pick in (("all",slice(None)),("verdict",-1)):
                    t=target[pick]
                    for name,value in components.items():
                        v=value[pick]
                        rel,cos=geometry(v,t)
                        local.append(dict(key=key,**case["group"],space=space,scope=scope,term=name,target_norm=float(np.linalg.norm(t)),relative_error=rel,cosine=cos,
                            relative_norm=float(np.linalg.norm(v)/max(np.linalg.norm(t),1e-30)),signed_share=float(np.vdot(v,t)/max(np.vdot(t,t),1e-30))))
                        if space=="product" and name=="D":
                            valid.append(np.linalg.norm(t)>=.001 and rel<=.10 and cos>=.99)
            finali=sum(sign*r[a+"_trajectory"][-1,1:].astype(float) for sign,a in ((1,"FF"),(-1,"FQ"),(-1,"QF"),(1,"QQ")))
            margins={a:float(np.diff(r[a+"_scores"].astype(float))[0]) for a in ARMS}
            mi=margins["FF"]-margins["FQ"]-margins["QF"]+margins["QQ"]
            records={}
            for edit in EDITS:
                term=allmixed if edit=="ALL" else primitives[edit] if edit in primitives else terms.get(edit)
                if edit.startswith("random"):
                    seed=int(hashlib.sha256((key+edit).encode()).hexdigest()[:16],16)
                    v=np.random.default_rng(seed).normal(size=(1,*terms["D"].shape))
                    norms=np.sqrt(np.sum(v*v,axis=(0,2),keepdims=True))
                    term=(v*np.linalg.norm(terms["D"],axis=-1)[None,:,None]/norms)[0]
                expected=(prod["FF"]-term).astype(np.float32)
                checked(checks,"direct_product_install",expected,d[edit+"_installed"],0.)
                final,scores=d[edit+"_final"].astype(float),d[edit+"_scores"].astype(float)
                checked(checks,"finite_head",head(headw,final[-1]),scores,.005)
                delta=final[1:]-r["FF_trajectory"][-1,1:].astype(float)
                margin=float(np.diff(scores)[0])
                remmargin=mi+margin-margins["FF"]
                row=dict(key=key,**case["group"],edit=edit,baseline_margin_interaction=mi,remaining_margin_interaction=remmargin,
                    margin_interaction_reduction=abs(mi)-abs(remmargin),native_margin_error=abs(margin-margins["FF"]))
                truth=r["FF_trajectory"][-1,1:].astype(float)-old["false_trajectory"][-1,1:].astype(float)
                for scope,pick in (("all",slice(None)),("verdict",-1)):
                    row[scope+"_remaining_ratio"]=float(np.linalg.norm((finali+delta)[pick])/max(np.linalg.norm(finali[pick]),1e-30))
                    row[scope+"_native_relative"]=geometry((final[1:]-old["false_trajectory"][-1,1:].astype(float))[pick],truth[pick])[0]
                outcomes.append(row)
                records[edit]=row
            passes=all(valid) and abs(mi)>=.10 and records["D"]["margin_interaction_reduction"]>=.10
            for scope in ("all","verdict"):
                passes=passes and records["D"][scope+"_remaining_ratio"]<=.75 and all(records["D"][scope+"_remaining_ratio"]<records[c][scope+"_remaining_ratio"] for c in ("random0","random1","random2"))
            primary.append(dict(key=key,**case["group"],passes=bool(passes),local_pass=bool(all(valid))))
        print("block9 finite audit",key[:8],flush=True)
    assert status["counts"]["regional_replays"]==128 and status["counts"]["primitive_intervention_forwards"]==320
    assert status["counts"]["local_attention_calls"]==32 and status["counts"]["local_MLP_calls"]==96
    for name,data in (("local-effects",local),("causal-outcomes",outcomes),("primary",primary)):
        table(run/(name+".csv"),data)
    write_json(run/"audit.json",dict(status="passed",retrospective=True,checks=checks,primary=decision(primary),
        local_D=decision([dict(r,passes=r["local_pass"]) for r in primary]),scorer_sha256=sha(__file__),design_sha256=sha(run/"design.json"),seal_sha256=sha(run/"seal.json"),
        csv_hashes={n:sha(run/(n+".csv")) for n in ("local-effects","causal-outcomes","primary")},interpretation="Selected-block finite policy-interaction diagnosis; no native False-only prediction or learned circuit."))
    print(json.dumps(decision(primary),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--input-244",type=Path,required=True)
    parser.add_argument("--input-242",type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args.run.resolve(),args.input_244.resolve(),args.input_242.resolve())
    except Exception as exc:
        write_json(args.run/"failure-audit.json",dict(error=repr(exc)))
        raise
