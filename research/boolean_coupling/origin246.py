"""Finite generation/propagation split and MLP9 product interventions."""

import argparse
import platform
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, capture_mlp, load_model, observe, write_json
from .mixing245 import SOURCES as PARENTS
from .mixing245 import random_rows
from .native244 import ARMS, bank_from, read, region, stamp
from .primitive242b import Taylor
from .reference242b import digest

ROOT=Path(__file__).resolve().parents[2]
EDITS=("D","P","G","ALL","gate","up","mix","random0","random1","random2")
SOURCES=[*PARENTS,*[ROOT/p for p in ("research/boolean_coupling/origin246.py","research/boolean_coupling/audit246.py","docs/experiments/246-block9-origin.md")]]
PROGRESS={}


def main(previous,regional,out):
    assert not out.exists()
    assert read(regional/"audit.json")["status"]=="passed"
    out.mkdir(parents=True)
    (out/"sources").mkdir()
    (out/"cases").mkdir()
    hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"design.json",dict(experiment=246,retrospective=True,posthoc_site_selection="block9 jump in244",primary="D_local_and_final",source_hashes=hashes,
        input244_audit_sha256=sha(regional/"audit.json"),input244_status_sha256=sha(regional/"status.json"),created_utc=stamp()))
    _,identity=locate(ROOT)
    assert identity==read(regional/"design.json")["checkpoint"]
    assert sha(Path(qwen.__file__))==read(regional/"design.json")["qwen_source_sha256"]
    model=load_model(True)
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__,device="mps",dtype="float32",coefficient_dtype="cpu_float64",revision=model.config._commit_hash,attention="eager",training=model.training)
    assert runtime==read(regional/"runtime.json")
    write_json(out/"runtime.json",runtime)
    layer=model.model.layers[8]
    counts,entries=Counter(),[]
    old_entries={e["key"]:e for e in read(regional/"status.json")["cases"]}
    for number,case in enumerate(read(ROOT/"research/boolean_coupling/cases242.json"),1):
        key,start=case["key"],case["position"]
        saved,checks,qual,inputs,mids={ },[],{},{},{}
        PROGRESS.update(key=key,data=saved,checks=checks,qualifications=qual)
        rp,fp=regional/"cases"/(key+".npz"),previous/"forecasts"/(key+".npz")
        assert sha(rp)==old_entries[key]["arrays_sha256"]
        meta=read(fp.with_suffix(".json"))
        with np.load(rp) as ref,np.load(fp) as old,torch.no_grad():
            for arm in ARMS:
                t=Taylor(bank_from(old,meta,arm))
                attn={}
                def save_attention(module,args,output):
                    attn["write"]=output[0].detach().clone()
                handle=layer.self_attn.register_forward_hook(save_attention)
                try:
                    with region(model,start,t,arm,counts,checks),capture_mlp(layer) as capture:
                        states,context,scores=observe(model,torch.tensor([case["true_ids"]],device="mps"))
                finally:
                    handle.remove()
                counts["regional_replays"]+=1
                for i,state in enumerate(states):
                    checked(checks,"tail_replay",array(state[0,start:]),ref[arm+"_trajectory"][i],0.)
                    assert digest(array(state[0,:start]))==meta["false_prefix_hashes"][i]
                checked(checks,"score_replay",array(scores),ref[arm+"_scores"],0.)
                inputs[arm]=states[7]
                mids[arm]=capture["mid"]
                for name,value in dict(input=states[7],attention=attn["write"],**capture).items():
                    saved[arm+"_"+name]=array(value[0,start+1:])
                saved[arm+"_output"]=array(states[8][0,start+1:])
                qual[arm]=dict(checks=t.reference.checks,witnesses=t.reference.witnesses,counts=dict(t.reference.counts))
            source=inputs["FQ"].double()+inputs["QF"].double()-inputs["QQ"].double()
            star=source.float()
            awrite=layer.self_attn(layer.input_layernorm(star),attention_mask=context["mask"],position_embeddings=context["rope"])[0]
            counts["local_attention_calls"]+=1
            saved["attention_star"]=array(awrite[0,start+1:])
            saved["attention_P"]=saved["FF_attention"].astype(float)-saved["attention_star"]
            saved["attention_G"]=saved["attention_star"].astype(float)-saved["FQ_attention"]-saved["QF_attention"]+saved["QQ_attention"]
            for arm,parent in (("finite0",mids["QQ"]),("finiteA",mids["FQ"]),("star",(mids["FQ"].double()+mids["QF"].double()-mids["QQ"].double()).float())):
                with capture_mlp(layer) as c:
                    z=layer.post_attention_layernorm(parent)
                    layer.mlp(z)
                for name,value in dict(normalized=z,**c).items():
                    saved[arm+"_"+name]=array(value[0,start+1:])
                counts["local_MLP_calls"]+=1
            p={a:saved[a+"_product"].astype(float) for a in ARMS}
            saved["P"]=p["FF"]-saved["star_product"]
            saved["G"]=saved["star_product"].astype(float)-saved["finiteA_product"]-p["QF"]+saved["finite0_product"]
            saved["D"]=saved["finiteA_product"].astype(float)-p["FQ"]-saved["finite0_product"]+p["QQ"]
            saved["ALL"]=p["FF"]-p["FQ"]-p["QF"]+p["QQ"]
            rA=saved["finiteA_phi"].astype(float)-saved["FQ_phi"]
            r0=saved["finite0_phi"].astype(float)-saved["QQ_phi"]
            du=saved["FQ_up"].astype(float)-saved["QQ_up"]
            saved["gate"]=(rA-r0)*saved["QQ_up"]
            saved["up"]=r0*du
            saved["mix"]=(rA-r0)*du
            saved["D_round"]=saved["D"]-saved["gate"]-saved["up"]-saved["mix"]
            checked(checks,"explicit_product_rounding",saved["D_round"],np.zeros_like(saved["D_round"]),1e-4)
            for edit in EDITS:
                term=saved[edit] if not edit.startswith("random") else random_rows(key,edit,saved["D"][None])[0]
                def intervention(module,args):
                    value=args[0].clone()
                    checked(checks,"native_product_replay",array(value[0,start+1:]),saved["FF_product"],0.)
                    proposed=(saved["FF_product"].astype(float)-term).astype(np.float32)
                    value[:,start+1:]=torch.tensor(proposed[None],device="mps")
                    checked(checks,"installed_product",array(value[0,start+1:]),proposed,0.)
                    checked(checks,"untouched_prefix_source",array(value[:,:start+1]),array(args[0][:,:start+1]),0.)
                    saved[edit+"_installed"]=array(value[0,start+1:])
                    return (value,)
                handle=layer.mlp.down_proj.register_forward_pre_hook(intervention)
                try:
                    states,_,scores=observe(model,torch.tensor([case["true_ids"]],device="mps"))
                finally:
                    handle.remove()
                counts["primitive_intervention_forwards"]+=1
                for i,state in enumerate(states):
                    assert digest(array(state[0,:start]))==meta["false_prefix_hashes"][i]
                saved[edit+"_final"],saved[edit+"_scores"]=array(states[-1][0,start:]),array(scores)
                assert np.isfinite(saved[edit+"_final"]).all()
        assert shutil.disk_usage(out).free>256*1024**2
        dest=out/"cases"/(key+".npz")
        np.savez_compressed(dest,**saved)
        write_json(dest.with_suffix(".json"),dict(key=key,group=case["group"],arrays_sha256=sha(dest),prior_regional_sha256=sha(rp),prior_forecast_sha256=sha(fp),checks=checks,qualifications=qual))
        entries.append(dict(key=key,arrays_sha256=sha(dest),record_sha256=sha(dest.with_suffix(".json"))))
        PROGRESS.clear()
        write_json(out/"progress.json",dict(cases=number,counts=dict(counts)))
        print(f"block9 origin {number}/32",flush=True)
    write_json(out/"status.json",dict(status="complete",cases=entries,counts=dict(counts)))
    write_json(out/"seal.json",dict(status="sealed",created_utc=stamp(),cases=entries,status_sha256=sha(out/"status.json"),design_sha256=sha(out/"design.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-242",type=Path,required=True)
    parser.add_argument("--input-244",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args.input_242.resolve(),args.input_244.resolve(),args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/"partial.npz",**PROGRESS["data"])
                write_json(args.out/"partial.json",{k:v for k,v in PROGRESS.items() if k!="data"})
            write_json(args.out/"failure.json",dict(error=repr(exc),recorded_utc=stamp()))
        raise
