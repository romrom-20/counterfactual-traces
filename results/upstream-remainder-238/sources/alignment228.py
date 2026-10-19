"""Preflight and globally sealed False-only forecasts of native receiving alignment."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from .core225b import HERE, array, capture_mlp, check, load_model, observe, sha, write_json
from .smooth228 import attention_map, mlp_map, promote, qualified_jvp
from .tail226b import block

ROOT=HERE.parents[1]
ARMS=("L","F","R")
SOURCES=[HERE/n for n in ("alignment228.py","audit228.py","smooth228.py","fixtures228.py",
    "cases228.json","historical-input-hashes-228.json","core225b.py","tail226b.py","settings.json","cases.json")]
SOURCES+=[ROOT/"docs/experiments/228-false-only-alignment.md"]
EXPECTED=dict(ordinary_false=32,ordinary_true=32,source_blocks=768,smooth_jvps=4320,
    fd_comparisons=4320,zero_skips=192,selected_mlps=128,target_mlps=96,suffix_blocks=384,forecast_heads=96)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def tensor_hash(x):
    return hashlib.sha256(array(x).tobytes()).hexdigest()


@contextmanager
def mids(model):
    values,handles=[],[]
    def save(module,args):
        values.append(args[0].detach().clone())
    for layer in model.model.layers[:24]:
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(save))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def source_block(model,index,operand,context):
    captured=[]
    def before(module,args):
        captured.append(args[0].detach().clone())
    handle=model.model.layers[index].post_attention_layernorm.register_forward_pre_hook(before)
    try:
        with torch.no_grad():
            output=block(model,index,operand,context)
    finally:
        handle.remove()
    assert len(captured)==1
    return output,captured[0]


def save_case(folder,case,data,checks,extra=None):
    path=folder/(case["key"]+".npz")
    np.savez_compressed(path,**data)
    record=dict(key=case["key"],group=case["group"],position=case["position"],arrays_sha256=sha(path),
                checks=checks,recorded_utc=stamp(),**(extra or {}))
    write_json(path.with_suffix(".json"),record)
    return dict(key=case["key"],arrays_sha256=sha(path),record_sha256=sha(path.with_suffix(".json")))


def checked_direction(function,anchor,direction,label,data,checks,counts):
    if not torch.count_nonzero(direction):
        counts["zero_skips"]+=1
        return torch.zeros_like(direction)
    _,tangent,plus,minus,error=qualified_jvp(function,anchor,direction)
    counts["smooth_jvps"]+=1
    counts["fd_comparisons"]+=1
    check(checks,"smooth_fd",error)
    for name,value in (("direction",direction),("tangent",tangent),("plus",plus),("minus",minus)):
        data[label+"_"+name]=array(value[0])
    return tangent


def initialize(out,phase,preflight=None):
    assert not out.exists(),"Never overwrite evidence."
    out.mkdir(parents=True)
    (out/"sources").mkdir()
    hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    if phase=="forecast":
        assert preflight is not None
        audit=json.loads((preflight/"audit.json").read_text())
        assert audit["status"]=="passed" and audit["source_hashes"]==hashes
    write_json(out/"design.json",dict(experiment=228,phase=phase,source_hashes=hashes,
        expected=EXPECTED,created_utc=stamp(),preflight=str(preflight) if preflight else None,
        preflight_audit_sha256=sha(preflight/"audit.json") if preflight else None,
        thresholds=dict(parent_relative=.25,parent_cosine=.95,residual_relative=.05,residual_cosine=.99,
            margin=.10,norm_guard=.001,total_passes=24,stratum_passes=6),
        transport="Reanchored smoothCPU64 receiving derivatives; public MPS finite source, selected MLPs and tail.",
        original_whole_block_jacobian_recertified=False))


def runtime(out,model):
    write_json(out/"runtime.json",dict(python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,transformers=transformers.__version__,device="mps",dtype="float32",
        transport_dtype="cpu_float64",attention="eager",revision=model.config._commit_hash))


def preflight(args):
    initialize(args.out,"preflight")
    model=load_model(True)
    runtime(args.out,model)
    case=json.loads((HERE/"cases.json").read_text())[0]
    start=case["position"]+1
    ids=torch.tensor([case["false_ids"]],device="mps")
    with mids(model) as middle:
        states,context,_=observe(model,ids)
    with torch.no_grad():
        embedding=model.model.embed_tokens(ids)
    boundaries=[embedding,*states]
    data,checks={},[]
    comparisons=0
    for index in range(24):
        maps=[("attention",attention_map(model.model.layers[index],boundaries[index],context,start),
               promote(boundaries[index][:,start:]),promote(middle[index][:,start:]))]
        if index<23:
            maps.append(("mlp",mlp_map(model.model.layers[index]),promote(middle[index][:,start:]),
                         promote(boundaries[index+1][:,start:])))
        for name,function,anchor,target in maps:
            check(checks,"promoted_primal",(function(anchor)-target).abs().max())
            for kind in (0,1):
                grid=torch.arange(anchor.numel(),dtype=torch.float64).reshape_as(anchor)
                direction=(torch.sin if kind==0 else torch.cos)(grid*.013+index*.29)
                direction*=.05*torch.linalg.vector_norm(anchor)/torch.linalg.vector_norm(direction)
                _,tangent,plus,minus,error=qualified_jvp(function,anchor,direction)
                check(checks,"preflight_fd",error)
                label=f"b{index+1}_{name}_{kind}"
                for key,value in (("tangent",tangent),("plus",plus),("minus",minus)):
                    data[label+"_"+key]=array(value[0])
                comparisons+=1
        print("preflight",index+1,flush=True)
    assert comparisons==94 and len(checks)==141
    np.savez_compressed(args.out/"qualification.npz",**data)
    write_json(args.out/"status.json",dict(status="complete",ordinary_false=1,smooth_comparisons=94,
        primal_comparisons=47,arrays_sha256=sha(args.out/"qualification.npz"),checks=checks))


def forecast(args):
    initialize(args.out,"forecast",args.preflight)
    (args.out/"forecasts").mkdir()
    (args.out/"observations").mkdir()
    model=load_model(True)
    runtime(args.out,model)
    cases=json.loads((HERE/"cases228.json").read_text())
    counts=Counter({k:0 for k in EXPECTED})
    entries=[]
    layer24=model.model.layers[23]
    np.savez_compressed(args.out/"weights.npz",norm=array(model.model.norm.weight),
        head=array(model.lm_head.weight[[895,830]]),eps=np.array(model.model.norm.variance_epsilon),
        norm24=array(layer24.post_attention_layernorm.weight),
        eps24=np.array(layer24.post_attention_layernorm.variance_epsilon),
        gate24=array(layer24.mlp.gate_proj.weight),up24=array(layer24.mlp.up_proj.weight))
    for number,case in enumerate(cases,1):
        start=case["position"]+1
        ids=torch.tensor([case["false_ids"]],device="mps")
        with mids(model) as middle,capture_mlp(layer24) as false24:
            states,context,false_scores=observe(model,ids)
        counts["ordinary_false"]+=1
        with torch.no_grad():
            embedding=model.model.embed_tokens(ids)
            true_embedding=model.model.embed_tokens(torch.tensor([case["true_ids"]],device="mps"))
        boundaries=[embedding,*states]
        checks,data=[],dict(false_final=array(states[-1][0,start:]),false_scores=array(false_scores),
            false_boundaries=array(torch.stack(boundaries)[:,0,start:]),
            false_mids=array(torch.stack(middle)[:,0,start:]))
        for name in ("mid","gate","phi","up"):
            data["false24_"+name]=array(false24[name][0,start:])
        prefix_hashes=[tensor_hash(s[0,:start-1]) for s in states]
        source=true_embedding[:,start-1].clone()
        deltas={arm:torch.zeros_like(promote(embedding[:,start:])) for arm in ARMS}
        trace={arm:{k:[] for k in ("delta_in","attn_jvp","pred_mid","linear_out","delta_out")} for arm in ARMS}
        source_out,source_mid,source_rows=[],[],[]
        for index in range(24):
            layer=model.model.layers[index]
            operand=boundaries[index].clone()
            operand[:,start-1]=source
            finite,mid=source_block(model,index,operand,context)
            counts["source_blocks"]+=1
            check(checks,"source_prefix",(finite[:,:start-1]-boundaries[index+1][:,:start-1]).abs().max(),0.)
            source=finite[:,start-1].detach().clone()
            source_rows.append(array(source[0]))
            sd=promote(finite[:,start:])-promote(boundaries[index+1][:,start:])
            source_out.append(array(sd[0]))
            source_mid.append(array((promote(mid[:,start:])-promote(middle[index][:,start:]))[0]))
            attention=attention_map(layer,boundaries[index],context,start)
            fa=promote(boundaries[index][:,start:])
            check(checks,"attention_primal",(attention(fa)-promote(middle[index][:,start:])).abs().max())
            mlp=mlp_map(layer) if index<23 else None
            fm=promote(middle[index][:,start:])
            if mlp:
                check(checks,"mlp_primal",(mlp(fm)-promote(boundaries[index+1][:,start:])).abs().max())
            for arm in ARMS:
                delta=deltas[arm]
                trace[arm]["delta_in"].append(array(delta[0]))
                ja=checked_direction(attention,fa,delta,f"qa{index}_{arm}",data,checks,counts)
                trace[arm]["attn_jvp"].append(array(ja[0]))
                proposed=promote(mid[:,start:])+ja
                trace[arm]["pred_mid"].append(array(proposed[0]))
                if index==23:
                    full_mid=mid.clone()
                    full_mid[:,start:]=proposed.to(device="mps",dtype=torch.float32)
                    check(checks,"installed_prefix",(full_mid[:,:start]-mid[:,:start]).abs().max(),0.)
                    with capture_mlp(layer) as cap,torch.no_grad():
                        state=full_mid+layer.mlp(layer.post_attention_layernorm(full_mid))
                    counts["target_mlps"]+=1
                    check(checks,"target_source",(state[:,start-1]-finite[:,start-1]).abs().max(),0.)
                    check(checks,"tail_prefix",(state[:,:start-1]-boundaries[24][:,:start-1]).abs().max(),0.)
                    for name in ("mid","gate","phi","up"):
                        data[arm+"_target_"+name]=array(cap[name][0,start:])
                    suffix=[array(state[0,start:])]
                    with torch.no_grad():
                        for following in range(24,28):
                            state=block(model,following,state,context)
                            counts["suffix_blocks"]+=1
                            check(checks,"tail_prefix",(state[:,:start-1]-boundaries[following+1][:,:start-1]).abs().max(),0.)
                            suffix.append(array(state[0,start:]))
                        scores=torch.nn.functional.linear(model.model.norm(state)[0,-1],model.lm_head.weight[[895,830]])
                    counts["forecast_heads"]+=1
                    data[arm+"_suffix"]=np.stack(suffix)
                    data[arm+"_scores"]=array(scores)
                    continue
                jm=checked_direction(mlp,fm,ja,f"qm{index}_{arm}",data,checks,counts)
                linear=sd+jm
                trace[arm]["linear_out"].append(array(linear[0]))
                output=linear
                if index in (21,22) and arm in ("F","R"):
                    full_mid=mid.clone()
                    full_mid[:,start:]=proposed.to(device="mps",dtype=torch.float32)
                    check(checks,"installed_prefix",(full_mid[:,:start]-mid[:,:start]).abs().max(),0.)
                    with torch.no_grad():
                        actual=full_mid+layer.mlp(layer.post_attention_layernorm(full_mid))
                    counts["selected_mlps"]+=1
                    finite_delta=promote(actual[:,start:])-promote(boundaries[index+1][:,start:])
                    correction=finite_delta-linear
                    tag=f"b{index+1}_{arm}"
                    data[tag+"_finite_delta"]=array(finite_delta[0])
                    data[tag+"_correction"]=array(correction[0])
                    data[tag+"_installed_mid"]=array(full_mid[0,start:])
                    if arm=="F":
                        output=finite_delta
                    else:
                        seed=int(hashlib.sha256((case["key"]+tag+"228").encode()).hexdigest()[:16],16)
                        random=np.random.default_rng(seed).standard_normal(tuple(correction.shape))
                        random*=float(torch.linalg.vector_norm(correction))/np.linalg.norm(random)
                        check(checks,"random_norm",abs(np.linalg.norm(random)-float(torch.linalg.vector_norm(correction))),1e-9)
                        data[tag+"_random"]=random[0]
                        output=linear+torch.tensor(random,dtype=torch.float64)
                trace[arm]["delta_out"].append(array(output[0]))
                deltas[arm]=output
            print("forecast",number,"block",index+1,flush=True) if index==23 else None
            del mlp,attention
        for arm in ARMS:
            for name,value in trace[arm].items():
                data[arm+"_"+name]=np.stack(value)
        data.update(source_out=np.stack(source_out),source_mid=np.stack(source_mid),source_rows=np.stack(source_rows))
        entries.append(save_case(args.out/"forecasts",case,data,checks,dict(prefix_hashes=prefix_hashes)))
        write_json(args.out/"status.json",dict(status="forecasting",completed=number,counts=dict(counts)))
    expected={**EXPECTED,"ordinary_true":0}
    assert dict(counts)==expected,(dict(counts),expected)
    assert not any((args.out/"observations").iterdir())
    write_json(args.out/"forecast-manifest.json",dict(entries=entries,sealed_utc=stamp(),sealed_monotonic_ns=time.monotonic_ns(),
        native_true_full_models_before_seal=0,source_hashes=json.loads((args.out/"design.json").read_text())["source_hashes"],
        counts=dict(counts)))
    write_json(args.out/"status.json",dict(status="forecast_complete",completed=32,counts=dict(counts)))
    write_json(args.out/"forecast-status.json",dict(status="forecast_complete",completed=32,counts=dict(counts)))


def observe_phase(args):
    design=json.loads((args.out/"design.json").read_text())
    assert all(sha(ROOT/p)==h for p,h in design["source_hashes"].items())
    seal=json.loads((args.out/"forecast-qualification-seal.json").read_text())
    assert all(sha(args.out/p)==h for p,h in seal["files"].items())
    assert not any((args.out/"observations").iterdir())
    counts=Counter(json.loads((args.out/"status.json").read_text())["counts"])
    manifest=json.loads((args.out/"forecast-manifest.json").read_text())
    assert counts["ordinary_true"]==0 and len(manifest["entries"])==32
    write_json(args.out/"observation-start.json",dict(started_utc=stamp(),started_monotonic_ns=time.monotonic_ns(),
        qualification_seal_sha256=sha(args.out/"forecast-qualification-seal.json")))
    model=load_model(True)
    for number,case in enumerate(json.loads((HERE/"cases228.json").read_text()),1):
        entry=manifest["entries"][number-1]
        path=args.out/"forecasts"/(case["key"]+".npz")
        assert entry["key"]==case["key"] and sha(path)==entry["arrays_sha256"]
        f=dict(np.load(path))
        meta=json.loads(path.with_suffix(".json").read_text())
        start=case["position"]+1
        with mids(model) as middle,capture_mlp(model.model.layers[23]) as cap:
            states,_,scores=observe(model,torch.tensor([case["true_ids"]],device="mps"))
        counts["ordinary_true"]+=1
        checks=[]
        prefix_hashes=[tensor_hash(s[0,:start-1]) for s in states]
        check(checks,"native_prefix_hash",0. if prefix_hashes==meta["prefix_hashes"] else float("inf"),0.)
        check(checks,"known_source",np.max(abs(array(torch.stack(states[:24])[:,0,start-1])-f["source_rows"])))
        data=dict(true_final=array(states[-1][0,start:]),true_scores=array(scores),
            true_states=array(torch.stack(states)[:,0,start:]),true_mids=array(torch.stack(middle)[:,0,start:]))
        for name in ("mid","gate","phi","up"):
            data["true24_"+name]=array(cap[name][0,start:])
        save_case(args.out/"observations",case,data,checks,dict(forecast_sha256=entry["arrays_sha256"],
            qualification_seal_sha256=sha(args.out/"forecast-qualification-seal.json"),prefix_hashes=prefix_hashes,
            observed_monotonic_ns=time.monotonic_ns()))
        write_json(args.out/"status.json",dict(status="observing",completed=number,counts=dict(counts)))
        print("observe",number,flush=True)
    assert dict(counts)==EXPECTED
    assert all(sha(ROOT/p)==h for p,h in design["source_hashes"].items())
    assert all(sha(args.out/p)==h for p,h in seal["files"].items())
    write_json(args.out/"status.json",dict(status="complete",completed=32,counts=dict(counts)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("preflight","forecast","observe"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--preflight",type=Path)
    args=parser.parse_args()
    args.out=args.out.resolve()
    if args.preflight:
        args.preflight=args.preflight.resolve()
    try:
        {"preflight":preflight,"forecast":forecast,"observe":observe_phase}[args.phase](args)
    except Exception as exc:
        if args.out.exists():
            write_json(args.out/("failure-"+args.phase+".json"),dict(error=repr(exc),recorded_utc=stamp()))
        raise
