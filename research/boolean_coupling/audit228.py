"""Independent NumPy qualification, sealed-order checks and frozen scoring for228."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")


def sources(out):
    design=json.loads((out/"design.json").read_text())
    assert all(sha(ROOT/p)==h for p,h in design["source_hashes"].items())
    return design


class Checks:
    def __init__(self):
        self.count=Counter()
        self.maximum=defaultdict(float)

    def check(self,name,error,bound):
        error=float(error)
        assert np.isfinite(error) and error<=bound,(name,error,bound)
        self.count[name]+=1
        self.maximum[name]=max(self.maximum[name],error)

    def record(self,items):
        for item in items:
            self.check("runner_"+item["name"],item["error"],item["bound"])


def finite_differences(data,checks):
    n=0
    for name in data:
        if name.endswith("_tangent"):
            tag=name.removesuffix("_tangent")
            fd=(data[tag+"_plus"]-data[tag+"_minus"])/2e-5
            checks.check("independent_fd",np.max(abs(fd-data[name])),1e-4)
            n+=1
    return n


def preflight(out):
    design=sources(out)
    target=out/"audit.json"
    assert not target.exists()
    status=json.loads((out/"status.json").read_text())
    assert status["status"]=="complete" and status["ordinary_false"]==1
    assert sha(out/"qualification.npz")==status["arrays_sha256"]
    checks=Checks()
    checks.record(status["checks"])
    n=finite_differences(dict(np.load(out/"qualification.npz")),checks)
    assert n==94
    write(target,dict(status="passed",source_hashes=design["source_hashes"],smooth_comparisons=n,
        checks=dict(checks.count),maximum_errors=dict(checks.maximum),model_calls=0,
        scope="Declared smoothCPU64 maps; no original Float32 whole-block recertification."))


def read_case(path):
    meta=json.loads(path.with_suffix(".json").read_text())
    assert sha(path)==meta["arrays_sha256"]
    return dict(np.load(path)),meta


def head(weights,x):
    x=x.astype(float)
    return (x/np.sqrt(np.mean(x*x)+float(weights["eps"]))*weights["norm"].astype(float))@weights["head"].astype(float).T


def parents(weights,mid):
    z=mid.astype(float)
    z=z/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+float(weights["eps24"]))*weights["norm24"].astype(float)
    return {name:z@weights[name+"24"].astype(float).T for name in ("gate","up")}


def qualification(out):
    design=sources(out)
    assert not (out/"forecast-qualification-seal.json").exists()
    status=json.loads((out/"forecast-status.json").read_text())
    assert status["status"]=="forecast_complete"
    assert status["counts"]=={**design["expected"],"ordinary_true":0}
    assert not any((out/"observations").iterdir())
    manifest=json.loads((out/"forecast-manifest.json").read_text())
    assert manifest["native_true_full_models_before_seal"]==0
    weights=dict(np.load(out/"weights.npz"))
    checks=Checks()
    catalog=set(json.loads((HERE/"historical-input-hashes-228.json").read_text())["hashes"])
    all_hashes=set()
    cases=json.loads((HERE/"cases228.json").read_text())
    files={name:sha(out/name) for name in ("design.json","runtime.json","forecast-status.json","forecast-manifest.json","weights.npz")}
    total_fd=0
    for case,entry in zip(cases,manifest["entries"]):
        assert case["key"]==entry["key"]
        fp=out/"forecasts"/(case["key"]+".npz")
        assert sha(fp)==entry["arrays_sha256"] and sha(fp.with_suffix(".json"))==entry["record_sha256"]
        f,meta=read_case(fp)
        files[str(fp.relative_to(out))]=sha(fp)
        files[str(fp.with_suffix(".json").relative_to(out))]=sha(fp.with_suffix(".json"))
        checks.record(meta["checks"])
        inputs=[]
        for bit in ("false","true"):
            h=hashlib.sha256(json.dumps(case[bit+"_ids"],separators=(",",":")).encode()).hexdigest()
            assert h not in catalog and h not in all_hashes
            all_hashes.add(h)
            inputs.append(h)
        assert inputs==case["input_hashes"]
        assert len(case["false_ids"])==len(case["true_ids"])
        changed=[i for i,(a,b) in enumerate(zip(case["false_ids"],case["true_ids"])) if a!=b]
        assert changed==[case["position"]]
        assert case["false_ids"][changed[0]]==895 and case["true_ids"][changed[0]]==830
        n=finite_differences(f,checks)
        assert n==135
        total_fd+=n
        for arm in ("L","F","R"):
            checks.check("initial_delta",np.max(abs(f[arm+"_delta_in"][0])),0.)
            checks.check("delta_chain",np.max(abs(f[arm+"_delta_in"][1:]-f[arm+"_delta_out"])),1e-9)
            for index in range(24):
                ja=f.get(f"qa{index}_{arm}_tangent",np.zeros_like(f[arm+"_delta_in"][index]))
                checks.check("attention_transport",np.max(abs(ja-f[arm+"_attn_jvp"][index])),1e-9)
                proposed=f["false_mids"][index].astype(float)+f["source_mid"][index]+ja
                checks.check("mid_update",np.max(abs(proposed-f[arm+"_pred_mid"][index])),1e-9)
                if index==23:
                    checks.check("target_cast",np.max(abs(proposed.astype(np.float32)-f[arm+"_target_mid"])),0.)
                    continue
                jm=f.get(f"qm{index}_{arm}_tangent",np.zeros_like(ja))
                linear=f["source_out"][index]+jm
                checks.check("linear_update",np.max(abs(linear-f[arm+"_linear_out"][index])),1e-9)
                expected=linear
                if index in (21,22) and arm in ("F","R"):
                    tag=f"b{index+1}_{arm}"
                    corr=f[tag+"_finite_delta"]-linear
                    checks.check("correction",np.max(abs(corr-f[tag+"_correction"])),1e-9)
                    checks.check("selected_cast",np.max(abs(proposed.astype(np.float32)-f[tag+"_installed_mid"])),0.)
                    if arm=="F":
                        expected=f[tag+"_finite_delta"]
                    else:
                        seed=int(hashlib.sha256((case["key"]+tag+"228").encode()).hexdigest()[:16],16)
                        random=np.random.default_rng(seed).standard_normal((1,*corr.shape))
                        random*=np.linalg.norm(corr)/np.linalg.norm(random)
                        checks.check("random_seed",np.max(abs(random[0]-f[tag+"_random"])),1e-9)
                        checks.check("random_norm",abs(np.linalg.norm(corr)-np.linalg.norm(f[tag+"_random"])),1e-9)
                        expected=linear+f[tag+"_random"]
                checks.check("output_update",np.max(abs(expected-f[arm+"_delta_out"][index])),1e-9)
            projected=parents(weights,f[arm+"_target_mid"])
            for name,reference in projected.items():
                checks.check("finite_"+name,np.max(abs(reference-f[arm+"_target_"+name])),1e-4)
            g=f[arm+"_target_gate"].astype(float)
            checks.check("finite_silu",np.max(abs(g/(1+np.exp(-g))-f[arm+"_target_phi"])),1e-4)
            checks.check("finite_head",np.max(abs(head(weights,f[arm+"_suffix"][-1,-1])-f[arm+"_scores"])),.005)
        checks.check("false_head",np.max(abs(head(weights,f["false_final"][-1])-f["false_scores"])),.005)
    assert len(cases)==32 and len(manifest["entries"])==32 and len(all_hashes)==64 and total_fd==4320
    result=dict(status="passed",source_hashes=design["source_hashes"],checks=dict(checks.count),
        maximum_errors=dict(checks.maximum),smooth_fd_comparisons=total_fd,fresh_input_hashes=len(all_hashes),
        native_true_full_models=0,model_calls=0,original_whole_block_jacobian_recertified=False)
    write(out/"forecast-audit.json",result)
    files["forecast-audit.json"]=sha(out/"forecast-audit.json")
    write(out/"forecast-qualification-seal.json",dict(files=files,sealed_monotonic_ns=time.monotonic_ns(),
        native_true_full_models=0,source_hashes=design["source_hashes"]))


def geometry(predicted,actual):
    p,t=predicted.astype(float).ravel(),actual.astype(float).ravel()
    pn,tn=np.linalg.norm(p),np.linalg.norm(t)
    cosine=float(p@t/(pn*tn)) if pn>0 and tn>0 else 0.
    relative=float(np.linalg.norm(p-t)/tn) if tn>0 else 1e300
    ratio=float(pn/tn) if tn>0 else 0.
    amplitude=(ratio-1)**2
    orientation=2*ratio*(1-cosine)
    assert abs(relative**2-amplitude-orientation)<=1e-8*max(1,relative**2)
    return dict(relative=relative,cosine=cosine,native_norm=float(tn),norm_ratio=ratio,
        amplitude_squared=amplitude,orientation_squared=orientation)


def score(out):
    design=sources(out)
    status=json.loads((out/"status.json").read_text())
    assert status["status"]=="complete" and status["counts"]==design["expected"]
    seal=json.loads((out/"forecast-qualification-seal.json").read_text())
    assert all(sha(out/p)==h for p,h in seal["files"].items())
    begin=json.loads((out/"observation-start.json").read_text())
    assert begin["started_monotonic_ns"]>seal["sealed_monotonic_ns"]
    weights=dict(np.load(out/"weights.npz"))
    checks=Checks()
    rows=[]
    layers=[]
    def margin(x):
        return float(np.float64(x[1])-np.float64(x[0]))
    for case in json.loads((HERE/"cases228.json").read_text()):
        f,fm=read_case(out/"forecasts"/(case["key"]+".npz"))
        o,om=read_case(out/"observations"/(case["key"]+".npz"))
        assert om["observed_monotonic_ns"]>begin["started_monotonic_ns"]
        assert om["forecast_sha256"]==fm["arrays_sha256"] and om["prefix_hashes"]==fm["prefix_hashes"]
        assert om["qualification_seal_sha256"]==sha(out/"forecast-qualification-seal.json")
        checks.record(om["checks"])
        checks.check("native_head",np.max(abs(head(weights,o["true_final"][-1])-o["true_scores"])),.005)
        for name,reference in parents(weights,o["true24_mid"]).items():
            checks.check("native_"+name,np.max(abs(reference-o["true24_"+name])),1e-4)
        actual=np.concatenate([o["true24_"+name][-1].astype(float)-f["false24_"+name][-1].astype(float) for name in ("gate","up")])
        actual_residual=o["true_final"][-1].astype(float)-f["false_final"][-1].astype(float)
        local=[]
        for arm in ("L","F","R"):
            pred=np.concatenate([f[arm+"_target_"+name][-1].astype(float)-f["false24_"+name][-1].astype(float) for name in ("gate","up")])
            g=geometry(pred,actual)
            predicted_residual=f[arm+"_suffix"][-1,-1].astype(float)-f["false_final"][-1].astype(float)
            residual=geometry(predicted_residual,actual_residual)
            row=dict(key=case["key"],**case["group"],arm=arm,
                **{"parent_"+k:v for k,v in g.items()},**{"residual_"+k:v for k,v in residual.items()},
                margin_error=abs(margin(f[arm+"_scores"])-margin(o["true_scores"])),
                predicted_margin=margin(f[arm+"_scores"]),native_margin=margin(o["true_scores"]))
            t=design["thresholds"]
            row["parent_bounds"]=(g["native_norm"]>=t["norm_guard"] and g["relative"]<=t["parent_relative"] and g["cosine"]>=t["parent_cosine"])
            row["native_bounds"]=(residual["native_norm"]>=t["norm_guard"] and residual["relative"]<=t["residual_relative"]
                and residual["cosine"]>=t["residual_cosine"] and row["margin_error"]<=t["margin"])
            local.append(row)
            for index in range(24):
                native_mid=o["true_mids"][index].astype(float)-f["false_mids"][index].astype(float)
                pred_mid=f[arm+"_pred_mid"][index]-f["false_mids"][index].astype(float)
                gg=geometry(pred_mid[-1],native_mid[-1])
                layers.append(dict(key=case["key"],**case["group"],arm=arm,block=index+1,**gg))
        for r in local:
            r["primary_pass"]=bool(r["arm"]=="F" and r["parent_bounds"] and
                all(r["parent_relative"]<s["parent_relative"] for s in local if s["arm"]!="F"))
        rows.extend(local)
    def whole(rr,key):
        strata={f"{schema}_{order}":sum(bool(r[key]) for r in rr if r["schema"]==schema and r["reverse"]==order)
            for schema in ("generic","descriptive") for order in (0,1)}
        total=sum(bool(r[key]) for r in rr)
        return dict(passes=total,n=len(rr),strata=strata,whole_pass=bool(total>=24 and min(strata.values())>=6))
    result=dict(experiment=228,status="complete",counts=status["counts"],
        source_hashes=design["source_hashes"],checks=dict(checks.count),maximum_errors=dict(checks.maximum),
        global_forecast_order_passed=True,original_whole_block_jacobian_recertified=False,
        primary=whole([r for r in rows if r["arm"]=="F"],"primary_pass"),arms=[])
    for arm in ("L","F","R"):
        rr=[r for r in rows if r["arm"]==arm]
        result["arms"].append(dict(arm=arm,parent_bounds=whole(rr,"parent_bounds"),native_bounds=whole(rr,"native_bounds"),
            **{name:float(np.median([r[name] for r in rr])) for name in
               ("parent_relative","parent_cosine","parent_norm_ratio","parent_amplitude_squared","parent_orientation_squared",
                "residual_relative","residual_cosine","margin_error")}))
    for filename,data in (("cases.csv",rows),("layerwise.csv",layers)):
        with (out/filename).open("w") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write(out/"audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("preflight","forecasts","score"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    {"preflight":preflight,"forecasts":qualification,"score":score}[args.phase](args.out.resolve())
