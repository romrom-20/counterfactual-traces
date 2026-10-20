"""Separate NumPy finite-primitive and causal-outcome audit for245."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha, softmax
from .audit242b import decision, geometry, head
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("QQ", "FQ", "QF", "FF")
EDITS = ("QK", "SM", "PV", "ALL", "random0", "random1", "random2")


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(run, regional, previous):
    assert not (run / "audit.json").exists()
    design, status, seal = (read(run/n) for n in ("design.json", "status.json", "seal.json"))
    assert design["experiment"] == 245 and design["retrospective"] and design["primary"] == "PV_local_and_final"
    assert seal["status"] == "sealed" and status["status"] == "complete" and len(status["cases"]) == 32
    assert seal["status_sha256"] == sha(run/"status.json") and seal["design_sha256"] == sha(run/"design.json")
    assert seal["cases"] == status["cases"]
    for name,h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == h
    # Resolve inputs by supplied directories, rather than requiring original paths.
    inputs = [regional/n for n in ("design.json","status.json","audit.json")]+[previous/n for n in ("design.json","forecast-status.json","weights.npz")]
    assert list(design["inputs"].values()) == [sha(p) for p in inputs]
    weights = dict(np.load(previous/"weights.npz"))
    cases = read(run/"sources/cases242.json")
    regional_entries = {e["key"]:e for e in read(regional/"status.json")["cases"]}
    forecast_entries = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    checks, local, outcomes, primary = [], [], [], []
    for case,ent in zip(cases,status["cases"],strict=True):
        key = case["key"]
        assert key == ent["key"]
        path = run/"cases"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        assert sha(path)==meta["arrays_sha256"]==ent["arrays_sha256"] and sha(path.with_suffix(".json"))==ent["record_sha256"]
        assert meta["prior_regional_sha256"]==regional_entries[key]["arrays_sha256"]==sha(regional/"cases"/path.name)
        assert meta["prior_forecast_sha256"]==forecast_entries[key]["arrays_sha256"]==sha(previous/"forecasts"/path.name)
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"]<=c["bound"]
        for q in meta["qualifications"].values():
            assert len(q["witnesses"])==q["counts"].get("distinct_directions",0)
            for c in q["checks"]:
                assert np.isfinite(c["error"]) and c["error"]<=c["bound"]
        with np.load(path) as d, np.load(regional/"cases"/path.name) as r, np.load(previous/"forecasts"/path.name) as f:
            q,k,v,p,y = ({a:d[a+"_"+label].astype(float) for a in ARMS} for label in ("q","k","v","p","y"))
            for a in ARMS:
                groups = int(d[a+"_groups"])
                k[a],v[a] = (np.repeat(z,groups,axis=0) for z in (k[a],v[a]))
                logits = q[a]@k[a].swapaxes(-1,-2)*float(d[a+"_scaling"])
                checked(checks,"finite_QK_bridge",logits,d[a+"_raw"],1e-4)
                checked(checks,"finite_softmax_bridge",softmax(d[a+"_raw"].astype(float)+d[a+"_mask"].astype(float)),p[a],1e-4)
                checked(checks,"finite_PV_bridge",p[a]@v[a],y[a],1e-4)
            for states in (q,k,v):
                mixed = states["FF"]-states["FQ"]-states["QF"]+states["QQ"]
                checked(checks,"zero_QKV_mixed",mixed,np.zeros_like(mixed),0.)
            lmix = d["FF_raw"].astype(float)-d["FQ_raw"].astype(float)-d["QF_raw"].astype(float)+d["QQ_raw"].astype(float)
            qk = (q["QF"]-q["QQ"])@(k["FQ"]-k["QQ"]).swapaxes(-1,-2)*float(d["FF_scaling"])
            checked(checks,"analytic_mixed_QK_bridge",qk,lmix,1e-4)
            rawstar = (d["FQ_raw"].astype(float)+d["QF_raw"].astype(float)-d["QQ_raw"].astype(float)).astype(np.float32)
            checked(checks,"finite_star_softmax",softmax(rawstar.astype(float)+d["FF_mask"].astype(float)),d["pstar"],1e-4)
            terms = dict(QK=(p["FF"]-d["pstar"])@v["FF"],SM=(d["pstar"]-p["FQ"]-p["QF"]+p["QQ"])@v["FF"],
                PV=(p["FQ"]-p["QQ"])@(v["QF"]-v["QQ"])+(p["QF"]-p["QQ"])@(v["FQ"]-v["QQ"]))
            target = y["FF"]-y["FQ"]-y["QF"]+y["QQ"]
            checked(checks,"IY_exact",d["IY"],target,0.)
            for name,value in terms.items():
                checked(checks,"saved_term_"+name,d[name],value,1e-12)
            remainder = target-sum(terms.values())
            checked(checks,"saved_rounding_remainder",d["E"],remainder,1e-12)
            checked(checks,"rounding_bound",remainder,np.zeros_like(remainder),1e-4)
            terms["E"] = remainder
            local_valid=[]
            for scope,pick in (("all",slice(None)),("verdict",-1)):
                t=target[:,pick]
                for name,term in terms.items():
                    value=term[:,pick]
                    rel,cos=geometry(value,t)
                    local.append(dict(key=key,**case["group"],scope=scope,term=name,target_norm=float(np.linalg.norm(t)),relative_error=rel,cosine=cos,
                        relative_norm=float(np.linalg.norm(value)/max(np.linalg.norm(t),1e-30)),signed_share=float(np.vdot(value,t)/max(np.vdot(t,t),1e-30))))
                    if name=="PV":
                        local_valid.append(np.linalg.norm(t)>=.001 and rel<=.10 and cos>=.99)
            final_i=sum(sign*r[a+"_trajectory"][-1,1:].astype(float) for sign,a in ((1,"FF"),(-1,"FQ"),(-1,"QF"),(1,"QQ")))
            margins={a:float(np.diff(r[a+"_scores"].astype(float))[0]) for a in ARMS}
            mi=margins["FF"]-margins["FQ"]-margins["QF"]+margins["QQ"]
            rows={}
            for edit in EDITS:
                if edit in ("QK","SM"):
                    prob=d["pstar"] if edit=="QK" else p["FF"]-(d["pstar"]-p["FQ"]-p["QF"]+p["QQ"])
                    expected=prob.astype(np.float32).astype(float)@v["FF"]
                else:
                    term=target if edit=="ALL" else terms["PV"]
                    if edit.startswith("random"):
                        seed=int(hashlib.sha256((key+edit).encode()).hexdigest()[:16],16)
                        value=np.random.default_rng(seed).normal(size=term.shape)
                        def norm(a):
                            return np.sqrt(np.sum(a*a,axis=(0,2),keepdims=True))
                        term=value*norm(term)/norm(value)
                    expected=(y["FF"]-term).astype(np.float32)
                checked(checks,"direct_primitive_installation_"+edit,expected,d[edit+"_installed_y"],1e-4)
                final,scores=d[edit+"_final"].astype(float),d[edit+"_scores"].astype(float)
                checked(checks,"finite_final_head",head(weights,final[-1]),scores,.005)
                changed=final[1:]-r["FF_trajectory"][-1,1:].astype(float)
                remaining=final_i+changed
                margin=float(scores[1]-scores[0])
                rem_margin=mi+margin-margins["FF"]
                row=dict(key=key,**case["group"],edit=edit,baseline_margin_interaction=mi,remaining_margin_interaction=rem_margin,
                    margin_interaction_reduction=abs(mi)-abs(rem_margin),native_margin_error=abs(margin-margins["FF"]))
                for scope,pick in (("all",slice(None)),("verdict",-1)):
                    row[scope+"_remaining_ratio"]=float(np.linalg.norm(remaining[pick])/max(np.linalg.norm(final_i[pick]),1e-30))
                    truth=r["FF_trajectory"][-1,1:].astype(float)-f["false_trajectory"][-1,1:].astype(float)
                    prediction=final[1:]-f["false_trajectory"][-1,1:].astype(float)
                    rel,cos=geometry(prediction[pick],truth[pick])
                    row[scope+"_native_relative"],row[scope+"_native_cosine"]=rel,cos
                outcomes.append(row)
                rows[edit]=row
            pv=rows["PV"]
            passes=all(local_valid) and abs(mi)>=.10 and pv["margin_interaction_reduction"]>=.10
            for scope in ("all","verdict"):
                passes=passes and pv[scope+"_remaining_ratio"]<=.75 and all(pv[scope+"_remaining_ratio"]<rows[c][scope+"_remaining_ratio"] for c in ("random0","random1","random2"))
            primary.append(dict(key=key,**case["group"],passes=bool(passes),local_pass=bool(all(local_valid))))
        print("finite first-mixing audit",key[:8],flush=True)
    assert status["counts"]["native_replays"]==32 and status["counts"]["regional_replays"]==96 and status["counts"]["primitive_intervention_forwards"]==224
    for name,rows in (("local-effects",local),("causal-outcomes",outcomes),("primary",primary)):
        table(run/(name+".csv"),rows)
    write_json(run/"audit.json",dict(status="passed",retrospective=True,checks=checks,primary=decision(primary),
        local_PV=decision([dict(r,passes=r["local_pass"]) for r in primary]),scorer_sha256=sha(__file__),design_sha256=sha(run/"design.json"),seal_sha256=sha(run/"seal.json"),
        csv_hashes={n:sha(run/(n+".csv")) for n in ("local-effects","causal-outcomes","primary")},
        interpretation="Finite first regional interaction and oracle primitive removals, not a prospective native-response predictor or learned circuit."))
    print(json.dumps(decision(primary),indent=2))


if __name__ == "__main__":
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
