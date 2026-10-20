"""Post-observation summaries and figures from separately audited244/245 CSVs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .audit229 import sha
from .core225b import write_json


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def med(data,key):
    return float(np.median([float(r[key]) for r in data]))


def main(native,mixing,figures):
    assert json.loads((native/"audit.json").read_text())["status"]=="passed"
    assert json.loads((mixing/"audit.json").read_text())["status"]=="passed"
    assert not (native/"description.json").exists() and not (mixing/"description.json").exists()
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    transfer=rows(native/"portability.csv")
    effects=rows(native/"signed-effects.csv")
    native_rows=rows(native/"native-outcomes.csv")
    summary=dict(posthoc=True,model_calls=0,source_sha256=sha(__file__),audit_sha256=sha(native/"audit.json"),
        portability={k:med(transfer,k) for k in ("all_relative","all_cosine","verdict_relative","verdict_cosine","margin_error")},
        native_medians={a:{k:med([r for r in native_rows if r["arm"]==a],k) for k in ("all_relative","verdict_relative","margin_error")} for a in ("QQ","FQ","QF","FF")},
        signed_medians={term:{k:med([r for r in effects if r["term"]==term and r["layer"]=="28" and r["scope"]=="all"],k) for k in ("relative_norm","signed_share")} for term in ("source_only","receiver_only","interaction")})
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    for i,(scope,color) in enumerate((("all","#217a76"),("verdict","#bd6e28"))):
        vals=np.array([float(r[scope+"_relative"])*100 for r in transfer])
        axs[0].scatter(i+np.linspace(-.12,.12,len(vals)),vals,color=color,alpha=.6,s=15)
        axs[0].plot([i-.2,i+.2],[np.median(vals)]*2,color=color,lw=3)
    axs[0].axhline(10,color="#555",ls=":")
    axs[0].set(xticks=[0,1],xticklabels=["All receivers","Verdict row"],ylabel="Interaction transfer error (%)",title="The old interaction does not transfer")
    for i,term in enumerate(("source_only","receiver_only","interaction")):
        vals=[float(r["relative_norm"]) for r in effects if r["term"]==term and r["layer"]=="28" and r["scope"]=="all"]
        axs[1].scatter(i+np.linspace(-.12,.12,len(vals)),vals,color=["#bd6e28","#4676bb","#217a76"][i],alpha=.6,s=15)
    axs[1].set(xticks=[0,1,2],xticklabels=["Source","Receivers","Interaction"],ylabel="Norm / full SiLU contrast norm",title="Native attention still has mixed effects")
    fig.suptitle("244: changing attention changes the quantitative interaction")
    for ext in ("png","svg"):
        fig.savefig(figures/("244-native-attention."+ext),dpi=180)
    plt.close(fig)
    write_json(native/"description.json",summary)
    local,outcomes=rows(mixing/"local-effects.csv"),rows(mixing/"causal-outcomes.csv")
    summary=dict(posthoc=True,model_calls=0,source_sha256=sha(__file__),audit_sha256=sha(mixing/"audit.json"),
        local={scope:{term:{k:med([r for r in local if r["scope"]==scope and r["term"]==term],k) for k in ("relative_norm","signed_share","relative_error","cosine","target_norm")} for term in ("QK","SM","PV","E")} for scope in ("all","verdict")},
        final={edit:{k:med([r for r in outcomes if r["edit"]==edit],k) for k in ("all_remaining_ratio","verdict_remaining_ratio","margin_interaction_reduction","all_native_relative","native_margin_error")} for edit in ("QK","SM","PV","ALL","random0","random1","random2")})
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout="constrained")
    for i,term in enumerate(("QK","SM","PV")):
        vals=[float(r["signed_share"]) for r in local if r["scope"]=="all" and r["term"]==term]
        axs[0].scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=14,alpha=.6)
    axs[0].axhline(0,color="#888",lw=.7)
    axs[0].set(xticks=[0,1,2],xticklabels=["QK mixing","Softmax","Weight × value"],ylabel="Signed projection / local mixed norm²",title="First local interaction")
    edits=("QK","SM","PV","ALL")
    for i,edit in enumerate(edits):
        sub=[r for r in outcomes if r["edit"]==edit]
        for ax,key in ((axs[1],"all_remaining_ratio"),(axs[2],"margin_interaction_reduction")):
            vals=[float(r[key]) for r in sub]
            ax.scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=14,alpha=.6)
            ax.plot([i-.2,i+.2],[np.median(vals)]*2,lw=2)
    axs[1].axhline(1,color="#555",ls=":")
    axs[1].set(ylabel="Final interaction norm / original norm",title="Remove that term at attention2")
    axs[2].axhline(0,color="#555",ls=":")
    axs[2].set(ylabel="Absolute margin interaction reduction (nats)",title="Finite readout effect")
    for ax in axs[1:]:
        ax.set_xticks(range(4),["QK","Softmax","W × V","All local"],fontsize=9)
    fig.suptitle("245: local creation and downstream importance are different tests")
    for ext in ("png","svg"):
        fig.savefig(figures/("245-first-mixing."+ext),dpi=180)
    plt.close(fig)
    write_json(mixing/"description.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-244",type=Path,required=True)
    parser.add_argument("--input-245",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True)
    args=parser.parse_args()
    main(args.input_244.resolve(),args.input_245.resolve(),args.figures.resolve())
