"""Posthoc246b signed summaries and plots, without model calls."""

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


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def med(data,key):
    return float(np.median([float(r[key]) for r in data]))


def main(run,figures):
    assert json.loads((run/"audit.json").read_text())["status"]=="passed"
    assert not (run/"description.json").exists()
    local,out=rows(run/"local-effects.csv"),rows(run/"causal-outcomes.csv")
    terms=("inherited","attention_P","attention_G","P","G","D","rounding")
    product=("P","G","D","gate","up","mix")
    edits=("D","P","G","ALL","gate","up","mix","random0","random1","random2")
    summary=dict(posthoc=True,model_calls=0,source_sha256=sha(__file__),audit_sha256=sha(run/"audit.json"),
        block={term:{k:med([r for r in local if r["space"]=="block" and r["scope"]=="all" and r["term"]==term],k) for k in ("target_norm","relative_norm","signed_share")} for term in terms},
        product={term:{k:med([r for r in local if r["space"]=="product" and r["scope"]=="all" and r["term"]==term],k) for k in ("relative_error","cosine","relative_norm","signed_share")} for term in product},
        final={edit:{k:med([r for r in out if r["edit"]==edit],k) for k in ("all_remaining_ratio","verdict_remaining_ratio","margin_interaction_reduction","all_native_relative","native_margin_error")} for edit in edits})
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.8),layout="constrained")
    for i,term in enumerate(terms[:-1]):
        sub=[r for r in local if r["space"]=="block" and r["scope"]=="all" and r["term"]==term]
        vals=[float(r["signed_share"]) for r in sub]
        axes[0].scatter(i+np.linspace(-.13,.13,len(vals)),vals,s=12,alpha=.6)
        axes[0].plot([i-.18,i+.18],[np.median(vals)]*2,lw=2)
    axes[0].set_xticks(range(6),["Incoming","Attn P","Attn G","MLP P","MLP G","SiLU D"],rotation=30,ha="right",fontsize=8)
    axes[0].axhline(0,color="#888",lw=.8)
    axes[0].set(title="Block9 interaction origin",ylabel="Signed projection / block interaction norm²")
    for i,edit in enumerate(edits[:7]):
        sub=[r for r in out if r["edit"]==edit]
        for ax,key in ((axes[1],"all_remaining_ratio"),(axes[2],"margin_interaction_reduction")):
            vals=[float(r[key]) for r in sub]
            ax.scatter(i+np.linspace(-.13,.13,len(vals)),vals,s=12,alpha=.6)
            ax.plot([i-.18,i+.18],[np.median(vals)]*2,lw=2)
    axes[1].axhline(1,color="#555",ls=":")
    axes[1].set(title="Direct product removals",ylabel="Final interaction norm / original norm")
    axes[2].axhline(0,color="#555",ls=":")
    axes[2].set(title="Finite margin effect",ylabel="Absolute interaction reduction (nats)")
    for ax in axes[1:]:
        ax.set_xticks(range(7),["D","P","G","All","Gate","Up","Mix"],rotation=30,ha="right",fontsize=8)
    fig.suptitle("246b: matched-input generation versus inherited mixed state")
    figures.mkdir(parents=True,exist_ok=True)
    for ext in ("png","svg"):
        fig.savefig(figures/("246b-block9-origin."+ext),dpi=180)
    plt.close(fig)
    write_json(run/"description.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True)
    args=parser.parse_args()
    main(args.run.resolve(),args.figures.resolve())
