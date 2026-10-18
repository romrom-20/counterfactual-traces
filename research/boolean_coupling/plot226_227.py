"""Plot recorded outcomes; no activations, model calls, or hidden filtering."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with path.open() as handle:
        return [r for r in csv.DictReader(handle) if r["schema"]=="descriptive"]


def points(ax, values, labels, colors):
    for index, (v, color) in enumerate(zip(values, colors)):
        jitter = np.linspace(-.11,.11,len(v))
        ax.scatter(index+jitter, v, color=color, alpha=.5, s=17, linewidths=0)
        ax.plot([index-.19,index+.19],[np.median(v)]*2,color=color,lw=3)
    ax.set_xticks(range(len(labels)), labels)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.15)


def main():
    tail = read(ROOT / "results/finite-tail-interaction-226b/cases.csv")
    primitive = read(ROOT / "results/mlp24-mixed-primitives-227b/cases.csv")
    fig, axes = plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    fig.suptitle("A downstream product interaction changes the margin; its vector account stays incomplete", fontsize=14)
    blue, gray, purple, orange = "#2166ac", "#888888", "#8064a2", "#d47a1f"
    values = [[float(r["remove_W"]) for r in tail if int(r["site"])==site and r["stage"]=="mlp"] for site in range(24,29)]
    points(axes[0],values,[f"MLP{s}" for s in range(24,29)],[blue]+[gray]*4)
    baseline=np.median([float(r["original_W"]) for r in primitive if r["component"]=="all"])
    axes[0].axhline(baseline,color="#333333",ls="--",lw=1,label="Original median gap")
    axes[0].set(title="226b: remove locally generated mixing",ylabel="Remaining margin interaction (nats)")
    axes[0].legend(frameon=False,fontsize=8)
    labels=["Original","Norm","SiLU","Product","All"]
    vals=[[float(r["original_W"]) for r in primitive if r["component"]=="all"]]
    vals += [[float(r["remove_W"]) for r in primitive if r["component"]==c] for c in ("N","S","P","all")]
    points(axes[1],vals,labels,[gray,gray,purple,orange,blue])
    axes[1].set(title="227b: remove a product-coordinate term",ylabel="Remaining margin interaction (nats)")
    vals=[[float(r["remove_vector_norm"])/float(r["original_vector_norm"]) for r in primitive if r["component"]==c] for c in ("N","S","P","all")]
    points(axes[2],vals,labels[1:],[gray,purple,orange,blue])
    axes[2].axhline(.75,color="#333333",ls="--",lw=1,label="Frozen vector criterion")
    axes[2].set(title="Only the combined removal passes both",ylabel="Final interaction norm / original")
    axes[2].legend(frameon=False,fontsize=8)
    fig.supxlabel("All 16 old descriptive contexts shown; bars mark medians. Synthetic dependent examples, not fresh forecasts.",fontsize=9)
    dest=ROOT / "docs/figures/226b-227b-downstream-interaction"
    fig.savefig(dest.with_suffix(".png"),dpi=160)
    fig.savefig(dest.with_suffix(".svg"))


if __name__=="__main__":
    main()
