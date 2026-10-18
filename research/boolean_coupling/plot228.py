"""Plot all recorded228 cases and registered post-observation trajectories."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS={"L":"#777777","F":"#2166ac","R":"#d47a1f"}
LABELS={"L":"Linear","F":"Finite 22/23","R":"Random orientation"}


def main(run,destination):
    with (run/"cases.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    with (run/"layerwise.csv").open() as handle:
        layers=list(csv.DictReader(handle))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for ax,(field,title,bound,log) in zip(axes.flat,(
        ("parent_relative","MLP24 signed gate/up relative error",.25,True),
        ("parent_cosine","MLP24 signed gate/up cosine",.95,False),
        ("residual_relative","Native final residual relative error",.05,True),
        ("margin_error","Native final margin error (nats)",.10,True),
    )):
        for index,arm in enumerate(("L","F","R")):
            values=[float(r[field]) for r in rows if r["arm"]==arm]
            ax.scatter(index+np.linspace(-.13,.13,len(values)),values,color=COLORS[arm],alpha=.55,s=18)
            ax.plot([index-.2,index+.2],[np.median(values)]*2,color=COLORS[arm],lw=3)
        ax.set_xticks(range(3),[LABELS[a] for a in ("L","F","R")])
        ax.set_title(title)
        ax.axhline(bound,color="black",ls="--",lw=1)
        if log:
            ax.set_yscale("log")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("228: sealed False-only forecasts versus 32 native observations",fontsize=14)
    fig.supxlabel("All 32 cases shown; bars mark medians. Dashed lines are frozen bounds, not statistical confidence intervals.",fontsize=9)
    destination.mkdir(parents=True,exist_ok=True)
    for suffix in (".png",".svg"):
        fig.savefig(destination/("228-false-only-alignment"+suffix),dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax,field,title in zip(axes,("relative","cosine"),("Post-attention residual relative error","Post-attention residual direction cosine")):
        for arm in ("L","F","R"):
            medians=[float(np.median([float(r[field]) for r in layers if r["arm"]==arm and int(r["layer"])==block])) for block in range(1,25)]
            ax.plot(range(1,25),medians,color=COLORS[arm],label=LABELS[arm],lw=2)
        for block in (22,23):
            ax.axvline(block,color="#bbbbbb",ls=":",lw=1)
        ax.set(title=title,xlabel="Block (before its MLP)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Registered diagnostic after observation; no observed target states entered the forecasts")
    fig.supxlabel("Medians across 32 dependent contexts. Finite MLP22/23 updates affect the following block's midpoint.",fontsize=9)
    for suffix in (".png",".svg"):
        fig.savefig(destination/("228-layerwise-alignment"+suffix),dpi=150)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    main(args.run.resolve(),args.destination.resolve())
