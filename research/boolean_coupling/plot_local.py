"""Figures of all local primitive and prospective forecast cases; no model calls."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def diagnostic(run, destination):
    with (run/"cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    description = json.loads((run/"composite-description.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    colors = {"generic":"#087d8c", "descriptive":"#bc663e"}
    rng = np.random.default_rng(230)
    for i, name in enumerate(("N", "K", "S", "V")):
        values = []
        for r in rows:
            v = float(r[name+"_relative"])
            values.append(v)
            axes[0].scatter(i+rng.uniform(-.12, .12), v, color=colors[r["schema"]], s=18, alpha=.7)
        axes[0].plot([i-.2, i+.2], [np.median(values)]*2, color="black")
    axes[0].axhline(.25, color="gray", ls="--", lw=.8)
    axes[0].set(xticks=range(4), xticklabels=("RMSNorm", "QK", "softmax", "weight × value"),
                ylabel="Relative error in receiver remainder R", title="Registered: no single primitive passes")
    codes = ("1000", "1001", "1010", "1011", "1111")
    for i, code in enumerate(codes):
        axes[1].bar(i, description["composites"][code]["median_relative"], color="#877aa0" if i < 4 else "#4b8482")
    axes[1].axhline(.25, color="gray", ls="--", lw=.8)
    axes[1].set(yscale="log", xticks=range(5), xticklabels=("N", "N+V", "N+S", "N+S+V", "all"),
                ylabel="Median relative remainder error (log scale)", title="Posthoc: the joint operation matters")
    for i, name in enumerate(("query", "key", "value")):
        values = []
        for r in rows:
            v = float(r["first_"+name+"_share"])
            values.append(v)
            axes[2].scatter(i+rng.uniform(-.12, .12), v, color=colors[r["schema"]], s=18, alpha=.7)
        axes[2].plot([i-.2, i+.2], [np.median(values)]*2, color="black")
    axes[2].axhline(0, color="gray", lw=.7)
    axes[2].set(xticks=range(3), xticklabels=("query", "key", "value"), ylabel="Signed projection onto R at verdict",
                title="First-row normalization travels through K/V")
    fig.suptitle("230 · Exact incoming state; 32 observed template contexts", fontsize=13)
    fig.text(.01, -.045, "Teal: generic; orange: descriptive. N/K/S/V are finite/False-affine node switches. "
             "The N+V composite was selected after observing these data.\n"
             "First-row paths describe the normalization-only correction, not the full R. "
             "Signed projections are not causal percentages or learned features.", fontsize=9)
    save(fig, destination)


def forecast(run, destination):
    with (run/"cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    colors = {"generic":"#087d8c", "descriptive":"#bc663e"}
    rng = np.random.default_rng(231)
    for ax, scope in zip(axes, ("verdict", "all")):
        for i, arm in enumerate(("L", "N", "NV", "R")):
            vals = []
            for row in rows:
                v = float(row[scope+"_"+arm+"_relative"])
                vals.append(v)
                ax.scatter(i+rng.uniform(-.12, .12), v, color=colors[row["schema"]], s=25, alpha=.8)
            ax.plot([i-.2, i+.2], [np.median(vals)]*2, color="black")
        ax.axhline(.025, color="gray", lw=.8, ls="--")
        ax.set(xticks=range(4), xticklabels=("linear", "norm", "norm+product", "matched random"),
               ylabel="Relative native displacement error", title="Verdict position" if scope == "verdict" else "All receiving rows (stronger criterion)")
    fig.suptitle("231 · Forecasts sealed before 16 new native True runs · attention 2 only", fontsize=13)
    fig.text(.01, -.045, "Teal: generic; orange: descriptive. Dashed line: frozen relative bound (cosine/control bounds also apply).\n"
             "Budget includes one exact edited block 1 and one source-only attention 2 per case. "
             "This does not forecast the final residual or verdict margin.", fontsize=9)
    save(fig, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("diagnostic", "forecast"), required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    {"diagnostic":diagnostic, "forecast":forecast}[args.kind](args.run, args.out)
