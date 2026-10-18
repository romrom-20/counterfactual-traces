"""Scientific figure of every 229 case; no fitted explanations or uncertainty bars."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(run, destination):
    with (run/"cases.csv").open() as f:
        cases = list(csv.DictReader(f))
    with (run/"outcomes.csv").open() as f:
        outcomes = list(csv.DictReader(f))
    lookup = {(r["key"], r["arm"]):r for r in outcomes}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    colors = {"generic":"#007d91", "descriptive":"#be6237"}
    rng = np.random.default_rng(229)
    names = ("R", "C", "QK", "SM", "AVs", "AVd")
    for i, name in enumerate(names):
        values = []
        for r in cases:
            v = float(r[name+"_signed_share"])
            values.append(v)
            axes[0].scatter(i+rng.uniform(-.13, .13), v, s=17, color=colors[r["schema"]], alpha=.75)
        axes[0].plot([i-.22, i+.22], [np.median(values)]*2, color="black", lw=2)
    axes[0].axhline(0, color="gray", lw=.7)
    axes[0].axhline(1, color="gray", lw=.7, ls="--")
    axes[0].set(xticks=range(6), xticklabels=names, ylabel="Signed projection onto first error E",
                title="Verdict position: where the first error comes from")
    for j, metric in enumerate(("vector_error", "margin_error")):
        for r in cases:
            baseline = float(lookup[r["key"], "base"][metric])
            for i, name in enumerate(("R", "C")):
                value = float(lookup[r["key"], "repair_"+name][metric])/baseline
                random = float(lookup[r["key"], "random_"+name][metric])/baseline
                x = i+j*3
                axes[1].scatter(x+rng.uniform(-.13, .13), value, color=colors[r["schema"]], s=17, alpha=.75)
                axes[1].scatter(x+.65+rng.uniform(-.13, .13), random, color=colors[r["schema"]], s=10, alpha=.3, marker="x")
    axes[1].axhline(1, color="gray", lw=.7)
    axes[1].axhline(.5, color="gray", lw=.7, ls="--")
    axes[1].set(xticks=(0, .65, 1, 1.65, 3, 3.65, 4, 4.65),
                xticklabels=("R", "rnd", "C", "rnd", "R", "rnd", "C", "rnd"),
                ylabel="Error / uncorrected early-error baseline",
                title="Finite suffix: residual (left), margin (right)")
    for i, name in enumerate(("QK", "SM", "AVs", "AVd")):
        for r in cases:
            value = float(lookup[r["key"], "remove_"+name]["signed_margin_error"])
            random = float(lookup[r["key"], "random_"+name]["signed_margin_error"])
            axes[2].scatter(i+rng.uniform(-.13, .13), value, s=17, color=colors[r["schema"]], alpha=.75)
            axes[2].scatter(i+.35+rng.uniform(-.08, .08), random, s=10, marker="x", color=colors[r["schema"]], alpha=.3)
    axes[2].axhline(0, color="gray", lw=.7)
    axes[2].set(xticks=range(4), xticklabels=("QK", "softmax", "AV source", "AV receiver"),
                ylabel="Margin after deletion − native margin (nats)",
                title="Direct primitive deletions; × matched random")
    fig.suptitle("229 · attention 2, exact incoming native state · 32 previously observed contexts", fontsize=13)
    fig.text(.01, -.045, "Teal: generic; orange: descriptive. Points are dependent template cases. "
             "R: receiver-only remainder; C: source–receiver interaction.\n"
             "Signed primitive allocation is order-defined. Finite suffix tests local sensitivity, not the remaining 228 linear forecast.", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(destination.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (run/"position-effects.csv").open() as f:
        positions = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, color in (("R", "#775a9d"), ("C", "#298065")):
        for offset in range(5):
            subset = [r for r in positions if int(r["receiving_offset"]) == offset]
            values = [float(r[name+"_signed_share"]) for r in subset]
            axes[0].scatter([offset+(-.13 if name == "R" else .13)]*len(values), values,
                            color=color, alpha=.35, s=18)
        medians = [np.median([float(r[name+"_signed_share"]) for r in positions
                              if int(r["receiving_offset"]) == i]) for i in range(5)]
        axes[0].plot(range(5), medians, label=name, color=color)
    axes[0].set(xticks=range(5), xlabel="Receiving offset after edited Boolean",
                ylabel="Signed projection onto error at this row",
                title="The interaction is concentrated near the source")
    axes[0].legend()
    import json
    description = json.loads((run/"description.json").read_text())
    for i, name in enumerate(names):
        for j, key in enumerate(("all", "verdict")):
            values = ([r[name+"_signed_share"] for r in description["cases"]] if key == "all"
                      else [float(r[name+"_signed_share"]) for r in cases])
            x = i+(-.15 if j == 0 else .15)
            axes[1].scatter([x]*len(values), values, color=("#287f85", "#c27c42")[j], alpha=.35, s=18)
            axes[1].plot([x-.10, x+.10], [np.median(values)]*2, color="black", lw=2)
    axes[1].set(xticks=range(6), xticklabels=names, ylabel="Signed projection onto E",
                title="All receiving rows (teal) vs verdict (orange)")
    for ax in axes:
        ax.axhline(0, color="gray", lw=.6)
        ax.axhline(1, color="gray", lw=.6, ls="--")
    fig.suptitle("229 · Position changes the apparent explanation", fontsize=13)
    fig.text(.01, -.045, "Post-observation geometry; all 32 cases at offsets 0–3, 16 at offset 4. "
             "Last receiving row differs with schema.\n"
             "R + C = E per case. Primitive medians need not add. No causal percentage or learned feature is inferred.", fontsize=9)
    for suffix in (".png", ".svg"):
        fig.savefig(destination.with_name("229-position-effects").with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.run, args.out)
