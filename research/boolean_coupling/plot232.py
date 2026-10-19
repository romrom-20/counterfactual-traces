"""All source-context forecast cases, retaining the primary/secondary distinction."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(run, destination):
    with (run/"cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    colors = {"generic":"#087d8c", "descriptive":"#bc663e"}
    markers = {"indigo":"o", "quartz":"^"}
    rng = np.random.default_rng(232)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    for ax, scope in zip(axes, ("all", "verdict")):
        for i, arm in enumerate(("FNV", "SL", "SNV", "R")):
            values = []
            for row in rows:
                value = float(row[scope+"_"+arm+"_relative"])
                values.append(value)
                ax.scatter(i+rng.uniform(-.12, .12), value, color=colors[row["schema"]],
                           marker=markers[row["block"]], s=22, alpha=.75)
            ax.plot([i-.2, i+.2], [np.median(values)]*2, color="black")
        ax.axhline(.025, color="gray", ls="--", lw=.8)
        ax.set(xticks=range(4), xticklabels=("False\nnorm+product", "source\nlinear", "source\nnorm+product", "matched\nrandom"),
               ylabel="Relative native displacement error", title="Primary: all receiving rows" if scope == "all" else "Secondary: verdict accuracy preservation")
    fig.suptitle("232 · Change the expansion context, retain the finite operations · 32 fresh pairs", fontsize=13)
    fig.text(.01, -.055, "Teal: generic; orange: descriptive. Circle/triangle: two dependent literal families. "
             "Dashed: frozen 2.5% bound; cosine applies to both, control comparisons to primary only.\n"
             "Exact edited block 1 and source-only attention 2 are declared work. "
             "This predicts attention 2, not final behavior or block-1 generation.", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(destination.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.run, args.out)
