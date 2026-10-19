"""Render every source-transport diagnostic/forecast case from recorded CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def panel(ax, groups, labels, values, bound, title):
    for i, name in enumerate(groups):
        subset = values(name)
        for j, (row, value) in enumerate(subset):
            offset = (j % 8-3.5)*.018
            color = "#007b8e" if row["schema"] == "generic" else "#b75b31"
            marker = "o" if row["block"] in ("indigo", "jade") else "^"
            ax.scatter(i+offset, value, color=color, marker=marker, s=25, alpha=.65)
        median = np.median([v for _, v in subset])
        ax.hlines(median, i-.21, i+.21, color="black", linewidth=2)
    ax.axhline(bound, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=9)


def save(fig, out, name):
    for suffix in ("png", "svg"):
        p = out/(name+"."+suffix)
        assert not p.exists(), "Never overwrite an inspected figure."
        fig.savefig(p, dpi=180, metadata={"Date":None} if suffix == "svg" else None)
    plt.close(fig)


def main(run233, run234, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.hashsalt":"transport-233-234"})
    with (run233/"operators.csv").open() as f:
        old = [r for r in csv.DictReader(f) if r["scope"] == "all"]
    groups = ["0000", "1000", "0100", "0010", "0001", "1010", "1111"]
    labels = ["False", "key", "softmax\nresponse", "source\nvalue", "receiver V\nweights", "key +\nvalue", "all\nchanges"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    panel(axes[0], groups, labels, lambda b:[(r, float(r["transport_relative"])) for r in old if r["bits"] == b],
          .25, "233 retrospective: error in changed tangent")
    axes[0].set_ylabel("Relative error to (JS - JF)d")
    panel(axes[1], groups, labels, lambda b:[(r, float(r["relative"])) for r in old if r["bits"] == b],
          .025, "Apply each operator repair to old FNV forecast")
    axes[1].set_ylabel("Native all-receiving displacement relative error")
    fig.suptitle("Source key and value changes work jointly in the False softmax background", fontsize=15)
    fig.text(.015, .035, "Every old case shown; black: median; dashed: distinct frozen diagnostic bounds. Teal/orange: schema; shapes: dependent literal families.", fontsize=9)
    fig.text(.015, .009, "Operator interventions are not native input edits or learned circuits. This diagnoses one attention update; native targets were already observed.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .93))
    save(fig, out, "233-source-transport-primitives")
    with (run234/"cases.csv").open() as f:
        new = list(csv.DictReader(f))
    groups = ["FNV", "K", "V", "A", "C", "R0", "R1", "R2"]
    labels = ["old FNV", "key only", "value only", "no mixed\nproduct", "joint K/V\n+ mixed", "random 0", "random 1", "random 2"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    for ax, scope, title in zip(axes, ("all", "verdict"), ("234 primary: all receiving rows", "Secondary: verdict accuracy preservation")):
        panel(ax, groups, labels, lambda b:[(r, float(r[scope+"_"+b+"_relative"])) for r in new], .025, title)
        ax.set_ylabel("Native displacement relative error")
    fig.suptitle("Fresh, globally sealed source-key/value forecasts and direct mixed-product removal", fontsize=15)
    fig.text(.015, .035, "Every fresh pair shown; black: median; dashed: 2.5% bound. Cosine/norm guards also apply; control comparisons apply to the primary only.", fontsize=9)
    fig.text(.015, .009, "Exact edited block 1 and a finite source-only attention call remain declared work. No final residual or verdict-margin forecast.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .93))
    save(fig, out, "234-source-key-value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-233", type=Path, required=True)
    parser.add_argument("--input-234", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_233, args.input_234, args.out)
