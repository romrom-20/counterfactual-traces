"""Render the explicitly posthoc transport metric identity; no model calls."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(run, out):
    with (run / "transport-metric.csv").open() as file:
        rows = list(csv.DictReader(file))
    metric = json.loads((run / "metric-description.json").read_text())["means"]
    keys = sorted({r["key"] for r in rows})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "metric241", "savefig.facecolor": "#fbfcfe"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for i, (a, b) in enumerate((("remainder_energy", "terminal_diagonal"), ("cross_energy", "terminal_pairwise"))):
        ax[0].bar(i - .18, metric[a], width=.32, color="#526f94", label="At local injection" if i == 0 else None)
        ax[0].bar(i + .18, metric[b], width=.32, color="#287f77", label="After transport to MLP16 input" if i == 0 else None)
    ax[0].axhline(0, color="#7a818b", lw=.7)
    ax[0].set_xticks([0, 1], ["Individual-vector energy", "Cross / pairwise energy"])
    ax[0].set(title="Both vector size and cancellation change", ylabel="Mean energy / terminal input error²")
    ax[0].legend(fontsize=8)
    x = np.arange(2, 31) / 2 + 1
    for name, label, color in (("diagonal_change", "Individual-vector energy change", "#526f94"),
        ("pairwise_change", "Pairwise-interaction change", "#ba6b4b"), ("total_change", "Net transport change", "#30343b")):
        values = np.array([[float(next(r for r in rows if r["key"] == key and int(r["node"]) == n)[name]) for n in range(2, 31)] for key in keys])
        ax[1].plot(x, np.mean(np.cumsum(values, 1), 0), color=color, label=label, lw=2)
    ax[1].axhline(0, color="#7a818b", lw=.7)
    ax[1].set(title="Large opposing changes leave a small net", xlabel="Block (attention then MLP)", ylabel="Mean cumulative change / terminal input error²")
    ax[1].legend(fontsize=8)
    for axis in ax:
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("A small net transport effect does not imply unimportant transport", fontsize=14, weight="bold", y=.99)
    fig.text(.07, .105, "All 32 old contexts, per-case normalized means. Transport changes: individual −.429; pairwise +.450; net +.0216.", fontsize=9)
    fig.text(.07, .055, "Posthoc arithmetic identity for the original False-affine maps and native remainders; not a learned circuit or a new causal test.", fontsize=9)
    fig.tight_layout(rect=(.01, .14, .995, .94), w_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix("." + suffix), dpi=180, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
