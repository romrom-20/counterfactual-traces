"""Render240's local recovery, finite margin gaps and remaining native error."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(run, name):
    with (run / (name + ".csv")).open() as file:
        return list(csv.DictReader(file))


def dots(axis, values, x, color):
    values = np.asarray(values)
    axis.scatter(x + np.linspace(-.07, .07, len(values)), values, color=color, s=14, alpha=.55, linewidths=0)
    axis.plot([x - .14, x + .14], [np.median(values)] * 2, color=color, lw=3)


def main(run, out):
    masks, final, native = (rows(run, n) for n in ("masks", "final", "native-errors"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "silu240", "savefig.facecolor": "#fbfcfe"})
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    m = [r for r in masks if r["scope"] == "all"]
    for x, key, color in ((0, "original239_relative", "#b66a4a"), (1, "repaired239_relative", "#267f76")):
        dots(ax[0], [100 * float(r[key]) for r in m], x, color)
    ax[0].set_xticks([0, 1], ["Quadratic SiLU", "Restore selected 1%"])
    ax[0].set_title("MLP16 generator-contrast error")
    ax[0].set_ylabel("Whole receiving relative error (%)")
    ax[0].set_yscale("log")
    for i, state in enumerate(("native", "own")):
        sub = [r for r in final if r["state"] == state]
        for shift, key, color in ((-.18, "quadratic_margin", "#b66a4a"), (.18, "mask_margin", "#267f76")):
            dots(ax[1], [float(r[key]) for r in sub], i + shift, color)
    ax[1].set_xticks([0, 1], ["Native input", "Forecast input"])
    ax[1].axhline(.1, color="#8d3651", ls="--", lw=1)
    ax[1].set_title("Conditional final margin discrepancy")
    ax[1].set_ylabel("Gap from same-input full SiLU (nats)")
    ax[1].set_yscale("log")
    for x, arm, label, color in ((0, "quadratic", "Quadratic", "#b66a4a"), (1, "mask", "Selected 1%", "#267f76"), (2, "full", "Full SiLU", "#586c80")):
        sub = [r for r in native if r["state"] == "own" and r["arm"] == arm]
        dots(ax[2], [100 * float(r["whole_relative"]) for r in sub], x, color)
    ax[2].set_xticks([0, 1, 2], ["Quadratic", "Selected 1%", "Full SiLU"])
    ax[2].set_ylim(0, 108 * max(float(r["whole_relative"]) for r in native if r["state"] == "own"))
    ax[2].set_title("Forecast input still misses native response")
    ax[2].set_ylabel("Final native whole-displacement error (%)")
    for axis in ax:
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("Predictable local SiLU tails do not explain the remaining native-response failure", fontsize=13, weight="bold", y=.99)
    fig.text(.055, .115, "All 32 old contexts · dots: cases; bars: medians · orange: quadratic; teal: selected 1%; gray: full SiLU", fontsize=9)
    fig.text(.055, .075, "Middle dashed line: preregistered .10-nat materiality guard. Eligible: native 3/32, forecast 0/32. Matched controls retained in CSVs.", fontsize=9)
    fig.text(.055, .035, "Native incoming states are oracle diagnostics. The mask uses False/forecast gates only; this is not a fresh full-response predictor.", fontsize=9)
    fig.tight_layout(rect=(.01, .16, .995, .94), w_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix("." + suffix), dpi=180, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
