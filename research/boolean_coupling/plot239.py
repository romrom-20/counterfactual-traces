"""Export all239 cases with both measurement scopes and the frozen bound."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(run, out):
    with (run/"contrasts.csv").open() as file:
        rows = list(csv.DictReader(file))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
        "axes.spines.right":False, "svg.hashsalt":"curvature239", "savefig.facecolor":"#fbfcfe"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, scope in zip(axes, ("all", "verdict")):
        for index, n in enumerate((31, 33, 35, 37)):
            for model, shift, color in (("quadratic", -.13, "#ba6648"), ("finite", .13, "#258c7e")):
                sub = sorted((r for r in rows if int(r["node"]) == n and r["model"] == model and r["scope"] == scope), key=lambda r:r["key"])
                values = np.array([float(r["relative_error"]) for r in sub])
                axis.scatter(index+shift+np.linspace(-.05, .05, len(sub)), values, color=color, s=14, alpha=.55, linewidths=0)
                axis.plot([index+shift-.09, index+shift+.09], [np.median(values)]*2, color=color, lw=3)
        axis.axhline(.25, color="#912d49", ls="--", lw=1)
        axis.set_xticks(range(4), ("MLP16", "MLP17", "MLP18", "MLP19"))
        axis.set_title("Whole receiving sequence" if scope == "all" else "Verdict position")
        axis.set_ylabel("Relative error of native−forecast generator contrast")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=.15)
        axis.set_ylim(.025, 2)
    fig.suptitle("False-state curvature explains some local contrasts, not all four", fontsize=14, weight="bold", y=.99)
    fig.text(.075, .11, "Orange: full quadratic jet · teal: finite normalization + quadratic SiLU/product · dots: all32 old cases; bars: medians", fontsize=9)
    fig.text(.075, .065, "Primary requires ≤25% error AND cosine≥.95 in BOTH scopes. Pass counts16/17/18/19:0/32,32/32,32/32,21/32.", fontsize=9)
    fig.text(.075, .025, "Native input directions are supplied; this is a retrospective local diagnostic, not an end-to-end False-only forecast.", fontsize=9)
    fig.tight_layout(rect=(.01, .15, .995, .94), w_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix("."+suffix), dpi=180, metadata={"Date":None} if suffix == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
