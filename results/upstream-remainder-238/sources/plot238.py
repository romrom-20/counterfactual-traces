"""Export a scientific all-case figure for the upstream238 diagnostic."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(run, out):
    import json
    with (run/"ledger.csv").open() as file:
        ledger = list(csv.DictReader(file))
    with (run/"outcomes.csv").open() as file:
        results = list(csv.DictReader(file))
    selected = json.loads((run/"selection.json").read_text())["selected"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
        "axes.spines.right":False, "svg.hashsalt":"upstream238", "savefig.facecolor":"#fbfcfe"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    ax = axes[0]
    for n in range(2, 39):
        group = [r for r in ledger if int(r["node"]) == n]
        x = float(np.median([float(r["local_remainder_norm"]) for r in group]))
        y = float(np.median([abs(float(r["signed_share"])) for r in group]))
        color = "#ba6648" if n % 2 else "#268d86"
        ax.scatter(max(x, 1e-8), max(y, 1e-8), color=color, s=48 if n in selected else 24,
            edgecolors="#263543" if n in selected else "none", linewidths=1)
        if n in selected:
            offset = {31:(12, 12), 33:(-58, -18), 35:(-58, 8), 37:(12, -10)}[n]
            ax.annotate("MLP"+str(n//2+1), (max(x, 1e-8), max(y, 1e-8)), xytext=offset,
                textcoords="offset points", fontsize=9, arrowprops={"arrowstyle":"-", "color":"#68798a", "lw":.7})
    ax.set_xlim(right=ax.get_xlim()[1]*1.5)
    ax.set_ylim(top=max(ax.get_ylim()[1], .45))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Local native-input remainder norm")
    ax.set_ylabel("Absolute signed terminal-error projection")
    ax.set_title("37 upstream generators\npoints: medians over 32 old contexts", fontsize=11, pad=12)
    ax.text(.03, .04, "Teal: attention · orange: MLP\nOutlined: four selected before outcomes", transform=ax.transAxes, fontsize=8)
    arms = ("base", "oracle4", "own4", "random4", "native")
    labels = ("Original", "Oracle R₄", "Finite own₄", "Row control", "Native reset")
    colors = ("#68798a", "#4b6fb5", "#258c7e", "#c78a46", "#4b6fb5")
    for axis, metric, threshold, title in ((axes[1], "all_relative", .05, "Final whole-displacement relative error"),
                                          (axes[2], "margin_error", .10, "Final margin error (nats)")):
        for index, arm in enumerate(arms):
            rows = sorted((r for r in results if r["arm"] == arm), key=lambda r:r["key"])
            values = np.array([float(r[metric]) for r in rows])
            jitter = np.linspace(-.16, .16, len(rows))
            for schema, marker in (("generic", "o"), ("descriptive", "s")):
                pick = np.array([r["schema"] == schema for r in rows])
                axis.scatter(index+jitter[pick], np.maximum(values[pick], 1e-6), color=colors[index], marker=marker,
                    s=16, alpha=.65, linewidths=0)
            axis.plot([index-.22, index+.22], [max(float(np.median(values)), 1e-6)]*2, color=colors[index], lw=3)
            axis.text(index, 1.025, str(sum(r["accuracy"] == "True" for r in rows))+"/32",
                transform=axis.get_xaxis_transform(), ha="center", fontsize=9)
        axis.axhline(threshold, color="#912d49", ls="--", lw=1)
        axis.set_yscale("log")
        axis.set_ylim(bottom=5e-7)
        axis.set_xticks(range(len(arms)), labels, rotation=25, ha="right")
        axis.set_title(title, fontsize=11, pad=25)
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("Does identifying upstream generators repair the full response?", y=.99, fontsize=15, weight="bold")
    fig.text(.045, .08, "All 32 dependent old contexts • bars: medians • circles: generic; squares: descriptive • zeros shown at10⁻⁶", fontsize=9)
    fig.text(.045, .045, "Counts require BOTH whole/verdict vector accuracy and margin≤0.10. Native remainders/reset are oracle diagnostics; final suffix is finite.", fontsize=9)
    fig.tight_layout(rect=(.01, .13, .995, .92), w_pad=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix("."+suffix), dpi=180, metadata={"Date":None} if suffix == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
