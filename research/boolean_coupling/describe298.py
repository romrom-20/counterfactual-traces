"""Conditional residual and margin errors for all298 cases and both directions."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/context-messages-298"
OUT = ROOT / "docs/figures/context-messages-298"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "effects.csv").open() as f:
        rows = list(csv.DictReader(f))
    groups = ["chosen", "union", "neighbor", "random1", "random2"]
    labels = [
        "Chosen\npositions",
        "Union of\nboth groups",
        "Neighbor\ncontrol",
        "Random\ncontrol 1",
        "Random\ncontrol 2",
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "298",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 5.7))
    for ax, field, title, bound in zip(
        axes,
        ["receiving_relative", "verdict_relative", "margin_ratio"],
        [
            "Conditional receiving vector",
            "Conditional verdict vector",
            "Finite margin error / case bound",
        ],
        [0.25, 0.25, 1],
        strict=True,
    ):
        for j, group in enumerate(groups):
            for schema, offset, color in [
                ("generic", -0.13, "#087f8c"),
                ("descriptive", 0.13, "#c16a28"),
            ]:
                for op, dx, marker in [("add", -0.035, "o"), ("remove", 0.035, "^")]:
                    rr = [
                        r
                        for r in rows
                        if r["group"] == group and r["schema"] == schema and r["operation"] == op
                    ]
                    vals = [
                        float(r["margin_error"]) / float(r["margin_bound"])
                        if field == "margin_ratio"
                        else float(r[field])
                        for r in rr
                    ]
                    ax.scatter(
                        j + offset + dx + np.linspace(-0.018, 0.018, len(vals)),
                        vals,
                        s=25,
                        c=color,
                        marker=marker,
                        alpha=0.8,
                    )
        ax.axhline(bound, color="#333", ls="--", lw=1)
        ax.set_xticks(range(len(groups)), labels, fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.15)
        ax.set_ylabel(
            "Relative vector error"
            if field != "margin_ratio"
            else "Absolute error divided by frozen bound"
        )
        ax.set_ylim(
            0.05 if field != "margin_ratio" else 0.005, 3 if field != "margin_ratio" else 100
        )
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", color="#087f8c", label="Generic"),
        Line2D([], [], marker="o", ls="", color="#c16a28", label="Descriptive"),
        Line2D([], [], marker="o", ls="", color="#444", label="Addition"),
        Line2D([], [], marker="^", ls="", color="#444", label="Removal"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.12), ncol=4, frameon=False
    )
    fig.suptitle(
        "The selected message groups do not explain the conditional vector response",
        x=0.06,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    fig.text(
        0.06,
        0.88,
        "Primary hypothesis FAILS: chosen group 0/8 joint passes; union 0/8. All-position reconstruction calibrates correctly.",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "Eight retained cases; both addition and removal shown. Five margin effects reach .02 nat; all eight remain denominator.\nA scalar component passes in four descriptive cases while their vector tests fail. Fixed native parents: not a prospective Boolean forecast.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.78, bottom=0.31, wspace=0.31)
    for ext in ["png", "svg"]:
        fig.savefig(
            OUT / ("context-messages." + ext),
            dpi=175,
            metadata={"Date": None} if ext == "svg" else {},
        )
    plt.close(fig)
    p = OUT / "context-messages.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    write(
        OUT / "provenance.json",
        dict(
            experiment=298,
            all_eight_cases=True,
            both_directions=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / n).relative_to(ROOT)): sha(DEST / n)
                for n in ["effects.csv", "audit.json"]
            },
            outputs={p.name: sha(p) for p in sorted(OUT.glob("context-messages.*"))},
        ),
    )


if __name__ == "__main__":
    main()
