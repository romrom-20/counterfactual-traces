"""Export all-case scientific figures from the retained237b audit CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ("legacy", "1001", "1101", "1011", "1111", "public")
LABELS = ("False-linear", "N + AV", "+ QK", "+ softmax", "Both", "Public finite")
COLORS = ("#68798a", "#bd6444", "#998535", "#28897e", "#486fb5", "#486fb5")


def main(run, out):
    with (run/"outcomes.csv").open() as file:
        rows = list(csv.DictReader(file))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
        "axes.spines.right":False, "svg.hashsalt":"late237b", "savefig.facecolor":"#fbfcfe"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, regime in enumerate(("forecast", "oracle")):
        for row, (metric, threshold, label) in enumerate((("all_relative", .05, "Final whole-displacement relative error"),
                                                         ("margin_error", .10, "Final absolute margin error (nats)"))):
            axis = axes[row, col]
            axis.set_facecolor("#fbfcfe")
            for index, mode in enumerate(MODES):
                sub = sorted((r for r in rows if r["arm"] == regime+"_"+mode), key=lambda r:r["key"])
                values = np.array([float(r[metric]) for r in sub])
                jitter = np.linspace(-.16, .16, len(sub))
                for schema, marker in (("generic", "o"), ("descriptive", "s")):
                    selected = np.array([r["schema"] == schema for r in sub])
                    axis.scatter(index+jitter[selected], np.maximum(values[selected], 1e-6), s=15, marker=marker,
                        color=COLORS[index], alpha=.6, linewidths=0)
                axis.plot([index-.23, index+.23], [max(float(np.median(values)), 1e-6)]*2, color=COLORS[index], lw=3)
                if row == 0:
                    passed = sum(r["accuracy"] == "True" for r in sub)
                    axis.text(index, 1.04, f"{passed}/32", transform=axis.get_xaxis_transform(), ha="center", fontsize=9)
            axis.axhline(threshold, ls="--", lw=1, color="#912d49")
            axis.set_yscale("log")
            axis.set_ylim(5e-7, 15 if row else 3)
            axis.set_ylabel(label)
            axis.set_xticks(range(len(MODES)), LABELS, rotation=18, ha="right")
            axis.grid(axis="y", which="major", alpha=.15)
            if row == 0:
                axis.set_title("Original forecast incoming state" if col == 0 else "Exact native reset before MLP20", pad=30)
    fig.suptitle("Which late-attention omissions still break the final response?", x=.5, y=.98, fontsize=16, weight="bold")
    fig.text(.08, .075, "32 previously observed dependent contexts • dots: all cases; bars: medians • circles: generic; squares: descriptive", fontsize=9)
    fig.text(.08, .052, "Counts require BOTH whole/verdict vector accuracy and margin ≤0.10. Dashed bounds: 5% / 0.10 nats. Zeros shown at10⁻⁶.", fontsize=9)
    fig.text(.08, .029, "Retrospective omissions; raw numerical bridges fail (see report). Public finite is a replay control, not a new predictor.", fontsize=9)
    fig.tight_layout(rect=(.015, .115, .995, .94))
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
