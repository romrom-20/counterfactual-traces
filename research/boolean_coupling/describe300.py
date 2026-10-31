"""All eight conditional outcomes, with low-materiality cases still visible."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-interaction-300"
OUT = ROOT / "docs/figures/query-interaction-300"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "effects.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    audit = read(DEST / "audit.json")
    summary = dict(experiment=300, retrospective=True, all_eight_cases=True, medians={})
    arms = ("A", "R", "J", "O")
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        summary["medians"][arm] = {
            f: float(np.median([float(r[f]) for r in rr]))
            for f in ("receiving_relative", "verdict_relative", "margin_error")
        }
    write(DEST / "descriptive-summary.json", summary)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "300",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.0))
    names = (
        "Attention\njoint piece",
        "Query\ninteraction",
        "Matched\nrandom",
        "Both\n(control)",
    )
    labels = [
        n + f"\n{audit['populations'][a]['passes']}/8" for a, n in zip(arms, names, strict=True)
    ]
    for ax, field, title in zip(
        axes,
        ("receiving_relative", "verdict_relative", "margin_ratio"),
        ("Receiving displacement", "Verdict displacement", "Finite margin error / bound"),
        strict=True,
    ):
        for i, arm in enumerate(arms):
            for schema, off, color, label in (
                ("generic", -0.08, "#087f8c", "Generic"),
                ("descriptive", 0.08, "#c16a28", "Descriptive"),
            ):
                rr = [r for r in rows if r["arm"] == arm and r["schema"] == schema]
                for k, r in enumerate(rr):
                    value = (
                        float(r["margin_error"]) / max(float(r["margin_bound"]), 1e-30)
                        if field == "margin_ratio"
                        else float(r[field])
                    )
                    material = r["material"] == "True"
                    ax.scatter(
                        i + off + (k - 1.5) * 0.018,
                        value,
                        s=32,
                        edgecolors=color,
                        facecolors=color if material else "white",
                        label=label if i == 0 and k == 0 else None,
                        linewidths=1.2,
                    )
            vals = [
                float(r["margin_error"]) / max(float(r["margin_bound"]), 1e-30)
                if field == "margin_ratio"
                else float(r[field])
                for r in rows
                if r["arm"] == arm
            ]
            ax.plot([i - 0.24, i + 0.24], [np.median(vals)] * 2, color="#222", lw=1.5)
        ax.axhline(1 if field == "margin_ratio" else 0.25, color="#555", ls="--", lw=1)
        ax.set_xticks(range(4), labels, fontsize=8.8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel(
            "Error / case-specific margin bound"
            if field == "margin_ratio"
            else "Relative vector error"
        )
        if field == "margin_ratio":
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(bottom=-0.02)
        ax.grid(axis="y", alpha=0.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Restoring either joint piece alone misses the complete conditional response",
        x=0.055,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.88,
        "Block 15 only, eight already observed native contexts; finite downstream interventions. General Question B remains open.",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.035,
        "All eight cases remain. Hollow points: three below the frozen margin-materiality guard. Both-piece control is numerically exact in all eight.\nCounts: joint passes. Black bars: medians. Complete success also requires cosine and target-norm guards; supplied native parents are not a False-only forecast.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.35, wspace=0.32)
    for ext in ("png", "svg"):
        options = dict(dpi=170) if ext == "png" else dict(metadata={"Date": None})
        fig.savefig(OUT / ("finite-interaction." + ext), facecolor="white", **options)
    plt.close(fig)
    p = OUT / "finite-interaction.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    write(
        OUT / "provenance.json",
        dict(
            experiment=300,
            retrospective=True,
            all_eight_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / n).relative_to(ROOT)): sha(DEST / n)
                for n in ("effects.csv", "audit.json")
            },
            outputs={p.name: sha(p) for p in OUT.glob("*") if p.suffix in (".svg", ".png")},
        ),
    )
    print(summary)


if __name__ == "__main__":
    main()
