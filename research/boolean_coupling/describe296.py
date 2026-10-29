"""All 32 sealed native geometry outcomes, retaining failed complete accounts."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-geometry-296/native-phase"
OUT = ROOT / "docs/figures/query-geometry-296"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "native.csv").open() as f:
        rows = list(csv.DictReader(f))
    a = read(DEST / "audit.json")
    arms = ["A", "D", "B", "R"]
    labels = [
        f"Magnitude\n{a['populations']['A']['passes']}/32",
        f"Direction\n{a['populations']['D']['passes']}/32",
        "Frozen\nquery\n0/32",
        "Random\nturn\n0/32",
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "296",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    for ax, field, title, bound in zip(
        axes,
        ["receiving_relative", "verdict_relative", "margin_error"],
        ["Complete receiving displacement", "Verdict displacement", "Finite verdict margin"],
        [0.05, 0.05, 0.1],
        strict=True,
    ):
        for j, arm in enumerate(arms):
            for layout, offset, color, label in [
                ("jsonlines", -0.21, "#087f8c", "JSON lines"),
                ("ini", -0.07, "#c16a28", "INI"),
                ("sexpr", 0.07, "#66528b", "S-expression"),
                ("memo", 0.21, "#668b3d", "Memo"),
            ]:
                values = [
                    float(r[field]) for r in rows if r["arm"] == arm and r["layout"] == layout
                ]
                ax.scatter(
                    j + offset + np.linspace(-0.035, 0.035, len(values)),
                    values,
                    s=18,
                    alpha=0.7,
                    color=color,
                    label=label if j == 0 else None,
                )
            med = np.median([float(r[field]) for r in rows if r["arm"] == arm])
            ax.plot([j - 0.28, j + 0.28], [med, med], color="#222", lw=1.5)
        ax.axhline(bound, color="#222", ls="--", lw=1)
        ax.set_yscale("log")
        vals = [float(r[field]) for r in rows]
        positive = [v for v in vals if v > 0]
        ax.set_ylim(min(positive) / 2, max(vals) * 1.4)
        ax.set_xticks(range(4), labels)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.15)
        ax.set_ylabel(
            "Absolute error (nats)" if field == "margin_error" else "Relative vector error"
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.12),
        ncol=4,
    )
    fig.suptitle(
        "Query orientation recovers more; neither restricted account is sufficient",
        x=0.055,
        ha="left",
        fontweight="bold",
        fontsize=15,
    )
    fig.text(
        0.055,
        0.88,
        "Both preregistered population claims FAIL. Complete native successes: magnitude 0/32; direction 1/32; controls 0/32.",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.035,
        "All 32 new inputs, four related synthetic encodings, one checkpoint. Forecasts sealed before native True. Black bars: medians.\nDashed lines are individual error bounds; complete success also requires both vectors, cosines and finite margin. No compact/circuit claim.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.27, top=0.78, wspace=0.32)
    for ext in ["png", "svg"]:
        fig.savefig(
            OUT / ("query-geometry." + ext),
            dpi=175,
            metadata={"Date": None} if ext == "svg" else {},
        )
    plt.close(fig)
    p = OUT / "query-geometry.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    write(
        OUT / "provenance.json",
        dict(
            experiment=296,
            all_32_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / n).relative_to(ROOT)): sha(DEST / n)
                for n in ["native.csv", "audit.json"]
            },
            outputs={p.name: sha(p) for p in sorted(OUT.glob("query-geometry.*"))},
        ),
    )


if __name__ == "__main__":
    main()
