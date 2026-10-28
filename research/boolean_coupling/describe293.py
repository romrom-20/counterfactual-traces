"""All-case native/paired results and same-input nonlinear write capture."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/response-subspace-293"
OUT = ROOT / "docs/figures/response-subspace-293"
ARMS = ("P", "L", "F", "E", "R")
LABELS = ("P: full", "L: linear only", "F: False range", "E: adaptive range", "R: random")
COLORS = ("#087f8c", "#7d7d7d", "#bf7421", "#5c58a0", "#b3444f")


def rows(name):
    with (DEST / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native = rows("native")
    pairs = rows("pairs")
    local = rows("local")
    audit = read(DEST / "audit.json")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "293",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    for ax, rr, field, title, bound in [
        (axes[0, 0], native, "receiving_relative", "Complete receiving displacement", 0.05),
        (axes[0, 1], native, "margin_error", "Finite verdict margin", 0.1),
        (axes[1, 0], pairs, "verdict_relative", "Paired schema verdict contrast", 0.1),
    ]:
        for j, (arm, col) in enumerate(zip(ARMS, COLORS, strict=True)):
            v = [float(r[field]) for r in rr if r["arm"] == arm]
            ax.scatter(
                j + np.linspace(-0.15, 0.15, len(v)),
                np.maximum(v, 1e-8),
                color=col,
                s=28,
                alpha=0.8,
            )
            ax.plot([j - 0.21, j + 0.21], [max(np.median(v), 1e-8)] * 2, color=col, lw=2)
        ax.axhline(bound, color="#222", ls="--", lw=1, label=f"Frozen bound {bound:g}")
        ax.set_yscale("log")
        ax.set_xticks(range(5), LABELS, rotation=15, ha="right", fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(
            "Absolute error (nats)" if field == "margin_error" else "Relative vector error"
        )
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 1]
    for basis, col, label in [("F", "#bf7421", "False range"), ("E", "#5c58a0", "Adaptive range")]:
        med = []
        for layer in range(1, 29):
            v = [
                float(r["relative"])
                for r in local
                if r["basis"] == basis
                and r["scope"] == "receiving"
                and int(r["layer"]) == layer
                and r["eligible"] == "True"
            ]
            med.append(np.median(v) if v else np.nan)
        ax.plot(range(1, 29), med, color=col, label=label, lw=2)
    ax.set_title("Same P inputs: nonlinear receiving write", loc="left", fontweight="bold")
    ax.set_ylabel("Median unexplained vector fraction")
    ax.set_xlabel("Block")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Restricting nonlinear MLP writes to 32 response directions",
        x=0.07,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    outcome = "; ".join(
        f"{a}: {audit['outcomes'][a]['native']}/8 native, {audit['outcomes'][a]['pairs']}/4 paired"
        for a in ARMS
    )
    fig.text(0.07, 0.925, outcome, fontsize=9)
    fig.text(
        0.07,
        0.025,
        "Qualified293c baseline; original science tables invalid. Eight old contexts, four dependent schema pairs. Dense operations still execute.\n32 directions constrain only the nonlinear MLP remainder; they do not bound the whole state or identify a circuit.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.855, hspace=0.52, wspace=0.26)
    for ext in ("png", "svg"):
        fig.savefig(
            OUT / ("response-subspace." + ext),
            dpi=175,
            metadata={"Date": None} if ext == "svg" else {},
        )
    plt.close(fig)
    p = OUT / "response-subspace.svg"
    p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines()) + "\n")
    write(
        OUT / "provenance.json",
        dict(
            experiment=293,
            all_eight_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / n).relative_to(ROOT)): sha(DEST / n)
                for n in ("native.csv", "pairs.csv", "local.csv", "audit.json")
            },
            outputs={p.name: sha(p) for p in sorted(OUT.glob("response-subspace.*"))},
        ),
    )


if __name__ == "__main__":
    main()
