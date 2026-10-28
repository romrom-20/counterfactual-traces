"""Render all-case sealed forecast results and posthoc trajectory accounting."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/joint-gate-292"
OUT = ROOT / "docs/figures/joint-gate-292"
ARMS = ("P", "D", "C", "B", "A")
LABELS = (
    "P: rational\n2 doublings",
    "D: rational\n0 doublings",
    "C: cubic",
    "B: clipped\ncubic",
    "A: omit\nmixed product",
)
COLORS = ("#087f8c", "#63758a", "#c34730", "#b78a24", "#7b4ba1")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "native-phase/native.csv").open() as f:
        rows = list(csv.DictReader(f))
    review = read(DEST / "posthoc-vector-review.json")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "joint292",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for ax, field, title, bound in zip(
        axes.flat[:3],
        ("receiving_relative", "verdict_relative", "margin_error"),
        ("Final receiving displacement", "Final verdict displacement", "Finite verdict margin"),
        (0.05, 0.05, 0.1),
        strict=True,
    ):
        for j, (arm, col) in enumerate(zip(ARMS, COLORS, strict=True)):
            vals = [float(r[field]) for r in rows if r["arm"] == arm]
            offsets = np.tile(np.linspace(-0.18, 0.18, 8), 4)
            ax.scatter(j + offsets, vals, s=17, c=col, alpha=0.7, edgecolors="none")
            ax.plot([j - 0.23, j + 0.23], [np.median(vals)] * 2, color=col, lw=2.4)
        ax.axhline(bound, color="#222", ls="--", lw=1, label=f"Frozen bound {bound:g}")
        ax.set_yscale("log")
        ax.set_xticks(range(5), LABELS, fontsize=8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(
            "Absolute error (nats)" if field == "margin_error" else "Relative vector error"
        )
        ax.grid(axis="y", alpha=0.16)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax = axes.flat[3]
    for scope, label, col in [
        ("source_states", "Source states", "#087f8c"),
        ("receiving_states", "Receiving states", "#c34730"),
        ("source_verdict_MLP_writes", "Source/verdict MLP writes", "#7b4ba1"),
    ]:
        arr = np.array(
            [
                [r["relative"] for r in review["layers"] if r["scope"] == scope and r["layer"] == j]
                for j in range(1, 29)
            ]
        )
        x = np.arange(1, 29)
        ax.fill_between(
            x, np.maximum(arr.min(1), 1e-8), np.maximum(arr.max(1), 1e-8), color=col, alpha=0.12
        )
        ax.plot(x, np.maximum(np.median(arr, axis=1), 1e-8), color=col, label=label, lw=1.8)
    ax.set_title("P across layers: median and full case range", loc="left", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("Block")
    ax.set_ylabel("Layerwise relative displacement error")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "A coherent nonlinear forecast recovers the native response",
        x=0.07,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.928,
        "32 fresh literal contexts; forecasts sealed before native True. P: 32/32 strict joint passes; D/C/B/A: 0/32 each.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.027,
        "Finite attention, normalization, projections, joint products and head remain. No speedup or learned-circuit claim.\nDots retain all cases; pairs/templates are dependent. Layerwise panel is posthoc, not another frozen endpoint.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.86, hspace=0.48, wspace=0.25)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUT / ("joint-gate-forecast." + suffix),
            dpi=175,
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(fig)
    write(
        OUT / "provenance.json",
        dict(
            experiment=292,
            all_32_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str(p.relative_to(ROOT)): sha(p)
                for p in [DEST / "native-phase/native.csv", DEST / "posthoc-vector-review.json"]
            },
            outputs={p.name: sha(p) for p in sorted(OUT.glob("joint-gate-forecast.*"))},
            boundary="All-case published records; posthoc layer accounting explicitly labeled. No uncertainty interval for dependent pairs.",
        ),
    )


if __name__ == "__main__":
    main()
