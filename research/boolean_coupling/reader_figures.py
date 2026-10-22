"""Posthoc reader figures from public CSVs; no model calls or new forecasts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    assert not out.exists()
    out.mkdir(parents=True)
    inputs = {}
    data = {}
    for folder in ("local-query-curvature-256", "receiver-mlp-curvature-258"):
        path = ROOT/"results"/folder/"outcomes.csv"
        with path.open() as stream:
            data[folder] = list(csv.DictReader(stream))
        inputs[str(path.relative_to(ROOT))] = sha(path)
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False,
        "figure.facecolor":"#fcfbf8", "axes.facecolor":"#fcfbf8", "savefig.facecolor":"#fcfbf8"})
    hashes = {}

    def save(fig, name):
        for ext in ("png", "svg"):
            path = out/(name+"."+ext)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            hashes[path.name] = sha(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 5), layout="constrained")
    ax.set(xlim=(0, 14), ylim=(0, 6))
    ax.axis("off")
    source = FancyBboxPatch((.4, 4.35), 13.1, 1.02, boxstyle="round,pad=.12", ec="#bbc6c9", fc="#eaf0ef")
    ax.add_patch(source)
    ax.text(6.95, 4.87, "Known Boolean edit + unchanged causal prefix\nSource keys and values are computed at each layer", ha="center", va="center", fontsize=12)
    xs = [.3, 3.85, 7.4, 10.95]
    texts = ["READ VALUES\nAttention at layer ℓ\nuses its current query", "UPDATE THE STATE\nAttention + gated MLP\nwrite to the receiver", "FORM A NEW QUERY\nLater normalization\nand learned projection", "READ VALUES AGAIN\nLater attention uses\nthe changed query"]
    for i, (x, label) in enumerate(zip(xs, texts, strict=True)):
        ax.add_patch(FancyBboxPatch((x, 1.9), 2.8, 1.55, boxstyle="round,pad=.13", ec="#7197a0", fc="#f2f7f6"))
        ax.text(x+1.4, 2.68, label, ha="center", va="center", fontsize=10.8, linespacing=1.55)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x+2.96, 2.67), (xs[i+1]-.2, 2.67), arrowstyle="-|>", mutation_scale=17, color="#456a75", lw=1.7))
    for x in (1.7, 12.35):
        ax.add_patch(FancyArrowPatch((x, 4.2), (x, 3.58), arrowstyle="-|>", mutation_scale=17, color="#456a75", lw=1.7))
    ax.text(6.9, 1.05, "Feedback is unrolled across depth; the transformer remains a feedforward computation.", ha="center", fontsize=11.4, color="#354f56")
    ax.text(6.9, .4, "The forecasts approximate selected normalization/activation operations. Other declared components stay finite.", ha="center", fontsize=10.1, color="#536267")
    fig.suptitle("The receiving state changes the way later information is read", fontsize=16, fontweight="semibold")
    save(fig, "changing-reader")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    specifications = [
        ("local-query-curvature-256", ("L", "Q", "C", "R"), ("Tangent", "Quadratic", "Cubic", "Random\ncontrol"), "Predicting query normalization"),
        ("receiver-mlp-curvature-258", ("Q", "R", "S", "C"), ("Quadratic\nSiLU", "Random 1%", "Ranked 1%", "Finite SiLU"), "Allocating beyond-quadratic SiLU terms")]
    for ax, (folder, arms, labels, title) in zip(axes, specifications, strict=True):
        counts = []
        for i, arm in enumerate(arms):
            selected = [r for r in data[folder] if r["arm"] == arm]
            assert len(selected) == 32
            values = [100*float(r["all_relative"]) for r in selected]
            passed = sum(r["passes"] == "True" for r in selected)
            color = "#247d78" if passed == 32 else "#bd7840"
            ax.scatter(i+np.linspace(-.13, .13, 32), values, s=20, alpha=.65, color=color, edgecolor="none")
            ax.plot([i-.22, i+.22], [float(np.median(values))]*2, lw=2.2, color="#273e47")
            counts.append(passed)
        ax.set_xticks(range(4), [f"{label}\n{count}/32 pass" for label, count in zip(labels, counts, strict=True)], fontsize=10)
        ax.set_yscale("log")
        ax.axhline(5, color="#8c8b84", ls="--", lw=1)
        ax.set(ylabel="Whole receiving-displacement error (%)")
        ax.set_title(title, pad=14)
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("Full-response forecasts on fresh literal pairs\nSeparate 32-pair sets per panel; black bars are medians", fontsize=14)
    save(fig, "forecast-results")
    payload = dict(posthoc=True, model_calls=0, source_sha256=sha(Path(__file__)), input_sha256=inputs,
        figure_sha256=hashes, all_cases=True,
        interpretation="Conceptual unrolled-depth diagram and all-case native forecast errors; pass counts use vector/cosine/norm/margin joint criteria, not only the plotted metric.")
    (out/"provenance.json").write_text(json.dumps(payload, indent=2)+"\n")
    print("Generated reader diagram and all-case forecast figure from sealed public CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
