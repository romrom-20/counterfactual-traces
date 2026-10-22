"""Posthoc all-case figures for the261–264 normalization interventions."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/mlp-normalization-origin-261-264"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = [EVIDENCE/"normalization-branches-energy-263/outcomes.csv",
        EVIDENCE/"normalization-component-interaction-264b/layerwise.csv"]
    rows, contrasts = [] , []
    for path, target in zip(inputs, (rows, contrasts), strict=True):
        with path.open() as stream:
            target.extend(csv.DictReader(stream))
    arms = ("L", "G", "U", "P", "D", "E", "H", "B")
    labels = ("Linear\nboth", "Finite\ngate", "Finite\nup", "+pattern", "+direction", "+energy", "+both", "Finite\nboth")
    keys = list(dict.fromkeys(r["key"] for r in rows))
    style = {}
    for key in keys:
        r = next(row for row in rows if row["key"] == key)
        style[key] = dict(color="#17798a" if r["schema"] == "generic" else "#c36532",
            marker="o" if r["reverse"] == "0" else "s", label=r["schema"]+" "+r["reverse"])
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False,
        "svg.fonttype":"none", "figure.facecolor":"#ffffff"})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1), gridspec_kw={"width_ratios":[1.35, 1.35, 1.]}, layout="constrained")
    for ax, field, scale, limit, title, ylabel in (
        (axes[0], "all_relative", 100, 5, "Vector accuracy improves before the explanation is complete", "Final whole-displacement error (%)"),
        (axes[1], "margin_error", 1, .1, "The verdict margin rejects the energy-only account", "Final margin error (nats)")):
        for index, arm in enumerate(arms):
            selected = [r for r in rows if r["arm"] == arm]
            assert len(selected) == 4
            for offset, key in zip(np.linspace(-.19, .19, 4), keys, strict=True):
                r = next(row for row in selected if row["key"] == key)
                ax.scatter(index+offset, scale*float(r[field]), s=35, **{k:v for k, v in style[key].items() if k != "label"}, zorder=3)
            ax.plot([index-.25, index+.25], [scale*np.median([float(r[field]) for r in selected])]*2, color="#252c32", lw=1.6)
        ax.axhline(limit, color="#9b244a", linestyle="--", lw=1, label="Frozen tolerance")
        ax.set_xticks(range(len(arms)), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, wrap=True, pad=13)
        ax.set_ylim(bottom=-.02*ax.get_ylim()[1])
        ax.grid(axis="y", alpha=.14)
    selected = [r for r in contrasts if r["scope"] == "verdict" and r["layer"] == "28" and r["contrast"] == "direction_alone"]
    assert len(selected) == 4
    for index, key in enumerate(keys):
        r = next(row for row in selected if row["key"] == key)
        value = float(r["cosine"])
        axes[2].bar(index, value, color=style[key]["color"], width=.62)
        axes[2].text(index, value+(.018 if value >= 0 else -.026), f"{value:.2f}", ha="center", va="center", fontsize=9)
    axes[2].axhline(0, color="#586069", lw=.8)
    axes[2].set_xticks(range(4), [style[k]["label"].replace(" ", "\n") for k in keys], fontsize=8)
    axes[2].set_ylim(-.24, .28)
    axes[2].set_ylabel("Cosine: (D − L) versus (H − P)")
    axes[2].set_title("The direction term's final vector effect\ndepends on its background", fontsize=10, pad=13)
    axes[2].grid(axis="y", alpha=.14)
    fig.suptitle("MLP normalization: a substantial cause, an incomplete compressed account", fontsize=15, fontweight="bold")
    fig.supxlabel("All four old contexts · evolving own-state interventions · finite final readout · no new native examples", fontsize=10)
    out = ROOT/"docs/figures/normalization"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out/("normalization-origin."+suffix), dpi=190, bbox_inches="tight")
    plt.close(fig)
    (out/"provenance.json").write_text(json.dumps(dict(posthoc=True, all_cases=True, model_calls=0,
        source_sha256=sha(Path(__file__)), input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        figure_sha256={p.name:sha(p) for p in sorted(out.glob("normalization-origin.*"))}), indent=2)+"\n")
    print("Rendered all32 intervention outcomes and all4 final verdict-vector contrasts.")


if __name__ == "__main__":
    main()
