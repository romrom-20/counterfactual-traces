"""Posthoc all-case267 plot: field generation, local response and final interaction."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/correction-field-feedback-267/field-state-267-retry"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    paths = [EVIDENCE/name for name in ("factorial.csv", "native.csv")]
    data, native = map(rows, paths)
    keys = list(dict.fromkeys(r["key"] for r in data))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "question-b-fields267"})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.1), layout="constrained")
    families = ("10", "01", "00", "r")
    labels = ("Generator\nadapts", "MLP point\nadapts", "Both\nfrozen", "Random\nfrozen writes")
    for offset, key in zip(np.linspace(-.16, .16, 4), keys, strict=True):
        group = next(r for r in data if r["key"] == key)
        color = "#167b88" if group["schema"] == "generic" else "#ba6237"
        marker = "o" if group["reverse"] == "0" else "s"
        for i, family in enumerate(families):
            row = next(r for r in data if r["key"] == key and r["family"] == family)
            if row["complete"] != "True":
                continue
            axes[0].scatter(i+offset, 100*float(row["all_relative"]), color=color, marker=marker, s=35, zorder=3)
            axes[1].scatter(i+offset, float(row["all_cosine"]), color=color, marker=marker, s=35, zorder=3)
        for i, arm in enumerate(("L", "H", "A11", "A10", "A01", "A00", "Ar")):
            row = next(r for r in native if r["key"] == key and r["arm"] == arm)
            if row["status"] == "complete":
                axes[2].scatter(i+offset, 100*float(row["all_relative"]), color=color, marker=marker, s=30, zorder=3)
    for i, family in enumerate(families):
        selected = [r for r in data if r["family"] == family]
        count = sum(r["passes"] == "True" for r in selected)
        axes[0].text(i, 1.01, f"{count}/4 joint", ha="center", transform=axes[0].get_xaxis_transform(), fontsize=9)
    axes[0].axhline(25, color="#9d3155", lw=1, ls="--")
    axes[0].set_ylabel("Error against adaptive final interaction (%)")
    axes[0].set_title("Does the signed interaction survive?", pad=26)
    axes[1].axhline(.95, color="#9d3155", lw=1, ls="--")
    axes[1].axhline(0, color="#777777", lw=.6)
    axes[1].set_ylim(-1.08, 1.08)
    axes[1].set_ylabel("Cosine with adaptive final interaction")
    axes[1].set_title("Alignment retains the sign", pad=26)
    for ax in axes[:2]:
        ax.set_xticks(range(4), labels, fontsize=9)
    axes[2].axhline(5, color="#9d3155", lw=1, ls="--")
    axes[2].set_xticks(range(7), ("L", "Finite\nH", "A11", "A10", "A01", "A00", "Ar"), fontsize=9)
    axes[2].set_ylabel("Whole native-displacement error (%)")
    axes[2].set_title("Native accuracy is a separate check", pad=26)
    for ax in axes:
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("Cross correction generation with the MLP operating point", fontsize=15, fontweight="bold")
    fig.supxlabel("Four old contexts · full finite SiLU · cubic receiver queries · teal: generic, orange: descriptive; squares: reversed order", fontsize=9)
    dest = ROOT/"docs/figures/fields"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest/"field-feedback.png", dpi=190, bbox_inches="tight")
    fig.savefig(dest/"field-feedback.svg", bbox_inches="tight", metadata={"Date": None})
    svg = dest/"field-feedback.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    (dest/"provenance.json").write_text(json.dumps(dict(posthoc=True, all_cases=True,
        model_calls=0, source_sha256=sha(Path(__file__)), matplotlib=matplotlib.__version__,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        figure_sha256={p.name: sha(p) for p in dest.glob("field-feedback.*")}), indent=2)+"\n")
    print("Rendered all267 cases; primary joint decisions include vectors and finite common-anchor margins.")


if __name__ == "__main__":
    main()
