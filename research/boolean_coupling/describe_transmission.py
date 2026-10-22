"""Posthoc all-case plot of265 removal and266 finite-activation controls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/"results/local-mlp-mixing-265-266"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    paths = [EVIDENCE/"finite-silu-interaction-266/comparison.csv",
        EVIDENCE/"local-mixing-continuation-265/outcomes.csv",
        EVIDENCE/"finite-silu-interaction-266/alignment.csv",
        ROOT/"results/mlp-normalization-origin-261-264/normalization-component-interaction-264b/layerwise.csv"]
    fidelity, outcomes, local, global_rows = map(read_rows, paths)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
        "axes.spines.right":False, "svg.fonttype":"none", "svg.hashsalt":"question-b-265-266"})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), layout="constrained", gridspec_kw={"width_ratios":[1, 1.15, 1.15]})
    for index, (mode, background) in enumerate((("adaptive", "H"), ("adaptive", "L"), ("fixed", "H"), ("fixed", "L"))):
        selected = [r for r in fidelity if r["mode"] == mode and r["background"] == background and r["scope"] == "all" and r["eligible"] == "True"]
        count = sum(r["passes"] == "True" for r in selected)
        fraction = 100*count/len(selected)
        color = "#17798a" if background == "H" else "#566178"
        axes[0].bar(index, fraction, color=color, width=.65)
        axes[0].scatter(index, fraction, color=color, s=24, zorder=3)
        axes[0].text(index, fraction+3, f"{count}/{len(selected)}", ha="center", fontsize=9)
    axes[0].axhline(90, color="#9b244a", linestyle="--", lw=1)
    axes[0].set_xticks(range(4), ["Adaptive\nH", "Adaptive\nL", "Common mask\nH", "Common mask\nL"], fontsize=8)
    axes[0].set_ylim(-3, 113)
    axes[0].set_ylabel("Local mixed-write fidelity passes (%)")
    axes[0].set_title("Separate masks misrepresent the\nfinite-SiLU interaction", fontsize=11, pad=13)
    keys = list(dict.fromkeys(r["key"] for r in outcomes))
    for index, arm in enumerate(("A", "K", "R")):
        selected = [r for r in outcomes if r["arm"] == arm]
        assert len(selected) == 4
        for offset, key in zip(np.linspace(-.16, .16, 4), keys, strict=True):
            r = next(row for row in selected if row["key"] == key)
            color = "#17798a" if r["schema"] == "generic" else "#c36532"
            axes[1].scatter(index+offset, 100*float(r["all_interaction_relative"]), color=color,
                marker="o" if r["reverse"] == "0" else "s", s=35, zorder=3)
        mid = 100*np.median([float(r["all_interaction_relative"]) for r in selected])
        axes[1].plot([index-.22, index+.22], [mid]*2, color="#252c32", lw=1.6)
    axes[1].axhline(25, color="#9b244a", linestyle="--", lw=1)
    axes[1].axhline(100, color="#777777", linestyle=":", lw=.9)
    axes[1].set_xticks(range(3), ["Remove adaptive\nlocal mixing", "Remove common-mask\nlocal mixing", "Matched random\nwrite"], fontsize=8)
    axes[1].set_ylim(0, 114)
    axes[1].set_ylabel("Unexplained final interaction (%)")
    axes[1].set_title("Local removal leaves the final\ninteraction intact", fontsize=11, pad=13)
    local = [r for r in local if r["background"] == "H" and r["layer"] == "28" and r["scope"] == "verdict"]
    global_rows = [r for r in global_rows if r["layer"] == "28" and r["scope"] == "verdict" and r["contrast"] == "direction_alone"]
    for offset, mode, color, marker, label in ((-.11, "finite", "#17798a", "o", "Same input, finite SiLU"),
        (0, "adaptive", "#6b4f91", "^", "Same input, sparse SiLU"),
        (.11, "global", "#c36532", "D", "Full evolving policies")):
        values = [float(next(r for r in (global_rows if mode == "global" else local) if r["key"] == key and (mode == "global" or r["mode"] == mode))["cosine"]) for key in keys]
        axes[2].scatter(np.arange(4)+offset, values, color=color, marker=marker, s=32, label=label, zorder=3)
    labels = []
    for key in keys:
        row = next(r for r in outcomes if r["key"] == key)
        labels.append(row["schema"]+"\n"+row["reverse"])
    axes[2].set_xticks(range(4), labels, fontsize=8)
    axes[2].set_ylim(-.27, 1.1)
    axes[2].axhline(0, color="#777777", lw=.7)
    axes[2].set_ylabel("Cosine: direction alone versus after pattern")
    axes[2].set_title("The large rotation appears across\nchanging computational states", fontsize=11, pad=13)
    axes[2].legend(frameon=False, loc="center left", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.14)
    fig.suptitle("State dependence survives removal of local MLP mixing", fontsize=15, fontweight="bold")
    fig.supxlabel("Four old contexts · 224 local input points · 28 complete forwards · qualified Float32 operations", fontsize=10)
    out = ROOT/"docs/figures/transmission"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"local-mixing.png", dpi=190, bbox_inches="tight")
    fig.savefig(out/"local-mixing.svg", bbox_inches="tight", metadata={"Date":None})
    svg = out/"local-mixing.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)
    (out/"provenance.json").write_text(json.dumps(dict(posthoc=True, all_cases=True, model_calls=0,
        matplotlib=matplotlib.__version__, source_sha256=sha(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        figure_sha256={p.name:sha(p) for p in out.glob("local-mixing.*")}), indent=2)+"\n")
    print("Rendered all fidelity decisions, all intervention cases and final same-input/global contrasts.")


if __name__ == "__main__":
    main()
