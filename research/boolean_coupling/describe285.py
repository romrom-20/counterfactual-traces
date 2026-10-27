"""Posthoc all-case figures for the frozen structural native prediction test."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("C", "L", "D", "M", "Z", "T")
COLORS = ("#247d99", "#bc8a33", "#568a70", "#7372ae", "#87919d", "#245c44")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run, out):
    assert not out.exists()
    audit = json.loads((run / "native-audit.json").read_text())
    assert audit["status"] == "passed" and audit["experiment"] == 285
    design = json.loads((run / "design.json").read_text())
    for n, h in audit["csv_sha256"].items():
        assert sha(run / n) == h

    def rows(n):
        with (run / (n + ".csv")).open() as f:
            return list(csv.DictReader(f))

    local, native, effect = rows("local"), rows("native"), rows("effect")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    values = [[100 * float(r["relative"]) for r in local if r["arm"] == a] for a in ARMS]
    assert all(len(v) == 32 for v in values)
    plotted = [[max(v, 0.1) for v in group] for group in values]
    box = ax.boxplot(plotted, tick_labels=ARMS, patch_artist=True, showfliers=False)
    for i, (v, color, patch) in enumerate(zip(plotted, COLORS, box["boxes"], strict=True)):
        patch.set(facecolor=color, alpha=0.7)
        ax.scatter(np.linspace(i + 0.88, i + 1.12, 32), v, s=11, color=color, alpha=0.6)
    ax.axhline(10, color="#777", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set(
        title="Native first-source write · all 32 cases",
        ylabel="Relative write error (%) · log scale",
    )
    x = np.arange(len(ARMS))
    ax = axes[0, 1]
    ax.bar(
        x - 0.18,
        [audit["native_counts"][a] for a in ARMS],
        0.36,
        color="#247d99",
        label="Native final response",
    )
    ax.bar(
        x + 0.18,
        [audit["H1_counts"][a] for a in ARMS],
        0.36,
        color="#bc8a33",
        label="H1: native + source write",
    )
    ax.axhline(24, color="#777", linestyle="--", linewidth=1)
    ax.set(
        xticks=x,
        xticklabels=ARMS,
        ylim=(0, 34),
        ylabel="Cases passing / 32",
        title="Does the fixed source rule transfer?",
    )
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax.text(
        0.5,
        0.97,
        "D/M: dialogue 4/8 (<6 required)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#555",
    )
    ax = axes[1, 0]
    by_key = {r["key"]: r for r in effect if r["arm"] == "T"}
    layouts = sorted({c["group"]["block"] for c in design["cases"]})
    layout_colors = ("#247d99", "#bc8a33", "#568a70", "#7372ae")
    for i, (layout, color) in enumerate(zip(layouts, layout_colors, strict=True)):
        keys = [c["key"] for c in design["cases"] if c["group"]["block"] == layout]
        y = [float(by_key[k]["target_margin"]) for k in keys]
        ax.scatter(np.linspace(i - 0.17, i + 0.17, 8), y, color=color, s=24, label=layout)
    for level in (-0.02, 0.02):
        ax.axhline(level, color="#777", linestyle="--", linewidth=1)
    ax.axhspan(-0.02, 0.02, color="#dde1e5", alpha=0.45)
    ax.axhline(0, color="#aaa", linewidth=0.7)
    ax.set(
        xticks=range(4),
        xticklabels=layouts,
        ylabel="Actual native T − matched donor M (nats)",
        title="Is direction regeneration behaviorally material?",
    )
    ax = axes[1, 1]
    ax.bar(
        x - 0.24,
        [audit["repair_counts"][a] for a in ARMS],
        0.24,
        color="#568a70",
        label="Local native repair",
    )
    ax.bar(
        x,
        [audit["effect_counts"][a] for a in ARMS],
        0.24,
        color="#7372ae",
        label="Continued material effect",
    )
    ax.bar(x + 0.24, [audit["H2_counts"][a] for a in ARMS], 0.24, color="#bc8a33", label="H2 joint")
    ax.axhline(24, color="#777", linestyle="--", linewidth=1)
    ax.set(
        xticks=x,
        xticklabels=ARMS,
        ylim=(0, 34),
        ylabel="Cases passing / 32",
        title="Local precision and native effect are separate",
    )
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.suptitle(
        "Native source prediction across four prompt structures",
        x=0.065,
        ha="left",
        weight="bold",
        fontsize=16,
    )
    fig.text(
        0.065,
        0.93,
        "32 new inputs / 16 dependent schema pairs · forecasts and qualification sealed before native True",
        fontsize=10,
        color="#444",
    )
    fig.text(
        0.065,
        0.025,
        "C: cubic · L: affine · D: historical donor · M: donor with own C forecast norm · Z: False write · T: native calibration.\nAll cases retained; H1/H2 also require 6/8 per layout. H2 requires |native T−M margin| ≥ .02 nat. Zero write errors plotted at .1%.",
        fontsize=8.5,
        color="#444",
    )
    fig.subplots_adjust(top=0.85, bottom=0.15, wspace=0.3, hspace=0.4)
    out.mkdir(parents=True)
    for ext in ("png", "svg"):
        fig.savefig(
            out / ("native-structure." + ext),
            dpi=170,
            bbox_inches="tight",
            metadata={"Creator": "Counterfactual Traces"},
        )
    plt.close(fig)
    p = out / "native-structure.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    summary = {
        a: {
            "local_median_relative": float(np.median(values[i])) / 100,
            "native_median_verdict_relative": float(
                np.median([float(r["verdict_relative"]) for r in native if r["arm"] == a])
            ),
            "native_median_margin_error": float(
                np.median([float(r["margin_error"]) for r in native if r["arm"] == a])
            ),
        }
        for i, a in enumerate(ARMS)
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                posthoc=True,
                model_forwards=0,
                predictions_modified=False,
                script_sha256=sha(Path(__file__)),
                audit_sha256=sha(run / "native-audit.json"),
                csv_sha256={
                    n + ".csv": sha(run / (n + ".csv")) for n in ("local", "native", "effect")
                },
                figures_sha256={
                    p.name: sha(p)
                    for p in out.iterdir()
                    if p.suffix in ("png", "svg", ".png", ".svg")
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    main(a.run.resolve(), a.out.resolve())
