"""Posthoc283/284 all-case prediction and signed intervention figures; no models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit229 import sha

ROOT = Path(__file__).resolve().parents[2]


def main(experiment):
    dest = ROOT / (
        "results/carried-state-response-283"
        if experiment == 283
        else "results/carried-state-forecast-284"
    )
    figdir = ROOT / f"docs/figures/carried-response/{experiment}"
    assert not figdir.exists()
    figdir.mkdir(parents=True)
    audit_name = "audit.json" if experiment == 283 else "native-audit.json"
    audit = json.loads((dest / audit_name).read_text())
    assert audit["status"] == "passed"

    def rows(name):
        with (dest / (name + ".csv")).open() as stream:
            return list(csv.DictReader(stream))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    arms = ["B", "K", "P", "H", "J", "G0", "G1", "G2", "N", "U"]
    colors = [
        "#87919d",
        "#7372ae",
        "#247d99",
        "#bc8a33",
        "#568a70",
        "#bbc1c9",
        "#bbc1c9",
        "#bbc1c9",
        "#a86b8b",
        "#245c44",
    ]
    summary = {}
    for ax, dataset, title in (
        (axes[0, 0], "incremental", "Increment beyond carry · 32 cases"),
        (axes[0, 1], "paired", "Paired source-write contrast · 16 pairs"),
    ):
        data = rows(dataset)
        actual = [[100 * float(r["relative"]) for r in data if r["arm"] == arm] for arm in arms]
        plotted = [[max(value, 0.1) for value in group] for group in actual]
        box = ax.boxplot(plotted, tick_labels=arms, patch_artist=True, showfliers=False)
        for i, (values, color, patch) in enumerate(zip(plotted, colors, box["boxes"], strict=True)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            ax.scatter(
                np.linspace(i + 0.86, i + 1.14, len(values)),
                values,
                s=10,
                color=color,
                alpha=0.65,
                zorder=3,
            )
        ax.axhline(25, color="#777", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel("Relative write error (%) · log scale")
        ax.set_title(title, loc="left", weight="bold")
        summary[dataset] = {
            arm: float(np.median(values)) / 100 for arm, values in zip(arms, actual, strict=True)
        }
    ax = axes[1, 0]
    x = np.arange(len(arms))
    ax.bar(
        x - 0.18,
        [audit["native_counts"][a] for a in arms],
        0.36,
        color="#247d99",
        label="Original native criterion",
    )
    ax.bar(
        x + 0.18,
        [audit["joint_counts"][a] for a in arms],
        0.36,
        color="#bc8a33",
        label="Full local + native account",
    )
    ax.axhline(24, color="#777", linestyle="--", linewidth=1)
    ax.set(xticks=x, xticklabels=arms, ylim=(0, 34), ylabel="Cases passing / 32")
    ax.set_title("Native accuracy alone is less discriminating", loc="left", weight="bold")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax = axes[1, 1]
    data = rows("ledger")
    components = ["gate", "up", "installation_or_downstream_nonadditivity", "unpredicted_remainder"]
    labels = [
        "Gate contribution",
        "Up contribution",
        "Rounding / later nonadditivity",
        "Unpredicted remainder",
    ]
    for j, (component, label, color) in enumerate(
        zip(components, labels, ["#bc8a33", "#568a70", "#87919d", "#a86b8b"], strict=True)
    ):
        medians = []
        for i, quantity in enumerate(("write", "receiving")):
            values = [
                float(r["signed_projection"])
                for r in data
                if r["quantity"] == quantity and r["component"] == component
            ]
            medians.append(float(np.median(values)))
            ax.scatter(
                np.linspace(i + (j - 1.5) * 0.18 - 0.04, i + (j - 1.5) * 0.18 + 0.04, len(values)),
                values,
                s=9,
                color=color,
                alpha=0.5,
                zorder=3,
            )
        ax.bar(np.arange(2) + (j - 1.5) * 0.18, medians, 0.16, label=label, color=color, alpha=0.75)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.axhline(1, color="#777", linestyle="--", linewidth=0.8)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Local write increment", "Receiving increment"],
        ylabel="Signed projection onto measured increment",
    )
    ax.set_title("Generated terms and their actual continuation", loc="left", weight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    title = "A local response predicts the missing source2 update"
    subtitle = (
        "283 · 32 OLD contexts / 16 dependent pairs · retrospective intervention test"
        if experiment == 283
        else "284 · 32 FRESH literals / 16 dependent pairs · forecasts + qualification public before native True"
    )
    fig.suptitle(title, fontsize=17, weight="bold", ha="left", x=0.07)
    fig.text(0.07, 0.93, subtitle, fontsize=10, color="#444")
    fig.text(
        0.07,
        0.025,
        "K: carry · P: combined derivative · H/J: gate/up only · G: matched random increments · N: full minus predicted increment · U: full calibration.\nAll cases retained; joint criteria include cosines/materiality. Zero errors plotted at 0.1%. Signed medians are not causal shares and need not add.",
        fontsize=8.3,
        color="#444",
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.43, wspace=0.28)
    for ext in ("png", "svg"):
        fig.savefig(
            figdir / ("carried-response." + ext),
            dpi=170,
            bbox_inches="tight",
            metadata={"Creator": "Counterfactual Traces"},
        )
    plt.close(fig)
    p = figdir / "carried-response.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    (figdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (figdir / "provenance.json").write_text(
        json.dumps(
            dict(
                experiment=experiment,
                posthoc=True,
                model_forwards=0,
                predictions_modified=False,
                script_sha256=sha(Path(__file__)),
                audit_sha256=sha(dest / audit_name),
                csv_sha256={
                    n + ".csv": sha(dest / (n + ".csv"))
                    for n in ("incremental", "paired", "native", "joint", "ledger")
                },
                figure_sha256={
                    p.name: sha(p) for p in figdir.iterdir() if p.suffix in (".png", ".svg")
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, choices=(283, 284), required=True)
    main(p.parse_args().experiment)
