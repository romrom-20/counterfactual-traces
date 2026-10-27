"""Posthoc all-case plots for the scalar readout diagnostic; no model calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main(run, qualifier, out):
    assert not out.exists()
    audit, qaudit = read(run / "audit.json"), read(qualifier / "audit.json")
    assert audit["status"] == qaudit["status"] == "passed"
    for name, h in audit["csv_sha256"].items():
        assert sha(run / name) == h

    def rows(name):
        with (run / (name + ".csv")).open() as stream:
            return list(csv.DictReader(stream))

    predictions, interventions, geometry = (
        rows("predictions"),
        rows("interventions"),
        rows("geometry"),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "false-readout-286",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    colors = {"F": "#ba5c48", "M": "#297f95"}
    names = {"F": "False-state gradient", "M": "Forecast-state gradient"}
    ax = axes[0, 0]
    for anchor, marker in (("F", "x"), ("M", "o")):
        rr = [r for r in predictions if r["anchor"] == anchor]
        ax.scatter(
            [float(r["target_effect"]) for r in rr],
            [float(r["prediction"]) for r in rr],
            label=names[anchor],
            color=colors[anchor],
            marker=marker,
            s=28,
            alpha=0.8,
        )
    ax.plot([-0.2, 0.2], [-0.2, 0.2], color="#777", linewidth=0.8)
    ax.axhline(0, color="#ddd", linewidth=0.6)
    ax.axvline(0, color="#ddd", linewidth=0.6)
    ax.set(
        title="Same source correction, different sensitivity",
        xlabel="Native T − M margin (nats)",
        ylabel="Gradient prediction (nats)",
        xlim=(-0.2, 0.2),
        ylim=(-0.2, 0.2),
    )
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    for i, anchor in enumerate(("F", "M")):
        values = [float(r["error"]) for r in predictions if r["anchor"] == anchor]
        ax.scatter(
            np.linspace(i - 0.13, i + 0.13, len(values)),
            values,
            s=25,
            color=colors[anchor],
            alpha=0.75,
        )
        med = float(np.median(values))
        ax.plot([i - 0.22, i + 0.22], [med, med], color=colors[anchor], linewidth=2.5)
        ax.text(i + 0.24, med, f"{med:.4f}", va="center", fontsize=9, color=colors[anchor])
    ax.axhline(0.1, color="#777", linestyle="--", linewidth=1)
    ax.text(
        0.98,
        0.102,
        ".1-nat bound",
        transform=ax.get_yaxis_transform(),
        ha="right",
        fontsize=8,
        color="#555",
    )
    ax.set(
        title="False sensitivity misses all six large errors",
        ylabel="Absolute native margin prediction error (nats)",
        xticks=[0, 1],
        xticklabels=["False: 26/32", "Forecast: 32/32"],
        ylim=(0, 0.195),
        xlim=(-0.4, 1.5),
    )
    ax = axes[1, 0]
    arms = ("parallel_F", "orthogonal_F", "parallel_M", "orthogonal_M", "random0", "random1")
    labels = ("F ∥", "F ⊥", "M ∥", "M ⊥", "Rand 1", "Rand 2")
    counts = [
        sum(r["finite_material_pass"] == "True" for r in interventions if r["arm"] == arm)
        for arm in arms
    ]
    eligible = sum(
        r["finite_material_eligible"] == "True" for r in interventions if r["arm"] == "parallel_M"
    )
    ax.bar(
        range(6),
        counts,
        color=[colors["F"], colors["F"], colors["M"], colors["M"], "#90979e", "#90979e"],
    )
    for i, count in enumerate(counts):
        ax.text(i, count + 0.5, str(count), ha="center", fontsize=9)
    ax.axhline(eligible, color="#777", linewidth=0.8, linestyle="--")
    ax.set(
        title="Direct source interventions at forecast state M",
        xticks=range(6),
        xticklabels=labels,
        ylabel="Material finite C − M effects reproduced / 32",
        ylim=(0, 34),
    )
    ax.text(
        0.98,
        0.94,
        f"Only {eligible}/32 effects reach .02 nat",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555",
    )
    ax = axes[1, 1]
    fractions = {}
    for i, anchor in enumerate(("F", "M")):
        values = [
            100 * float(r["component_norm_fraction"])
            for r in interventions
            if r["arm"] == "parallel_" + anchor
        ]
        fractions[anchor] = float(np.median(values)) / 100
        ax.scatter(
            np.linspace(i - 0.12, i + 0.12, len(values)),
            values,
            color=colors[anchor],
            s=25,
            alpha=0.75,
        )
        med = float(np.median(values))
        ax.plot([i - 0.2, i + 0.2], [med, med], color=colors[anchor], linewidth=2.5)
        ax.text(i + 0.22, med, f"{med:.2f}%", color=colors[anchor], va="center", fontsize=9)
    cosine = float(np.median([float(r["gradient_cosine"]) for r in geometry]))
    ax.set(
        title="A small projection can carry a scalar effect",
        xticks=[0, 1],
        xticklabels=["Along False gradient", "Along forecast gradient"],
        ylabel="Projection norm / total source correction (%)",
        xlim=(-0.35, 1.6),
        ylim=(0, 8),
    )
    ax.text(
        0.98,
        0.95,
        f"Median gradient cosine: {cosine:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555",
    )
    fig.suptitle(
        "The readout changes with the edited computational state",
        x=0.065,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.065,
        0.93,
        "32 already observed contexts · scalar margin diagnostic and direct source-residual interventions · no fresh forecast",
        fontsize=10,
        color="#444",
    )
    fig.text(
        0.065,
        0.025,
        "F/M: gradients at False / forecast incoming suffix states. Parallel/orthogonal components are applied at M, not at F.\nRandom controls match the F-parallel norm. All 32 cases retained. Material error bound: .005 + .25 × |finite effect|.\nFull-shape adjoints pass a separate 1e-4 check; 192 original cached-coordinate diagnostics remain failed.",
        fontsize=8.5,
        color="#444",
    )
    fig.subplots_adjust(top=0.85, bottom=0.16, hspace=0.43, wspace=0.31)
    out.mkdir(parents=True)
    for extension in ("png", "svg"):
        fig.savefig(
            out / ("state-dependent-readout." + extension),
            dpi=170,
            bbox_inches="tight",
            metadata={"Creator": "Counterfactual Traces", "Date": None},
        )
    plt.close(fig)
    svg = out / "state-dependent-readout.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    summary = dict(
        experiment=286,
        retrospective=True,
        all_cases=32,
        predictions=audit["predictions"],
        median_gradient_cosine=cosine,
        median_parallel_norm_fraction=fractions,
        finite_material_counts=dict(zip(arms, counts, strict=True)),
        finite_material_eligible=eligible,
        full_shape_qualification=qaudit,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                posthoc=True,
                model_forwards=0,
                predictions_modified=False,
                script_sha256=sha(Path(__file__)),
                audit_sha256=sha(run / "audit.json"),
                qualifier_audit_sha256=sha(qualifier / "audit.json"),
                csv_sha256=audit["csv_sha256"],
                figures_sha256={
                    p.name: sha(p) for p in out.iterdir() if p.suffix in (".png", ".svg")
                },
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualifier", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.qualifier.resolve(), args.out.resolve())
