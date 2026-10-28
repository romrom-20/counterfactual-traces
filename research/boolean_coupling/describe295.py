"""All conditional probes, all candidates, with frozen failures visible."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/finite-drive-295"
OUT = ROOT / "docs/figures/finite-drive-295"
ARMS = [
    "CHANGE_RANGE",
    "ENDPOINT_DRIVE",
    "ORACLE_DIRECTION",
    "MIDPOINT_DRIVE",
    "SIMPSON_DRIVE",
    "INTEGRAL16",
    "SHUFFLED_DRIVE",
]
LABELS = [
    "294 range\nprojection",
    "Endpoint\ndrive",
    "Oracle\nline fit",
    "Midpoint\ndrive",
    "Simpson\ndrive",
    "Integral\ncalibration",
    "Shuffled\ndrive",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "295",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, scope in zip(axes, ["source", "verdict"], strict=True):
        for j, arm in enumerate(ARMS):
            for layer, offset, color in [
                (1, -0.16, "#c16a28"),
                (15, 0.0, "#087f8c"),
                (28, 0.16, "#66528b"),
            ]:
                values = [
                    float(r["relative"])
                    for r in rows
                    if r["arm"] == arm and r["scope"] == scope and r["layer"] == str(layer)
                ]
                ax.scatter(
                    j + offset + np.linspace(-0.035, 0.035, len(values)),
                    values,
                    color=color,
                    s=20,
                    alpha=0.7,
                    label="Block " + str(layer) if j == 0 else None,
                )
        ax.axhline(0.25, color="#222", ls="--", lw=1, label="Vector error bound .25")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2.3)
        ax.set_xticks(range(7), LABELS)
        ax.set_ylabel("Relative nonlinear-write vector error")
        ax.set_title(
            scope.capitalize() + " probes (24 points per candidate)", loc="left", fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.17)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle(
        "Actual drive helps; endpoint response still misses the finite write",
        x=0.055,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.895,
        "Primary endpoint account: 40/48, FAIL (source 16/24). Oracle direction: 41/48, FAIL (source 17/24).",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.03,
        "Eight OLD contexts; supplied P gate/up endpoints. All 48 probes retained. Midpoint/Simpson: 48/48, descriptive.\nIntegral is known-function calibration; oracle line fit uses the target. No native forecast, causal repair or learned circuit.",
        fontsize=10,
        color="#444",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.80, wspace=0.20)
    for ext in ["png", "svg"]:
        fig.savefig(
            OUT / ("finite-drive." + ext), dpi=175, metadata={"Date": None} if ext == "svg" else {}
        )
    plt.close(fig)
    p = OUT / "finite-drive.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    write(
        OUT / "provenance.json",
        dict(
            experiment=295,
            all_48_probes=True,
            script_sha256=sha(Path(__file__)),
            inputs={str((DEST / "metrics.csv").relative_to(ROOT)): sha(DEST / "metrics.csv")},
            outputs={p.name: sha(p) for p in sorted(OUT.glob("finite-drive.*"))},
        ),
    )


if __name__ == "__main__":
    main()
