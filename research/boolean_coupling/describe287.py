"""Plot every287 case; boundary-only qualification and failed full gate visible."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-receiver-readout-287"
OUT = ROOT / "docs/figures/source-receiver-readout-287"


def rows(name):
    with (DEST / "boundary-qualification" / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def main(out):
    assert not out.exists(), "Choose a new figure directory; preserve prior renderings"
    out.mkdir(parents=True)
    audit = read(DEST / "boundary-qualification/audit.json")
    hypo = [r for r in rows("hypotheses") if r["gradient_path"] == "cached"]
    native = rows("native")
    readout = [r for r in rows("readouts") if r["gradient_path"] == "cached"]
    colors = {"S": "#156c76", "R": "#bc5e39", "ADD": "#72539e"}
    labels = {"S": "Source only", "R": "Receivers only", "ADD": "Additive (secondary)"}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(14.3, 4.4), gridspec_kw={"width_ratios": [1, 1.1, 1]})
    x = np.arange(3)
    counts = [audit["hypotheses"]["cached"][a]["passes"] for a in colors]
    axs[0].bar(x, counts, color=list(colors.values()), width=0.62)
    for i, n in enumerate(counts):
        axs[0].text(i, n + 0.6, f"{n}/32", ha="center")
    axs[0].axhline(24, color=".5", ls="--", lw=1, label="Population count bound")
    axs[0].set(
        ylim=(0, 36),
        xticks=x,
        xticklabels=["Source\nonly", "Receivers\nonly", "Additive\nsecondary"],
        ylabel="Joint cases passed",
        title="Which initial rows suffice?",
    )
    for arm in colors:
        rr = [r for r in hypo if r["arm"] == arm]
        axs[1].scatter(
            [float(r["finite_target"]) for r in rr],
            [float(r["finite_prediction"]) for r in rr],
            s=38 if arm == "S" else 22,
            alpha=0.75,
            label=labels[arm],
            facecolors="none" if arm == "S" else colors[arm],
            edgecolors=colors[arm],
            marker="s" if arm == "ADD" else "o",
            zorder=3 if arm == "S" else 2,
        )
    extent = (
        max(abs(float(r[k])) for r in hypo for k in ("finite_target", "finite_prediction")) * 1.12
    )
    axs[1].plot([-extent, extent], [-extent, extent], color=".5", lw=1, zorder=0)
    axs[1].set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel="Full finite effect shift (nat)",
        ylabel="Predicted finite effect shift (nat)",
        title="Test the signed effect too",
    )
    axs[1].legend(fontsize=8, loc="upper left", frameon=False)
    positions, corner_colors = (
        dict(FF=0, MF=1, FM=2, MM=3),
        ["#80858b", "#156c76", "#bc5e39", "#25394d"],
    )
    for c, color in zip(positions, corner_colors, strict=True):
        rr = [r for r in native if r["corner"] == c]
        vals = [100 * float(r["receiving_relative"]) for r in rr]
        offsets = np.linspace(-0.16, 0.16, len(vals))
        axs[2].scatter(positions[c] + offsets, vals, s=16, alpha=0.65, color=color)
        axs[2].plot(
            [positions[c] - 0.2, positions[c] + 0.2], [np.median(vals)] * 2, color=color, lw=2
        )
    axs[2].axhline(5, color=".5", ls="--", lw=1)
    axs[2].set(
        yscale="log",
        xticks=range(4),
        xticklabels=["FF", "MF", "FM", "MM"],
        ylabel="Receiving-vector error (%)",
        title="Sensitivity is not full fidelity",
    )
    axs[2].text(
        0.02,
        0.40,
        "Same source +d in every corner",
        transform=axs[2].transAxes,
        fontsize=8,
        va="top",
    )
    fig.suptitle(
        "Changing the source state recreates the changed sensitivity",
        fontsize=15,
        x=0.035,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.035,
        0.015,
        "All 32 old contexts • source first in FF/MF/FM/MM • later receivers always adapt • original full-trace gate FAILED; boundary287b qualified",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.015, 0.055, 1, 0.94))
    for suffix in ("png", "svg"):
        path = out / ("source-receiver-readout." + suffix)
        fig.savefig(path, dpi=180)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    summary = dict(
        all_cases=32,
        hypotheses=audit["hypotheses"],
        native_counts=audit["native_counts"],
        native_scalar_counts=audit["native_scalar_counts"],
        original_numeric_gate_passed=False,
        material_eligible=audit["material_eligible"],
        classification_disagreements=audit["classification_disagreements"],
        median_receiving_relative={
            c: float(
                np.median([float(r["receiving_relative"]) for r in native if r["corner"] == c])
            )
            for c in positions
        },
        median_native_scalar_error={
            c: float(np.median([float(r["native_error"]) for r in readout if r["corner"] == c]))
            for c in positions
        },
    )
    write(out / "summary.json", summary)
    write(
        out / "provenance.json",
        dict(
            script_sha256=sha(Path(__file__)),
            original_audit_sha256=sha(DEST / "audit.json"),
            qualifier_audit_sha256=sha(DEST / "boundary-qualification/audit.json"),
            csv_sha256=audit["csv_sha256"],
            figures_sha256={p.name: sha(p) for p in sorted(out.glob("source-receiver-readout.*"))},
            all_cases=32,
            selected_or_omitted_cases=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.out.resolve())
