"""All-case plots of the receiving nonlinear-response test."""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/receiving-nonlinearity-290"
OUT = ROOT / "docs/figures/receiving-nonlinearity-290"
ARMS = tuple(f"{j:02b}" for j in range(4))


def rows(name):
    with (DEST / (name + ".csv")).open() as source:
        return list(csv.DictReader(source))


def main(out):
    assert not out.exists(), "Choose a fresh rendering directory"
    out.mkdir(parents=True)
    audit = read(DEST / "audit.json")
    assert audit["status"] == "passed"
    h, v, n = (rows(name) for name in ("hypotheses", "receiving_vectors", "native_scalar"))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3))
    ax = axes[0, 0]
    finite = [sum(r["finite_pass"] == "True" for r in h if r["arm"] == arm) for arm in ARMS]
    joint = [audit["arms"][arm]["passes"] for arm in ARMS]
    ax.barh(range(4), finite, color="#c4ccce", height=0.6, label="Finite effect only")
    ax.scatter(joint, range(4), color="#176b73", s=36, label="Joint gradient + effect", zorder=3)
    for j, count in enumerate(joint):
        ax.text(count + 0.5, j - 0.23, str(count), fontsize=8, color="#176b73")
    ax.axvline(24, color=".5", ls="--", lw=1)
    ax.set(
        yticks=range(4),
        yticklabels=("00 baseline", "01 gated MLP", "10 normalization", "11 all-E calibration"),
        ylim=(3.8, -0.8),
        xlim=(-0.8, 34),
        xlabel="Cases passed (of 32)",
        ylabel="Receiving norm / gated MLP bits",
        title="Normalization or gated MLP response?",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax, data, field, bound, ylabel, title in (
        (
            axes[0, 1],
            h,
            "relative",
            0.25,
            "Gradient-gap relative error (ratio)",
            "Every gradient comparison",
        ),
        (
            axes[1, 0],
            v,
            "relative",
            0.25,
            "Receiving-correction error (ratio)",
            "Secondary receiving-vector recovery",
        ),
        (
            axes[1, 1],
            n,
            "error",
            0.1,
            "Native donor scalar error (nat)",
            "Scalar calibration is a different target",
        ),
    ):
        for j, arm in enumerate(ARMS):
            rr = [r for r in data if r["arm"] == arm]
            vals = [float(r[field]) for r in rr]
            ax.scatter(
                j + np.linspace(-0.15, 0.15, len(vals)),
                vals,
                s=14,
                alpha=0.65,
                color={"00":"#8e979c", "01":"#176b73", "10":"#a66b45", "11":"#637480"}[arm],
            )
            ax.plot([j - 0.2, j + 0.2], [statistics.median(vals)] * 2, color=".2", lw=1)
        ax.axhline(bound, color=".5", ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set(
            xticks=range(4),
            xticklabels=ARMS,
            xlabel="Receiving norm / gated MLP bits",
            ylabel=ylabel,
            title=title,
        )
        ax.set_ylim(bottom=0)
    fig.suptitle("Receiving normalization and gated MLP response", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.012,
        "32 reused contexts · source/head/attention edited · 11 is an all-E calibration · all cases retained",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.955), h_pad=2, w_pad=2)
    for suffix in ("png", "svg"):
        path = out / ("receiving-nonlinearity." + suffix)
        fig.savefig(path, dpi=170)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    summary = dict(
        experiment=290,
        retrospective=True,
        cases=32,
        primary_pass=audit["primary_pass"],
        arms=audit["arms"],
        required_checks=audit["required_checks"],
        original_coordinate_failures=audit["original_coordinate_failures"],
        general_question_B_solved=False,
        statistics={
            arm: {
                "median_relative": statistics.median(
                    float(r["relative"]) for r in h if r["arm"] == arm
                ),
                "median_cosine": statistics.median(
                    float(r["cosine"]) for r in h if r["arm"] == arm
                ),
                "median_finite_error": statistics.median(
                    float(r["finite_error"]) for r in h if r["arm"] == arm
                ),
            }
            for arm in ARMS
        },
    )
    write(out / "summary.json", summary)
    write(
        out / "provenance.json",
        dict(
            experiment=290,
            all_32_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / (name + ".csv")).relative_to(ROOT)): sha(DEST / (name + ".csv"))
                for name in ("hypotheses", "receiving_vectors", "native_scalar")
            }
            | {str((DEST / "audit.json").relative_to(ROOT)): sha(DEST / "audit.json")},
            outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.out.resolve())
