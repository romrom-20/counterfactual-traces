"""All-case scientific plots of the conditional receiving-coefficient factorial."""

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
DEST = ROOT / "results/receiving-coefficients-291"
OUT = ROOT / "docs/figures/receiving-coefficients-291"
ARMS = (*tuple(f"{j:03b}" for j in range(8)), "R0", "R1", "R2")
LABELS = (
    "000 baseline",
    "001 up coefficient",
    "010 up multiplier",
    "011 both up terms",
    "100 gate response",
    "101 gate + coefficient",
    "110 gate + multiplier",
    "111 full calibration",
    "R0 shuffled",
    "R1 shuffled",
    "R2 shuffled",
)
COLORS = (
    "#929b9e",
    "#697ea1",
    "#bd8b4b",
    "#866c95",
    "#39938c",
    "#207984",
    "#477554",
    "#233f54",
    "#b2a8a0",
    "#b2a8a0",
    "#b2a8a0",
)


def rows(name):
    with (DEST / (name + ".csv")).open() as source:
        return list(csv.DictReader(source))


def main(out):
    assert not out.exists(), "Use a fresh rendering directory"
    out.mkdir(parents=True)
    audit = read(DEST / "audit.json")
    assert audit["status"] == "passed"
    h, v, native = (rows(n) for n in ("hypotheses", "receiving_vectors", "native_scalar"))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axes[0, 0]
    finite = [sum(r["finite_pass"] == "True" for r in h if r["arm"] == a) for a in ARMS]
    joint = [audit["arms"][a]["passes"] for a in ARMS]
    ax.barh(range(len(ARMS)), finite, height=0.6, color="#d8ddde", label="Finite effect only")
    ax.scatter(joint, range(len(ARMS)), c=COLORS, s=35, zorder=3, label="Joint gradient + effect")
    for j, n in enumerate(joint):
        ax.text(n + 0.5, j - 0.16, str(n), fontsize=8)
    ax.axvline(24, ls="--", lw=1, color=".55")
    ax.set(
        yticks=range(len(ARMS)),
        yticklabels=LABELS,
        ylim=(10.8, -0.8),
        xlim=(-0.7, 34),
        xlabel="Cases passed (of 32)",
        title="Which coefficient changes suffice?",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ax, data, title in [
        (axes[0, 1], h, "Boundary-gradient contrast"),
        (axes[1, 0], v, "Signed receiving-displacement contrast"),
    ]:
        for j, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
            vals = [float(r["relative"]) for r in data if r["arm"] == arm]
            ax.scatter(j + np.linspace(-0.15, 0.15, len(vals)), vals, s=12, alpha=0.6, color=color)
            ax.plot([j - 0.2, j + 0.2], [statistics.median(vals)] * 2, lw=1, color=".2")
        ax.axhline(0.25, ls="--", lw=1, color=".55")
        ax.set_yscale("linear")
        ax.set(
            xticks=range(len(ARMS)),
            xticklabels=ARMS,
            ylabel="Relative vector error (ratio)",
            xlabel="Gate response / up multiplier / up coefficient",
            title=title,
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(bottom=0)
    ax = axes[1, 1]
    for j, (arm, color) in enumerate(zip(ARMS, COLORS, strict=True)):
        vals = [float(r["error"]) for r in native if r["arm"] == arm]
        ax.scatter(j + np.linspace(-0.15, 0.15, len(vals)), vals, s=14, alpha=0.65, color=color)
        ax.plot([j - 0.2, j + 0.2], [statistics.median(vals)] * 2, color=".2", lw=1)
    ax.axhline(0.1, color=".5", ls="--", lw=1)
    ax.set(
        xticks=range(len(ARMS)),
        xticklabels=ARMS,
        xlabel="Gate response / up multiplier / up coefficient",
        ylabel="Native scalar calibration error (nat)",
        title="Scalar accuracy can conceal vector error",
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(bottom=0)
    fig.suptitle("How the receiving MLP changes its response", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.012,
        "32 reused contexts · source/attention/head supplied · receiving norms False · 111 is calibration",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96), h_pad=2.3, w_pad=2.5)
    for suffix in ("png", "svg"):
        p = out / ("receiving-coefficients." + suffix)
        fig.savefig(p, dpi=170)
        if suffix == "svg":
            p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    plt.close(fig)
    summary = dict(
        experiment=291,
        retrospective=True,
        cases=32,
        primary_pass=audit["primary_pass"],
        arms=audit["arms"],
        required_checks=audit["required_checks"],
        original_coordinate_failures=audit["original_coordinate_failures"],
        general_question_B_solved=False,
        statistics={
            a: {
                f"median_{name}": statistics.median(float(r[field]) for r in h if r["arm"] == a)
                for name, field in [
                    ("relative", "relative"),
                    ("cosine", "cosine"),
                    ("finite_error", "finite_error"),
                ]
            }
            for a in ARMS
        },
    )
    write(out / "summary.json", summary)
    write(
        out / "provenance.json",
        dict(
            experiment=291,
            all_32_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / (n + ".csv")).relative_to(ROOT)): sha(DEST / (n + ".csv"))
                for n in ("hypotheses", "receiving_vectors", "native_scalar")
            }
            | {str((DEST / "audit.json").relative_to(ROOT)): sha(DEST / "audit.json")},
            outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out.resolve())
