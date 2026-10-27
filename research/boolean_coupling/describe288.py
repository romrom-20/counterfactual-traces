"""Plot every retained 288 context, including failed accounts and signed interactions."""

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
DEST = ROOT / "results/tangent-routing-288"
OUT = ROOT / "docs/figures/tangent-routing-288"
NAMES = ("SOURCE", "RECEIVING_AND_HEAD", "RECEIVING_BLOCKS", "HEAD", "ADD")
LABELS = ("Source\nrecurrence", "Receiving\n+ head", "Receiving\nblocks", "Final\nhead", "Additive\nsecondary")
COLORS = ("#176b73", "#b45a3b", "#324c69", "#7b6b88", "#858b8d")


def rows(name):
    with (DEST / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def main(out):
    assert not out.exists(), "Choose a new output; preserve previous renderings"
    out.mkdir(parents=True)
    audit = read(DEST / "audit.json")
    assert audit["status"] == "passed"
    hypotheses, ledger = rows("hypotheses"), rows("ledger")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax = axes[0, 0]
    counts = [audit["hypotheses"][n]["passes"] for n in NAMES]
    finite_counts = [sum(r["finite_pass"] == "True" for r in hypotheses if r["hypothesis"] == name) for name in NAMES]
    ax.barh(range(5), finite_counts, color="#bec6c9", height=.52, label="Finite contrast only")
    ax.scatter(counts, range(5), color="#176b73", s=42, label="Joint vector + finite", zorder=3)
    for j, n in enumerate(finite_counts):
        ax.text(n + .6, j, str(n), va="center")
    ax.axvline(24, color=".5", linestyle="--", linewidth=1)
    ax.set(yticks=range(5), yticklabels=[label.replace("\n", " ") for label in LABELS],
           xlim=(-.7, 33), ylim=(5.1, -.6), xlabel="Cases passed (of 32)",
           title="Every joint account fails: 0/32" if not any(counts) else "Joint and finite-only success")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[0, 1]
    for j, name in enumerate(NAMES):
        rr = [r for r in hypotheses if r["hypothesis"] == name]
        ax.scatter(j + np.linspace(-.15, .15, len(rr)), [float(r["relative"]) for r in rr],
                   color=COLORS[j], s=18, alpha=.7)
    ax.axhline(.25, color=".5", linestyle="--", linewidth=1)
    ax.set(xticks=range(5), xticklabels=LABELS, ylabel="Gradient-shift relative error (ratio)",
           yscale="log", title="All 32 vector comparisons")
    ax = axes[1, 0]
    extent = max(abs(float(r[n])) for r in hypotheses for n in ("finite_target", "finite_prediction")) * 1.12
    ax.plot([-extent, extent], [-extent, extent], color=".65", linewidth=1)
    for j, name in enumerate(NAMES[:4]):
        rr = [r for r in hypotheses if r["hypothesis"] == name]
        ax.scatter([float(r["finite_target"]) for r in rr], [float(r["finite_prediction"]) for r in rr],
                   label=LABELS[j].replace("\n", " "), color=COLORS[j], s=24,
                   marker=("o", "s", "^", "x")[j], alpha=.65)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
           xlabel="Full finite sensitivity contrast (nat)", ylabel="Account's finite contrast (nat)",
           title="Check the signed effect, not only the vector")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 1]
    terms = ("100", "010", "001", "110", "101", "011", "111")
    for j, term in enumerate(terms):
        rr = [r for r in ledger if r["term"] == term]
        vals = [float(r["finite_contribution"]) for r in rr]
        ax.scatter(j + np.linspace(-.16, .16, len(vals)), vals, s=15, alpha=.65,
                   color="#324c69" if j < 3 else "#9b6550")
        ax.plot([j - .19, j + .19], [statistics.median(vals)] * 2, color=".15", linewidth=1.5)
    ax.axhline(0, color=".7", linewidth=1)
    ax.set(xticks=range(7), xticklabels=("S", "R", "H", "S×R", "S×H", "R×H", "S×R×H"),
           xlabel="Source / receiving / head interaction term", ylabel="Signed finite contribution (nat)",
           title="Cancellation stays visible")
    fig.suptitle("A changed source can alter more than its own local response", fontsize=15, y=.98)
    fig.text(.5, .012, "32 reused contexts · finite recentered networks · all cases shown · broad update maps, not a learned circuit", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .955), h_pad=2.4, w_pad=2)
    for suffix in ("png", "svg"):
        p = out / ("tangent-routing." + suffix)
        fig.savefig(p, dpi=170)
        if suffix == "svg":
            p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    plt.close(fig)
    summary = dict(experiment=288, contexts=32, retrospective=True, hypotheses=audit["hypotheses"],
                   required_numeric_checks=audit["required_checks"],
                   original_coordinate_diagnostic_failures=audit["original_coordinate_diagnostic_failures"],
                   general_question_B_solved=False,
                   statistics={name: {
                       "median_relative": statistics.median(float(r["relative"]) for r in hypotheses if r["hypothesis"] == name),
                       "median_finite_error": statistics.median(float(r["finite_error"]) for r in hypotheses if r["hypothesis"] == name),
                       "median_cosine": statistics.median(float(r["cosine"]) for r in hypotheses if r["hypothesis"] == name),
                   } for name in NAMES})
    write(out / "summary.json", summary)
    write(out / "provenance.json", dict(
        experiment=288, all_32_cases=True,
        script_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (
            DEST / "audit.json", DEST / "hypotheses.csv", DEST / "ledger.csv")},
        outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
    ))
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    main(a.out.resolve())
