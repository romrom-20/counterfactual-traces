"""Posthoc all-case figure separating268's interaction test from269 native prediction."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    inputs = [ROOT/"results/amplitude-direction-268-269/amplitude-direction-268/factorial.csv",
              ROOT/"results/signed-amplitude-forecast-269/native-outcomes.csv"]
    old, fresh = map(rows, inputs)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "question-b-amplitude268269"})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.1), layout="constrained", gridspec_kw={"width_ratios": [1, 1.35, 1.35]})
    groups = ("10", "01", "00", "r")
    labels = ("Amplitude\nonly", "Direction\nonly", "Both\nfrozen", "Random\nD axis")
    for index, family in enumerate(groups):
        selected = [r for r in old if r["family"] == family]
        for offset, row in zip(np.linspace(-.16, .16, 4), selected, strict=True):
            if row["complete"] == "True":
                axes[0].scatter(index+offset, 100*float(row["all_relative"]), s=34,
                    color="#187e8c" if row["schema"] == "generic" else "#c26739",
                    marker="o" if row["reverse"] == "0" else "s", zorder=3)
        axes[0].text(index, 1.01, f"{sum(r['passes']=='True' for r in selected)}/4 joint",
                     ha="center", transform=axes[0].get_xaxis_transform(), fontsize=9)
    axes[0].axhline(25, color="#a22e54", lw=1, ls="--")
    axes[0].set_xticks(range(4), labels, fontsize=9)
    axes[0].set_ylabel("Whole policy-interaction error (%)")
    axes[0].set_title("Old-context interaction sufficiency", pad=24)
    arms = ("L", "H", "A11", "A10", "A01", "A00", "Ar")
    names = ("L seed", "Finite H", "Live D", "Amplitude", "Direction", "Frozen", "Random")
    for index, arm in enumerate(arms):
        selected = [r for r in fresh if r["arm"] == arm]
        assert len(selected) == 32
        for offset, row in zip(np.linspace(-.22, .22, 32), selected, strict=True):
            if row["status"] != "complete":
                continue
            color = "#187e8c" if row["schema"] == "generic" else "#c26739"
            for ax, field, multiplier in ((axes[1], "all_relative", 100), (axes[2], "margin_error", 1)):
                ax.scatter(index+offset, multiplier*float(row[field]), color=color, s=15, alpha=.65, zorder=3)
        axes[1].text(index, 1.01, f"{sum(r['passes']=='True' for r in selected)}/32",
                     ha="center", transform=axes[1].get_xaxis_transform(), fontsize=8)
        for ax, field, multiplier in ((axes[1], "all_relative", 100), (axes[2], "margin_error", 1)):
            values = [multiplier*float(r[field]) for r in selected if r["status"] == "complete"]
            if values:
                ax.plot([index-.24, index+.24], [np.median(values)]*2, color="#252c32", lw=1.5)
    axes[1].axhline(5, color="#a22e54", lw=1, ls="--")
    axes[1].set_yscale("symlog", linthresh=5)
    axes[1].set_yticks((0, 2, 5, 10, 25, 50), ("0", "2", "5", "10", "25", "50"))
    axes[1].set_ylabel("Native receiving-displacement error (%)")
    axes[1].set_title("32 globally sealed fresh predictions", pad=24)
    axes[2].axhline(.1, color="#a22e54", lw=1, ls="--")
    axes[2].set_yscale("symlog", linthresh=.1)
    axes[2].set_yticks((0, .05, .1, .3, 1, 3), ("0", ".05", ".1", ".3", "1", "3"))
    axes[2].set_ylabel("Finite verdict-margin error (nats)")
    axes[2].set_title("Scalar accuracy is checked separately", pad=24)
    for ax in axes[1:]:
        ax.set_xticks(range(7), names, rotation=30, ha="right", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Signed amplitudes, changing directions, and what the prediction actually tests", fontsize=14, fontweight="bold")
    fig.supxlabel("268: four old contexts · 269: 32 new literal pairs in dependent templates · teal: generic, orange: descriptive · joint counts include both vectors and margin", fontsize=9)
    out = ROOT/"docs/figures/amplitude"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"amplitude-prediction.png", dpi=190, bbox_inches="tight")
    fig.savefig(out/"amplitude-prediction.svg", bbox_inches="tight", metadata={"Date": None})
    path = out/"amplitude-prediction.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)
    (out/"provenance.json").write_text(json.dumps(dict(posthoc=True, all_cases=True, model_calls=0,
        source_sha256=sha(Path(__file__)), matplotlib=matplotlib.__version__,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        figure_sha256={p.name: sha(p) for p in out.glob("amplitude-prediction.*")}), indent=2)+"\n")
    print("Rendered all old/fresh cases; forecast accuracy is separate from interaction sufficiency.")


if __name__ == "__main__":
    main()
