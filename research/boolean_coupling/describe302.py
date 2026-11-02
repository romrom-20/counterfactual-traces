"""Scientific figures for every 302 case; failed accounts and references retained."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import sha, write

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/first-source-geometry-302"
OUT = ROOT / "docs/figures/first-source-geometry-302"


def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    svg = OUT / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "302",
        }
    )
    effect, phase = rows("effects.csv"), rows("phase.csv")
    arms = ["K", "V", "H", "S", "RH", "RK", "RV"]
    labels = ["Keys", "Values", "Hidden", "Seen", "R.hidden", "R.keys", "R.values"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    for ax, fields, title in [
        (axs[0, 0], ["source_relative", "emit_relative"], "Local source and next emission"),
        (
            axs[0, 1],
            ["receiving_relative", "verdict_relative"],
            "Conditional finite downstream effects",
        ),
    ]:
        for j, field in enumerate(fields):
            for i, arm in enumerate(arms):
                rr = [r for r in effect if r["arm"] == arm]
                y = [max(float(r["A_" + field]), float(r["B_" + field])) for r in rr]
                ax.scatter(
                    i + np.linspace(-0.13, 0.13, 8) + (j - 0.5) * 0.16,
                    y,
                    s=22,
                    marker="o" if j == 0 else "x",
                    color="#285f87" if j == 0 else "#72777c",
                    alpha=0.85,
                    label=field.replace("_relative", "") if i == 0 else None,
                )
        ax.axhline(0.25, color="#a33b35", ls="--", lw=1, label="Frozen 25% bound")
        ax.set_xticks(range(len(arms)), labels, rotation=25)
        ax.set_ylabel("Relative vector error (worse reference)")
        ax.set_xlabel("Intervention component")
        ax.set_title(title)
        ax.legend(fontsize=8)
    ax = axs[1, 0]
    rr = [r for r in effect if r["arm"] == "K"]
    for alias, marker, col in [("A", "o", "#285f87"), ("B", "x", "#72777c")]:
        ax.scatter(
            np.arange(8),
            [float(r[alias + "_target_margin"]) for r in rr],
            marker=marker,
            color=col,
            label="Reference " + alias,
        )
    ax.axhspan(-0.02, 0.02, color="#eeeeee", label="Below .02-nat materiality")
    ax.axhline(0, color="#8a8a8a", lw=0.7)
    ax.set_xticks(range(8), [r["layout"] + "/" + r["mention_order"] for r in rr], rotation=30)
    ax.set_ylabel("KV-minus-native Boolean coupling (nats)")
    ax.set_xlabel("All eight cases (layout / mention order)")
    ax.set_title("The conditional target depends on reference")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    ax.scatter(range(8), [float(r["remaining_norm_ratio"]) for r in phase], color="#285f87", s=32)
    ax.axhline(0.25, color="#a33b35", ls="--", lw=1, label="Phase-dominant requirement")
    ax.axhline(1, color="#8a8a8a", lw=0.7, label="Native contrast norm")
    ax.set_xticks(range(8), [r["layout"] + "/" + r["mention_order"] for r in phase], rotation=30)
    ax.set_xlabel("All eight cases (layout / mention order)")
    ax.set_ylabel("Remaining source-contrast norm / native norm")
    ax.set_ylim(0, 1.08)
    ax.set_title("Fixed-center phase reorientation leaves most contrast")
    ax.legend(fontsize=8)
    fig.suptitle(
        "302: keys, values and geometry each fail the complete frozen account (0/8)", fontsize=14
    )
    save(fig, "first-source-effects")
    ledger, joint = rows("ledger.csv"), rows("posthoc-joint/joint.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for i, (kind, label) in enumerate([("write", "Attention write"), ("mlp", "MLP write")]):
        rr = [r for r in ledger if r["component"] == kind]
        axs[0].scatter(
            i + np.linspace(-0.13, 0.13, 8),
            [float(r["signed_target_fraction"]) for r in rr],
            color="#285f87",
            s=25,
        )
    axs[0].axhline(0, color="#8a8a8a", lw=0.7)
    axs[0].axhline(1, color="#8a8a8a", lw=0.7)
    axs[0].set_xticks([0, 1], ["Attention write", "MLP write"])
    axs[0].set_xlabel("Native source component")
    axs[0].set_ylabel("Signed projection / native source contrast")
    axs[0].set_title("Large MLP projection is accounting, not sufficiency")
    for i, alias in enumerate("AB"):
        rr = [r for r in joint if r["component"] == "source" and r["alias"] == alias]
        axs[1].scatter(
            i + np.linspace(-0.13, 0.13, 8),
            [float(r["signed_target_fraction"]) for r in rr],
            color="#285f87",
            s=25,
        )
    axs[1].axhline(0, color="#8a8a8a", lw=0.7)
    axs[1].set_xticks([0, 1], ["Reference A", "Reference B"])
    axs[1].set_xlabel("Reference for defined K/V corner interaction")
    axs[1].set_ylabel("Signed joint projection / native source contrast")
    axs[1].set_title("Joint-term sign reverses with the reference")
    fig.suptitle(
        "Posthoc vector ledgers: all eight cases; projections are not unique causal shares",
        fontsize=12,
    )
    save(fig, "signed-ledgers")
    inputs = ["effects.csv", "phase.csv", "ledger.csv", "posthoc-joint/joint.csv"]
    write(
        OUT / "provenance.json",
        dict(
            all_eight_cases=True,
            posthoc_ledgers=True,
            script_sha256=sha(ROOT / "research/boolean_coupling/describe302.py"),
            inputs={str((DATA / n).relative_to(ROOT)): sha(DATA / n) for n in inputs},
            outputs={p.name: sha(p) for p in OUT.iterdir() if p.suffix in [".svg", ".png"]},
        ),
    )


if __name__ == "__main__":
    main()
