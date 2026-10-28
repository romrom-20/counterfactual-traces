"""Plot every conditional probe, with frozen decisions and weaker pattern separate."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/response-change-294"
OUT = ROOT / "docs/figures/response-change-294"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (DEST / "decisions.csv").open() as f:
        dec = list(csv.DictReader(f))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "294",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, scope, title in zip(
        axes[:2], ("source", "verdict"), ("Source probes", "Verdict probes"), strict=True
    ):
        for arm, label, color in [
            ("FALSE_RANGE", "False", "#bf7421"),
            ("ENDPOINT_RANGE", "Endpoint", "#66528b"),
            ("CHANGE_RANGE", "Response change", "#087f8c"),
            ("RANDOM_RANGE", "Random", "#999"),
        ]:
            med = []
            for j, layer in enumerate((1, 15, 28)):
                values = [
                    float(r["relative"])
                    for r in rows
                    if r["scope"] == scope and int(r["layer"]) == layer and r["arm"] == arm
                ]
                med.append(np.median(values))
                offset = {
                    "FALSE_RANGE": -0.18,
                    "ENDPOINT_RANGE": -0.06,
                    "CHANGE_RANGE": 0.06,
                    "RANDOM_RANGE": 0.18,
                }[arm]
                ax.scatter(
                    j + offset + np.linspace(-0.025, 0.025, len(values)),
                    values,
                    s=15,
                    color=color,
                    alpha=0.65,
                )
            ax.plot(
                np.arange(3)
                + {
                    "FALSE_RANGE": -0.18,
                    "ENDPOINT_RANGE": -0.06,
                    "CHANGE_RANGE": 0.06,
                    "RANDOM_RANGE": 0.18,
                }[arm],
                med,
                color=color,
                label=label,
                lw=1.5,
            )
        ax.axhline(0.25, color="#222", ls="--", lw=1, label="Complete-vector bound .25")
        ax.set_xticks(range(3), ("1", "15", "28"))
        ax.set_xlabel("Block")
        ax.set_ylabel("Relative nonlinear-write vector error")
        ax.set_ylim(0, 1.04)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[2]
    for scope, color, marker in [("source", "#c34b32", "o"), ("verdict", "#087f8c", "^")]:
        rr = [r for r in dec if r["scope"] == scope]
        ax.scatter(
            [float(r["best_endpoint_error"]) for r in rr],
            [float(r["change_error"]) for r in rr],
            c=color,
            marker=marker,
            label=scope,
            s=27,
            alpha=0.8,
        )
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, color="#777", lw=1, label="Equal error")
    ax.plot(x, 0.5 * x, color="#222", ls="--", lw=1, label="Frozen twofold threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Better endpoint error")
    ax.set_ylabel("Response-change error")
    ax.set_title("Improvement falls short", loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.15)
    fig.suptitle(
        "Response change selects better directions, but does not explain the write",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.885,
        "Conditional saved P inputs: 48 probes, eight old contexts. Primary twofold improvement: 8/48; complete vector account: 0/48.",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "All probes retained. Rank 32, same input sketch and power budget. No new model forward, native forecast or causal intervention.\nThe weaker all 48 improvement is descriptive; it does not rescue either frozen hypothesis.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.79, wspace=0.35)
    for ext in ("png", "svg"):
        fig.savefig(
            OUT / ("response-change." + ext),
            dpi=175,
            metadata={"Date": None} if ext == "svg" else {},
        )
    plt.close(fig)
    p = OUT / "response-change.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    write(
        OUT / "provenance.json",
        dict(
            experiment=294,
            all_48_probes=True,
            script_sha256=sha(Path(__file__)),
            inputs={
                str((DEST / n).relative_to(ROOT)): sha(DEST / n)
                for n in ("metrics.csv", "decisions.csv")
            },
            outputs={p.name: sha(p) for p in sorted(OUT.glob("response-change.*"))},
        ),
    )


if __name__ == "__main__":
    main()
