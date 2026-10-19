"""Plot complete old-forecast error accounting and native-reset factorial outcomes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def outcomes(ax, rows, arms, labels, title):
    for index, arm in enumerate(arms):
        values = [r for r in rows if r["arm"] == arm]
        for j, row in enumerate(values):
            color = "#007b8e" if row["schema"] == "generic" else "#b75b31"
            ax.scatter(index+(j%8-3.5)*.018, float(row["margin_error"]), s=22, alpha=.65, color=color)
        ax.hlines(np.median([float(r["margin_error"]) for r in values]), index-.23, index+.23,
            linewidth=2, color="black")
    ax.axhline(.10, color="#888888", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(arms)), labels, fontsize=8)
    ax.set_ylim(bottom=-.05)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Absolute final margin error (nats)")


def save(fig, out, name):
    for suffix in ("png", "svg"):
        path = out/(name+"."+suffix)
        assert not path.exists(), "Never overwrite a figure."
        fig.savefig(path, dpi=180, metadata={"Date":None} if suffix == "svg" else None)
    plt.close(fig)


def main(input235, input236, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.hashsalt":"global-failure-235-236"})
    ledger = [r for r in read(input235/"ledger.csv") if r["scope"] == "all" and r["kind"] != "cast"]
    rows = read(input235/"outcomes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for node in range(47):
        values = [r for r in ledger if int(r["node"]) == node]
        color = "#aa4362" if values[0]["kind"] == "mlp" else "#356ba1"
        for j, row in enumerate(values):
            axes[0].scatter(node+(j%8-3.5)*.025, float(row["signed_share"]), s=5, alpha=.12, color=color)
        axes[0].plot(node, np.median([float(r["signed_share"]) for r in values]), marker="o", markersize=3, color=color)
    axes[0].axhline(0, color="#888888", linewidth=.8)
    axes[0].set_xticks([0, 9, 19, 29, 39, 46], ["A1", "M5", "M10", "M15", "M20", "A24"])
    axes[0].set_title("235: complete signed write-error ledger", fontsize=12)
    axes[0].set_ylabel("Projection onto terminal parent error / error norm²")
    axes[0].set_xlabel("Chronological update; blue: attention, rose: MLP")
    axes[0].text(.02, .96, "Top four: MLP23,22,21,20", transform=axes[0].transAxes, va="top", fontsize=10)
    outcomes(axes[1], rows, ["base", "gen4", "state4", "top4", "random4_0", "random4_1", "random4_2", "full"],
        ["original", "local\nerror", "input-state\nresponse", "top four\nwrites", "random0", "random1", "random2", "all oracle\nwrites"],
        "Terminal-vector interventions; final readout finite")
    fig.suptitle("Late write discrepancies are large, but their terminal repair is not a complete account", fontsize=14)
    fig.text(.015, .035, "Every old case shown; black: median. Signed ledger projections are accounting, not unique causal fractions. Cast term retained in raw ledger.", fontsize=9)
    fig.text(.015, .009, "Terminal corrections use native oracle writes and do not recompute upstream forecast states. TOP4 joint rule passes 2/32; no prospective result.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .92))
    save(fig, out, "235-global-error-budget")
    rows = read(input236/"outcomes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    labels = ["unchanged", "finite\nMLP20", "finite\nMLP21", "both\nfinite", "row-matched\ncontrol"]
    for ax, incoming, title in zip(axes, ("forecast", "oracle"),
            ("236: original forecast input", "Exact native reset before MLP20 (oracle)")):
        arms = [incoming+"_"+m for m in ("00", "10", "01", "11", "R")]
        current_labels = list(labels)
        if incoming == "oracle":
            arms.append("finite_oracle")
            current_labels.append("all\nfinite")
        outcomes(ax, rows, arms, current_labels, title)
    fig.suptitle("Upstream MLP20/21 repairs help; later attention approximations still prevent native closure", fontsize=14)
    fig.text(.015, .035, "Every old case shown; black: median; teal/orange: schema; dashed: 0.10-nat margin bound. Vector/cosine bounds also apply.", fontsize=9)
    fig.text(.015, .009, "All approximate arms pass 0/32 full-response accuracy; fully finite replay passes 32/32. Oracle resets are not valid prospective inputs.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .92))
    save(fig, out, "236-upstream-factorial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-235", type=Path, required=True)
    parser.add_argument("--input-236", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_235, args.input_236, args.out)
