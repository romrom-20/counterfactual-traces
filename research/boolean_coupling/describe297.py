"""All native decisions and fixed-parent diagnostics; no hidden-state inference."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-routes-297/native-phase"
OUT = ROOT / "docs/figures/query-routes-297"
ARMS = ["S", "C", "B"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "native.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    audit = read(DEST / "audit.json")
    local = read(DEST / "local-records.json")["statistics"]
    summary = dict(
        experiment=297,
        descriptive=True,
        all_32_cases=True,
        primary_claims_failed=not audit["H1_source"] and not audit["H2_other"],
        medians={},
    )
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        summary["medians"][arm] = {
            field: float(np.median([float(r[field]) for r in rr]))
            for field in [
                "receiving_relative",
                "verdict_relative",
                "margin_error",
                "query_receiving_relative",
                "query_verdict_relative",
            ]
        }
    write(DEST / "descriptive-summary.json", summary)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "297",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6))
    labels = ["Source\nallocation\n0/32", "Context\nselection\n0/32", "Frozen\nquery\n0/32"]
    for ax, field, title, bound in zip(
        axes,
        ["receiving_relative", "verdict_relative", "margin_error"],
        ["Receiving displacement", "Verdict displacement", "Finite verdict margin"],
        [0.05, 0.05, 0.1],
        strict=True,
    ):
        for j, arm in enumerate(ARMS):
            for layout, offset, color, label in [
                ("jsonlines", -0.21, "#087f8c", "JSON lines"),
                ("ini", -0.07, "#c16a28", "INI"),
                ("sexpr", 0.07, "#66528b", "S-expression"),
                ("memo", 0.21, "#668b3d", "Memo"),
            ]:
                vals = [float(r[field]) for r in rows if r["arm"] == arm and r["layout"] == layout]
                ax.scatter(
                    j + offset + np.linspace(-0.03, 0.03, len(vals)),
                    vals,
                    s=20,
                    alpha=0.75,
                    color=color,
                    label=label if j == 0 else None,
                )
            med = summary["medians"][arm][field]
            ax.plot([j - 0.28, j + 0.28], [med, med], color="#222", lw=1.5)
        ax.axhline(bound, color="#222", ls="--", lw=1)
        ax.set_xticks(range(3), labels)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel(
            "Absolute error (nats)" if field == "margin_error" else "Relative error"
        )
        ax.grid(axis="y", alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.51, 0.13), ncol=4, frameon=False
    )
    fig.suptitle(
        "Context selection retains more; neither restricted account is sufficient",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.88,
        "Both frozen native population claims FAIL. The joint route is the full native calibration, not a discovered shortcut.",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "All 32 fresh full inputs, four related synthetic encodings, one checkpoint. Forecasts sealed before native True.\nBlack bars: medians. Dashed lines: individual bounds; complete success also requires both vectors, cosines and margin.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.33, wspace=0.34)
    save(fig, "native-routes")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.9), sharey=True)
    for ax, layer in zip(axes, [2, 15, 28], strict=True):
        for j, arm in enumerate(["S", "C", "interaction"]):
            for schema, off, color, label in [
                ("generic", -0.09, "#087f8c", "Generic"),
                ("descriptive", 0.09, "#c16a28", "Descriptive"),
            ]:
                keys = {
                    c["key"]
                    for c in read(ROOT / "results/query-routes-297/pre-native/design.json")["cases"]
                    if c["group"]["schema"] == schema
                }
                rr = [
                    r
                    for r in local
                    if r["key"] in keys
                    and r["layer"] == layer
                    and r["arm"] == arm
                    and r["is_verdict"]
                    and r["target_eligible"]
                ]
                vals = [r["signed_target_fraction"] for r in rr]
                ax.scatter(
                    j + off + np.linspace(-0.025, 0.025, len(vals)),
                    vals,
                    s=27,
                    color=color,
                    alpha=0.8,
                    label=label if j == 0 else None,
                )
            vals = [
                r["signed_target_fraction"]
                for r in local
                if r["layer"] == layer
                and r["arm"] == arm
                and r["is_verdict"]
                and r["target_eligible"]
            ]
            ax.plot([j - 0.28, j + 0.28], [np.median(vals)] * 2, color="#222", lw=1.5)
        ax.axhline(0, color="#777", lw=0.7)
        ax.axhline(1, color="#777", lw=0.7, ls="--")
        ax.set_xticks(range(3), ["Source\nallocation", "Context\nselection", "Interaction"])
        ax.set_title("Block " + str(layer), loc="left", fontweight="bold")
        ax.set_ylim(-0.12, 1.15)
        if layer == 2:
            ax.set_ylabel("Signed projection on local query write", fontsize=9)
        ax.grid(axis="y", alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.13), ncol=2, frameon=False
    )
    fig.suptitle(
        "At fixed native parents, context selection carries most of the projected write",
        x=0.065,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.86,
        "Eight preselected cases. Verdict receiving position. Actual output weights applied to saved head-output differences.",
        fontsize=9,
    )
    fig.text(
        0.065,
        0.035,
        "These are conditional local projections, not percentages of the final verdict or full-vector accuracy.\nAll rows remain in the records; the plotted target norms exceed .001. Separate medians need not add to one.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.76, bottom=0.29, wspace=0.35)
    save(fig, "local-routes")
    inputs = ["native.csv", "audit.json", "local-records.json"]
    write(
        OUT / "provenance.json",
        dict(
            experiment=297,
            all_32_cases=True,
            local_cases=8,
            local_diagnostic_is_retrospective=True,
            script_sha256=sha(Path(__file__)),
            inputs={str((DEST / n).relative_to(ROOT)): sha(DEST / n) for n in inputs},
            outputs={p.name: sha(p) for p in sorted(OUT.glob("*-routes.*"))},
        ),
    )


def save(fig, stem):
    for ext in ["png", "svg"]:
        fig.savefig(
            OUT / (stem + "." + ext), dpi=175, metadata={"Date": None} if ext == "svg" else {}
        )
    plt.close(fig)
    p = OUT / (stem + ".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
