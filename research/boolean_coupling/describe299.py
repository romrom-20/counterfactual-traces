"""Display every native outcome and signed, fixed-parent attention-write effects."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/query-generators-299/native-phase"
OUT = ROOT / "docs/figures/query-generators-299"
ARMS = ("C", "M", "B")


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=170, facecolor="white")
    fig.savefig(OUT / (name + ".svg"), metadata={"Date": None})
    plt.close(fig)
    p = OUT / (name + ".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (DEST / "native.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    audit = read(DEST / "audit.json")
    local = read(DEST / "local-records.json")["statistics"]
    summary = dict(
        experiment=299,
        descriptive=True,
        all_32_cases=True,
        primary_claims_failed=not audit["H1_carry"] and not audit["H2_last_MLP"],
        medians={},
    )
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        summary["medians"][arm] = {
            f: float(np.median([float(r[f]) for r in rr]))
            for f in (
                "receiving_relative",
                "verdict_relative",
                "margin_error",
                "query_receiving_relative",
                "query_verdict_relative",
            )
        }
    write(DEST / "descriptive-summary.json", summary)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "299",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.8))
    titles = {"C": "Carried\nstate", "M": "Recent\nMLP write", "B": "Frozen\nquery"}
    labels = [titles[a] + f"\n{audit['populations'][a]['passes']}/32 joint passes" for a in ARMS]
    for ax, field, title, bound in zip(
        axes,
        ("receiving_relative", "verdict_relative", "margin_error"),
        ("Receiving displacement", "Verdict displacement", "Finite verdict margin"),
        (0.05, 0.05, 0.1),
        strict=True,
    ):
        for j, arm in enumerate(ARMS):
            for layout, offset, color, label in (
                ("jsonlines", -0.21, "#087f8c", "JSON lines"),
                ("ini", -0.07, "#c16a28", "INI"),
                ("sexpr", 0.07, "#66528b", "S-expression"),
                ("memo", 0.21, "#668b3d", "Memo"),
            ):
                vals = [float(r[field]) for r in rows if r["arm"] == arm and r["layout"] == layout]
                ax.scatter(
                    j + offset + np.linspace(-0.03, 0.03, len(vals)),
                    np.maximum(vals, 1e-9),
                    s=20,
                    alpha=0.75,
                    color=color,
                    label=label if j == 0 else None,
                )
            med = summary["medians"][arm][field]
            ax.plot([j - 0.28, j + 0.28], [max(med, 1e-9)] * 2, color="#222", lw=1.5)
        ax.axhline(bound, color="#222", ls="--", lw=1)
        ax.set_xticks(range(3), labels)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_ylabel(
            "Absolute error (nats)" if field == "margin_error" else "Relative vector error"
        )
        ax.grid(axis="y", alpha=0.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.14),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "What generates the receiving-query change?",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.88,
        "Query-only operand policies. Every other operation evolves on its own state; this is an expensive intervention test.",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "All 32 fresh inputs; four related synthetic encodings; one checkpoint. Forecasts sealed before native True.\nBlack bars: medians. Complete success requires both vectors, cosine guards and margin. Any zero error is displayed at 1e-9.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.34, wspace=0.34)
    save(fig, "native-generators")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    for col, layer in enumerate((2, 15, 28)):
        for row, field in enumerate(("signed_target_fraction", "relative")):
            ax = axes[row, col]
            for j, arm in enumerate(("C", "M", "interaction")):
                for schema, offset, color, label in (
                    ("generic", -0.08, "#087f8c", "Generic"),
                    ("descriptive", 0.08, "#c16a28", "Descriptive"),
                ):
                    rr = [
                        r
                        for r in local
                        if r["kind"] == "attention_write"
                        and r["layer"] == layer
                        and r["arm"] == arm
                        and r["schema"] == schema
                        and r["is_verdict"]
                        and r["target_eligible"]
                    ]
                    vals = [r[field] for r in rr]
                    ax.scatter(
                        j + offset + np.linspace(-0.025, 0.025, len(vals)),
                        vals,
                        s=27,
                        color=color,
                        alpha=0.8,
                        label=label if j == 0 else None,
                    )
                vals = [
                    r[field]
                    for r in local
                    if r["kind"] == "attention_write"
                    and r["layer"] == layer
                    and r["arm"] == arm
                    and r["is_verdict"]
                    and r["target_eligible"]
                ]
                if vals:
                    ax.plot([j - 0.28, j + 0.28], [np.median(vals)] * 2, color="#222", lw=1.5)
            ax.axhline(0, color="#aaa", lw=0.8)
            ax.axhline(1 if row == 0 else 0.25, color="#666", ls="--", lw=0.9)
            ax.set_xticks(range(3), ["Carry", "Recent MLP", "Interaction"])
            ax.set_title(f"Block {layer}", loc="left", fontweight="bold")
            ax.set_ylabel("Signed target projection" if row == 0 else "Relative vector error")
            ax.grid(axis="y", alpha=0.15)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Local query operands at identical native parents",
        x=0.065,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.91,
        "Projected attention-write effects at the verdict row: large signed contributions can still miss the full vector.",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.025,
        "Eight retained native witnesses; all receiving rows are in the records. These supplied-state local readouts are retrospective.\nThe carry includes earlier MLPs. Corners and their interaction are defined interventions, not learned features or unique causal percentages.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.18, hspace=0.49, wspace=0.35)
    save(fig, "local-generators")
    inputs = {
        str((DEST / n).relative_to(ROOT)): sha(DEST / n)
        for n in ("native.csv", "audit.json", "local-records.json")
    }
    outputs = {p.name: sha(p) for p in OUT.glob("*") if p.suffix in (".png", ".svg")}
    write(
        OUT / "provenance.json",
        dict(
            experiment=299,
            all_32_cases=True,
            script_sha256=sha(Path(__file__)),
            inputs=inputs,
            outputs=outputs,
            local_scope="Fixed native parents, eight witnesses; signed projection and vector error; not a learned circuit or a False-only local forecast.",
        ),
    )
    print(summary)


if __name__ == "__main__":
    main()
