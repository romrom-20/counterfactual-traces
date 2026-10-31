"""Every equivalent-alias pair, materiality, causal-route and native-transfer result."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/alias-source-routes-301/native-phase"
OUT = ROOT / "docs/figures/alias-source-routes-301"
COLORS = {"jsonlines": "#087f8c", "ini": "#c16a28", "sexpr": "#66528b", "memo": "#668b3d"}


def load(name):
    with (DEST / name).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=170, facecolor="white")
    fig.savefig(OUT / (name + ".svg"), metadata={"Date": None}, facecolor="white")
    plt.close(fig)
    p = OUT / (name + ".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    effects, native = load("effects.csv"), load("native-predictions.csv")
    audit = read(DEST / "audit.json")
    accounts = ("source", "remaining", "random")
    summary = dict(
        experiment=301,
        descriptive=True,
        all_32_pairs=True,
        material_alias_cases=audit["material_alias_cases"],
        medians={},
    )
    for account in accounts:
        rr = [r for r in effects if r["account"] == account]
        summary["medians"][account] = {
            f: float(np.median([float(r[f]) for r in rr]))
            for f in ("receiving_relative", "verdict_relative", "margin_error")
        }
    paired = []
    for key in sorted({r["key"] for r in native}):
        rr = [r for r in native if r["key"] == key]
        paired.append(
            dict(
                key=key,
                layout=rr[0]["layout"],
                **{
                    f: max(float(r[f]) for r in rr)
                    for f in ("receiving_relative", "verdict_relative", "margin_error")
                },
            )
        )
    summary["native_worst_alias_medians"] = {
        f: float(np.median([r[f] for r in paired]))
        for f in ("receiving_relative", "verdict_relative", "margin_error")
    }
    contrasts = [
        float(r["native_alias_margin_contrast"]) for r in effects if r["account"] == "source"
    ]
    summary["median_absolute_alias_contrast"] = float(np.median(np.abs(contrasts)))
    write(DEST / "descriptive-summary.json", summary)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "301",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.1))
    ax = axes[0]
    for i, (layout, color) in enumerate(COLORS.items()):
        for order, off, marker in ((0, -0.10, "o"), (1, 0.10, "^")):
            rr = [
                r
                for r in effects
                if r["account"] == "source"
                and r["layout"] == layout
                and int(r["mention_order"]) == order
            ]
            for k, r in enumerate(rr):
                material = r["material"] == "True"
                ax.scatter(
                    i + off + (k - 1.5) * 0.024,
                    float(r["native_alias_margin_contrast"]),
                    marker=marker,
                    s=28,
                    edgecolors=color,
                    facecolors=color if material else "white",
                    label=layout if order == 0 and k == 0 else None,
                )
    ax.axhspan(-0.25, 0.25, color="#ddd", alpha=0.5)
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(range(4), ["JSON\nlines", "INI", "S-expr.", "Memo"])
    ax.set_ylabel("Boolean-coupling difference B - A (nats)")
    ax.set_title(
        f"Native alias contrast: {audit['material_alias_cases']}/32 material",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    labels = [
        n + f"\n{audit['populations'][a]['passes']}/32"
        for a, n in zip(
            accounts, ("Source\ngenerator", "Remaining\ncontext", "Matched\nrandom"), strict=True
        )
    ]
    for ax, field, title, bound in zip(
        axes[1:],
        ("receiving_relative", "margin_ratio"),
        ("Worst receiving-vector error", "Coupling error / frozen bound"),
        (0.25, 1),
        strict=True,
    ):
        for i, account in enumerate(accounts):
            for layout, offset in zip(COLORS, (-0.21, -0.07, 0.07, 0.21), strict=True):
                rr = [r for r in effects if r["account"] == account and r["layout"] == layout]
                for k, r in enumerate(rr):
                    value = (
                        float(r["margin_error"]) / max(float(r["margin_bound"]), 1e-30)
                        if field == "margin_ratio"
                        else float(r[field])
                    )
                    material = r["material"] == "True"
                    ax.scatter(
                        i + offset + (k - 3.5) * 0.008,
                        value,
                        s=22,
                        edgecolors=COLORS[layout],
                        facecolors=COLORS[layout] if material else "white",
                        linewidths=1,
                    )
            vals = [
                float(r["margin_error"]) / max(float(r["margin_bound"]), 1e-30)
                if field == "margin_ratio"
                else float(r[field])
                for r in effects
                if r["account"] == account
            ]
            ax.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, color="#222", lw=1.5)
        ax.axhline(bound, color="#444", ls="--", lw=1)
        ax.set_xticks(range(3), labels)
        ax.set_yscale("symlog", linthresh=0.02)
        ax.set_ylim(bottom=-0.003)
        ax.set_ylabel(
            "Error / case-specific bound" if field == "margin_ratio" else "Relative vector error"
        )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Equivalent names: where is the change in Boolean coupling generated?",
        x=0.055,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.88,
        "Identical instructions and matched positions; two valid metadata aliases. All forecasts sealed before ordinary native True.",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.035,
        "All 32 pairs remain. Hollow points are below the .25-nat materiality guard. Counts are complete joint passes, not single metrics.\nBoth reference contexts and both vector scopes must pass. Black bars: medians. Circles/triangles in the first panel: alias mention order.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.33, wspace=0.35)
    save(fig, "alias-routes")

    fig, axes = plt.subplots(1, 3, figsize=(12.7, 5.4))
    for ax, field, title, bound in zip(
        axes,
        ("receiving_relative", "verdict_relative", "margin_error"),
        ("Receiving displacement", "Verdict displacement", "Finite verdict margin"),
        (0.05, 0.05, 0.1),
        strict=True,
    ):
        for i, (layout, color) in enumerate(COLORS.items()):
            rr = [r[field] for r in paired if r["layout"] == layout]
            ax.scatter(i + np.linspace(-0.08, 0.08, len(rr)), rr, color=color, s=25, alpha=0.8)
        ax.axhline(bound, color="#444", ls="--", lw=1)
        ax.set_xticks(range(4), ["JSON\nlines", "INI", "S-expr.", "Memo"])
        ax.set_ylabel(
            "Absolute error (nats)" if field == "margin_error" else "Relative vector error"
        )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        f"Separate full-response transfer: {audit['native_pair_passes']}/32 pairs pass for BOTH aliases",
        x=0.06,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.84,
        "Each point is the worse of the two alias errors. Two False observations and dense finite hybrids are used.",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.035,
        "All 32 pairs, all cases. Dashed lines: individual bounds; cosines and norm guards also apply.\nA native prediction success cannot rescue a failed causal account of the alias contrast. No compact/single-state predictor is claimed.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.72, bottom=0.24, wspace=0.35)
    save(fig, "native-transfer")

    emitted = read(DEST / "emission-records.json")
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 7.8))
    for row, (field, heading) in enumerate(
        (
            ("residual_statistics", "Source residual after block"),
            ("statistics", "Source value before block"),
        )
    ):
        records = emitted[field]
        for col, metric in enumerate(("relative", "cosine")):
            ax = axes[row, col]
            for key in sorted({r["key"] for r in records}):
                rr = sorted([r for r in records if r["key"] == key], key=lambda r: r["layer"])
                ax.plot(
                    [r["layer"] for r in rr],
                    [r[metric] for r in rr],
                    color=COLORS[rr[0]["layout"]],
                    alpha=0.16,
                    lw=0.8,
                )
            med = [np.median([r[metric] for r in records if r["layer"] == j]) for j in range(1, 29)]
            ax.plot(range(1, 29), med, color="#222", lw=2)
            ax.set_title(heading, loc="left", fontweight="bold", fontsize=11)
            ax.set_xlabel("Block")
            ax.set_ylabel(
                "Relative difference of Boolean changes"
                if metric == "relative"
                else "Cosine between Boolean changes"
            )
            ax.axhline(0 if metric == "relative" else 1, color="#888", ls="--", lw=0.7)
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "The Boolean change becomes dependent on the chosen alias",
        x=0.065,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.91,
        "Same initial Boolean embedding and positions; only two earlier name tokens differ. All 32 native pairs, no new model calls.",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.025,
        "Read-only POSTHOC profile. Thin lines: pairs; black: median. Initial emitted differences are exactly equal in every pair.\nResidual and value maps have different dimensions; their normalized differences are not an invariant gain or a learned feature.",
        fontsize=9,
        color="#444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.15, hspace=0.42, wspace=0.3)
    save(fig, "source-direction")
    inputs = {
        str((DEST / n).relative_to(ROOT)): sha(DEST / n)
        for n in ("effects.csv", "native-predictions.csv", "audit.json")
    }
    inputs[str((DEST / "emission-records.json").relative_to(ROOT))] = sha(
        DEST / "emission-records.json"
    )
    write(
        OUT / "provenance.json",
        dict(
            experiment=301,
            all_32_pairs=True,
            script_sha256=sha(Path(__file__)),
            inputs=inputs,
            outputs={p.name: sha(p) for p in OUT.glob("*") if p.suffix in (".png", ".svg")},
        ),
    )
    print(summary)


if __name__ == "__main__":
    main()
