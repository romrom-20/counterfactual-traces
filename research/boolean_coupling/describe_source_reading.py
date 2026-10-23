"""Posthoc all-case source-reading factorial and signed native ledger; no model calls."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/source-attention-reading-275"
OUT = ROOT / "docs/figures/source-reading"


def rows(name):
    with (DATA / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    emission, native, ledger = (rows(n) for n in ("emission", "native", "paired-source-ledger"))
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2))
    colors = dict(generic="#2a6f91", descriptive="#aa542a")
    arms = ("W", "V", "B")
    labels = ("False probabilities\ncurrent self V (W)", "Current probabilities\nFalse self V (V)", "Both False\n(B)")
    for i, arm in enumerate(arms):
        group = [r for r in emission if r["arm"] == arm]
        for j, r in enumerate(group):
            offset = ((j * 13) % 31 - 15) / 150
            axes[0].scatter(i + offset, 100 * float(r["relative"]), color=colors[r["schema"]], s=17,
                            marker="o" if int(r["reverse"]) == 0 else "^", alpha=.7)
        value = median(100 * float(r["relative"]) for r in group)
        axes[0].plot([i - .19, i + .19], [value, value], color="black", linewidth=2)
        for offset, prefix, color, label in ((-.18, "", "#2a6f91", "Full source intervention"),
                                            (.18, "value_", "#888888", "Emitted V only; native source K")):
            records = [r for r in native if r["arm"] == prefix + arm]
            for j, r in enumerate(records):
                jitter = ((j * 13) % 31 - 15) / 400
                axes[1].scatter(i + offset + jitter, 100 * float(r["all_relative"]), color=color, s=12, alpha=.6)
            value = median(100 * float(r["all_relative"]) for r in records)
            axes[1].plot([i + offset - .1, i + offset + .1], [value, value], color=color, linewidth=2.5,
                         label=label if i == 0 else None)
    for ax in axes[:2]:
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_xlabel("Source-query factorial rule")
        ax.set_ylim(bottom=0)
    axes[0].axhline(10, color="#666666", linestyle=":", linewidth=1)
    axes[0].set(title="Source emission across 28 layers", ylabel="Aggregate value-change error (%)")
    axes[1].axhline(5, color="#666666", linestyle=":", linewidth=1)
    axes[1].set(title="Complete native response N−F", ylabel="Whole receiving-vector error (%)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -.29))
    for term, color, style in (("attention", "#72548d", "-"), ("mlp", "#aa542a", "-"), ("rounding", "#777777", ":")):
        curve = [median(float(r["signed_projection"]) for r in ledger if r["term"] == term and int(r["layer"]) == i) for i in range(1, 29)]
        axes[2].plot(range(1, 29), curve, color=color, linestyle=style, label=term.upper() if term == "mlp" else term.title())
    axes[2].axhline(1, color="#666666", linestyle="--", linewidth=.8)
    axes[2].set(title="Cumulative native source ledger", xlabel="Source block output", ylabel="Signed projection on paired h contrast")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=s.title()) for s, c in colors.items()]
    handles.append(plt.Line2D([], [], color="black", linewidth=2, label="Across-case median"))
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.035, .09))
    fig.suptitle("How does the edited source generate its context-dependent message?", fontsize=12, y=.98)
    fig.text(.5, .038, "32 old contexts / 16 matched pairs · all source MLPs and receivers finite · signed ledgers are descriptive, not causal layer rankings",
             ha="center", fontsize=8, color="#555555")
    fig.text(.5, .012, "Native tests also require verdict-vector/cosine/margin bounds · block28 source output has no later receiver read · no fresh predictor or learned circuit",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .19, 1, .94), w_pad=2.3)
    fig.savefig(OUT / "source-reading-factorial.png", dpi=220)
    fig.savefig(OUT / "source-reading-factorial.svg", metadata={"Date": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, name in enumerate(("first-source", "first-source-pairs")):
        group = [r for r in rows(name) if r["arm"] == "W"]
        value = median(100 * float(r["relative"]) for r in group)
        ax.bar(i, value, width=.5, color="#dddddd", zorder=2)
        for j, r in enumerate(group):
            ax.scatter(i + ((j * 13) % 31 - 15) / 110, 100 * float(r["relative"]),
                       color=colors[r["schema"]], marker="o",
                       s=20, alpha=.7, zorder=3)
        ax.text(i, max(value, max(100 * float(r["relative"]) for r in group)) + 4,
                f"Median {value:.1f}%", ha="center", fontsize=10)
    ax.set_xticks((0, 1), ("Full Boolean\nsource displacement", "Cross-background\ndisplacement difference"))
    ax.set(xlim=(-.65, 1.65), ylim=(0, 100), ylabel="Relative vector error (%)", xlabel="Native target at the first source-block output")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax.set_title("A close total vector can miss its context dependence", fontsize=12)
    ax.legend(handles=handles[:2], loc="upper left", frameon=False, fontsize=9)
    fig.text(.5, .05, "Fixed source probabilities W · all32 old cases / 16 matched pairs · posthoc saved-array analysis", ha="center", fontsize=8)
    fig.text(.5, .012, "The two errors use different target norms; this does not identify a learned feature or establish a causal rescue.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(OUT / "first-block-context.png", dpi=220)
    fig.savefig(OUT / "first-block-context.svg", metadata={"Date": None})
    plt.close(fig)
    for svg in OUT.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(dict(posthoc=True, model_calls=0,
        source_sha256=sha(Path(__file__)), inputs={str((DATA / (n + ".csv")).relative_to(ROOT)): sha(DATA / (n + ".csv"))
        for n in ("emission", "native", "paired-source-ledger", "first-source", "first-source-pairs")},
        scope="All32 old cases; complete native accuracy, source-emitter fidelity and retrospective signed algebra are separate quantities."), indent=2) + "\n")


if __name__ == "__main__":
    main()
