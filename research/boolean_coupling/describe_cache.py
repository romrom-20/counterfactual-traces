"""Posthoc native accuracy and paired latency figure; every old269 context."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/causal-prefix-budget-272"
OLD = ROOT / "results/signed-amplitude-forecast-269/native-outcomes.csv"
OUT = ROOT / "docs/figures/cache"


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    old, new, timings = read(OLD), read(DATA / "native.csv"), read(DATA / "timing.csv")
    keys = list(dict.fromkeys(r["key"] for r in new))
    lookup = {(r["key"], r["arm"]): r for r in old + new}
    assert len(keys) == 32 and all((k, a) in lookup for k in keys for a in ("H", "A10", "P"))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for j, key in enumerate(keys):
        meta = lookup[key, "P"]
        color = "#2a6f91" if meta["schema"] == "generic" else "#aa542a"
        marker = "o" if int(meta["reverse"]) == 0 else "^"
        offset = ((j * 13) % 31 - 15) / 160
        for i, arm in enumerate(("H", "A10", "P")):
            value = 100 * float(lookup[key, arm]["all_relative"])
            assert .0005 < value < 10
            axes[0].scatter(i + offset, value, color=color, marker=marker, s=19, alpha=.65)
        group = [r for r in timings if r["key"] == key]
        full, suffix = (1000 * median(float(r[name]) for r in group) for name in ("full_seconds", "suffix_seconds"))
        axes[1].plot([0, 1], [full, suffix], color=color, linewidth=.8, alpha=.28)
        axes[1].scatter([offset, 1 + offset], [full, suffix], color=color, marker=marker, s=16, alpha=.65)
    for i, arm in enumerate(("H", "A10", "P")):
        value = median(100 * float(lookup[k, arm]["all_relative"]) for k in keys)
        axes[0].plot([i - .15, i + .15], [value, value], color="black", linewidth=2)
    for i, name in enumerate(("full_seconds", "suffix_seconds")):
        value = median(1000 * median(float(r[name]) for r in timings if r["key"] == k) for k in keys)
        axes[1].plot([i - .15, i + .15], [value, value], color="black", linewidth=2)
    axes[0].set(title="Native receiving-vector error", ylabel="Relative error (%) · log scale",
                xlabel="Prediction on the same 32 old contexts", yscale="log", ylim=(.0005, 10))
    axes[0].set_xticks(range(3), ["Finite MLP /\ncubic query H", "Fixed correction\naxes A10", "Cached finite\nsuffix P"])
    axes[0].axhline(5, linestyle=":", color="#666666", linewidth=1)
    axes[0].text(-.08, 5.5, "5% whole-vector bound", fontsize=8, color="#555555")
    axes[1].set(title="Marginal counterfactual latency", ylabel="Milliseconds · median of 3 repetitions per case",
                xlabel="Same model; GPU synchronized; cache-copy cost included", ylim=(0, None))
    axes[1].set_xticks([0, 1], ["Full native\nforward T", "Cached finite\nsuffix P"])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.15)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=s)
               for c, s in (("#2a6f91", "Generic"), ("#aa542a", "Descriptive"))]
    handles.append(plt.Line2D([], [], color="black", linewidth=2, label="Across-case median"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, .05))
    fig.suptitle("A causal cache gives a stronger cost baseline for these late edits", fontsize=12, y=.98)
    fig.text(.5, .015, "All 32 old 269 contexts · latency excludes shared False observation/model loading · no new task or checkpoint confirmation",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .15, 1, .94), w_pad=2)
    fig.savefig(OUT / "causal-cache-baseline.png", dpi=220)
    fig.savefig(OUT / "causal-cache-baseline.svg", metadata={"Date": None})
    plt.close(fig)
    p = OUT / "causal-cache-baseline.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    files = [OLD, DATA / "native.csv", DATA / "timing.csv"]
    (OUT / "provenance.json").write_text(json.dumps(dict(posthoc=True, new_model_calls=0,
        script_sha256=sha(Path(__file__)), inputs={str(p.relative_to(ROOT)): sha(p) for p in files},
        scope="Same 32 old 269 cases. Exact affected-subgraph reference in real arithmetic; changed-shape Float32 checks remain separate. Accuracy of269 remains valid."), indent=2) + "\n")


if __name__ == "__main__":
    main()
