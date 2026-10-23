"""Posthoc figure of273's source geometry, causal effect and native margin."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/native-value-interchange-273"
OUT = ROOT / "docs/figures/native-values"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    source, effect, native = (rows(DATA / n) for n in ("source-geometry.csv", "effects.csv", "native.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    colors = dict(generic="#2a6f91", descriptive="#aa542a")
    for h, style in ((0, "-"), (1, "--")):
        curve = [median(float(r["cosine"]) for r in source if int(r["layer"]) == i and int(r["head"]) == h) for i in range(1, 29)]
        axes[0].plot(range(1, 29), curve, style, label=f"KV head {h}", color=("#72548d" if h == 0 else "#555555"))
    axes[0].set(title="Source value-change geometry", xlabel="Block", ylabel="Other-schema / own displacement cosine")
    axes[0].set_ylim(-.05, 1.05)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for i, arm in enumerate(("X", "D", "A", "R")):
        group = [r for r in effect if r["arm"] == arm]
        for j, r in enumerate(group):
            offset = ((j * 13) % 31 - 15) / 150
            axes[1].scatter(i + offset, 100 * float(r["all_relative"]), color=colors[r["schema"]],
                            marker=("o" if int(r["reverse"]) == 0 else "^"), alpha=.7, s=15)
        value = median(100 * float(r["all_relative"]) for r in group)
        axes[1].plot([i - .17, i + .17], [value, value], color="black", linewidth=2)
    axes[1].axhline(25, color="#666666", linewidth=1, linestyle=":")
    axes[1].set(title="Native source-value effect U−Z", ylabel="Whole receiving-vector error (%)", xlabel="Source value change installed")
    axes[1].set_xticks(range(4), ["Donor\nX", "Donor axis\nown norm D", "Own axis\ndonor norm A", "Random axis\nown norm R"])
    arm_rows = [r for r in native if r["arm"] == "X"]
    for r in arm_rows:
        axes[2].scatter(float(r["native_margin_response"]), float(r["margin_response"]), color=colors[r["schema"]],
                        marker=("o" if int(r["reverse"]) == 0 else "^"), alpha=.8, s=24)
    vals = [float(r[k]) for r in arm_rows for k in ("native_margin_response", "margin_response")]
    lo, hi = min(0., min(vals)), max(vals)
    pad = .04 * max(hi - lo, 1.)
    axes[2].plot([lo - pad, hi + pad], [lo - pad, hi + pad], ":", color="#666666", linewidth=1)
    axes[2].set(title="Donor substitution: final verdict", xlabel="Native True−False margin (nats)", ylabel="X−False margin (nats)")
    axes[2].set_xlim(lo - pad, hi + pad)
    axes[2].set_ylim(lo - pad, hi + pad)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.15)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=s.title()) for s, c in colors.items()]
    handles.append(plt.Line2D([], [], color="black", linewidth=2, label="Across-case median"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.5, .045))
    fig.suptitle("Is the Boolean signal portable, or does its source context matter?", fontsize=12, y=.98)
    fig.text(.5, .015, "32 old contexts · only source V changes swapped · all receiving computation finite · prompt backgrounds also differ in position/length",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .15, 1, .94), w_pad=2)
    fig.savefig(OUT / "source-value-portability.png", dpi=220)
    fig.savefig(OUT / "source-value-portability.svg", metadata={"Date": None})
    plt.close(fig)
    p = OUT / "source-value-portability.svg"
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(dict(posthoc=True, model_calls=0,
        source_sha256=sha(Path(__file__)), inputs={str((DATA / n).relative_to(ROOT)): sha(DATA / n)
        for n in ("source-geometry.csv", "effects.csv", "native.csv")},
        scope="Source geometry, native conditional value-path interventions and actual finite margins. No circuit or novelty claim."), indent=2) + "\n")


if __name__ == "__main__":
    main()
