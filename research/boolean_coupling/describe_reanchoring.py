"""Posthoc all-case source-emitter and intervention figure for274; no model calls."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/source-residual-reanchoring-274"
OUT = ROOT / "docs/figures/reanchoring"


def rows(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    emission, effects, native = (rows(n) for n in ("emission.csv", "effects.csv", "native.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.1))
    arms = ("X", "P", "D", "J")
    labels = ("Copy donor\nvalue X", "Donor step,\nown center P", "Own step,\ndonor center D", "Additive corners\nJ (oracle)")
    colors = dict(generic="#2a6f91", descriptive="#aa542a")
    for ax, data, field, title, ylabel, bound in (
        (axes[0], emission, "relative", "Source emission across 28 layers", "Aggregate value-vector error (%)", 10),
        (axes[1], effects, "all_relative", "Final native source-value effect U−Z", "Whole receiving-vector error (%)", 25),
    ):
        for i, arm in enumerate(arms):
            group = [r for r in data if r["arm"] == arm]
            for j, r in enumerate(group):
                offset = ((j * 13) % 31 - 15) / 140
                ax.scatter(i + offset, 100 * float(r[field]), c=colors[r["schema"]],
                           marker="o" if int(r["reverse"]) == 0 else "^", s=16, alpha=.7)
            value = median(100 * float(r[field]) for r in group)
            ax.plot([i - .19, i + .19], [value, value], color="black", linewidth=2)
        ax.axhline(bound, color="#666666", linestyle=":", linewidth=1)
        ax.set(title=title, ylabel=ylabel, xlabel="Installed source emission rule")
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_ylim(bottom=0)
    for i, arm in enumerate(arms):
        a = [r for r in native if r["arm"] == arm]
        b = [r for r in effects if r["arm"] == arm]
        for offset, group, color, label in ((-.17, b, "#888888", "Looser source-value effect"),
                                            (.17, a, "#2a6f91", "Strict complete native response")):
            count = sum(r["passes"] == "True" for r in group)
            axes[2].bar(i + offset, count, width=.29, color=color, label=label if i == 0 else None)
            axes[2].text(i + offset, count + .6, str(count), ha="center", va="bottom", fontsize=9)
    axes[2].set(title="Passing final vector + margin tests", xlabel="Installed source emission rule", ylabel="Cases passing (of 32)", ylim=(0, 36))
    axes[2].set_xticks(range(4), labels, fontsize=8)
    axes[2].set_yticks((0, 8, 16, 24, 32))
    axes[2].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -.23))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=s.title()) for s, c in colors.items()]
    handles.append(plt.Line2D([], [], color="black", linewidth=2, label="Across-case median"))
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.04, .065))
    fig.suptitle("Reanchoring a donor displacement does not recover the native source message", fontsize=12, y=.98)
    fig.text(.5, .02, "32 old contexts · finite RMSNorm/V and receivers · native source keys retained · D/J use own-True displacement · no fresh forecast",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .15, 1, .94), w_pad=2.4)
    fig.savefig(OUT / "source-reanchoring.png", dpi=220)
    fig.savefig(OUT / "source-reanchoring.svg", metadata={"Date": None})
    plt.close(fig)
    svg = OUT / "source-reanchoring.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(dict(posthoc=True, model_calls=0,
        source_sha256=sha(Path(__file__)), inputs={str((DATA / n).relative_to(ROOT)): sha(DATA / n)
        for n in ("emission.csv", "effects.csv", "native.csv")},
        scope="All32 old cases; emission and distinct causal-effect/native endpoints. No learned-feature or priority claim."), indent=2) + "\n")


if __name__ == "__main__":
    main()
