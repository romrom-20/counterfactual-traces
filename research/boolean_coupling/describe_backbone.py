"""Posthoc all-context figure for the270/271 primitive and controller tests."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/backbone-feedback-270-271"
OUT = ROOT / "docs/figures/backbone"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    files = [DATA / folder / name for folder in
             ("backbone-curvature-feedback-270", "amplitude-query-mediation-271")
             for name in ("native.csv", "restoration.csv")]
    old_native, old_effect, new_native, new_effect = map(rows, files)
    case_keys = list(dict.fromkeys(r["key"] for r in old_native))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for key in case_keys:
        meta = next(r for r in old_native if r["key"] == key)
        color = "#2a6f91" if meta["schema"] == "generic" else "#aa542a"
        reverse = int(meta["reverse"])
        label = meta["schema"].capitalize() + (", reversed" if reverse else ", forward")
        style = dict(color=color, marker="^" if reverse else "o", markersize=5,
                     linestyle="--" if reverse else "-", linewidth=1.25, label=label)
        native = {(r["key"], r["arm"]): r for r in old_native + new_native}
        arms = ("H", "B", "J", "G", "X", "R", "S", "Q", "C")
        axes[0].plot(range(len(arms)), [100 * float(native[key, a]["all_relative"])
                                      for a in arms], **style)
        effect = {(r["key"], r["arm"]): r for r in old_effect + new_effect}
        arms = ("G", "X", "S", "Q", "C")
        axes[1].plot(range(len(arms)), [100 * float(effect[key, a]["all_relative"])
                                      for a in arms], **style)
    axes[0].set(title="Native receiving-vector error", ylabel="Relative vector error (%)",
                xlabel="Defined continuation policy", ylim=(0, 9))
    axes[0].set_xticks(range(9), ["Finite\nH", "Full\nB", "Off\nJ", "Gate\nG", "Cross\nX",
                                 "Random\nR", "Amps\nS", "Queries\nQ", "Both\nC"])
    axes[0].axhline(5, color="#666666", linestyle=":", linewidth=1)
    axes[0].text(.05, 5.13, "5% whole-vector bound", fontsize=8, color="#555555")
    axes[1].set(title="Error in recovering the missing B − J effect",
                ylabel="Relative error in B − J (%)", xlabel="Restoration or controller clamp",
                ylim=(0, 110))
    axes[1].set_xticks(range(5), ["Gate\nG", "Cross\nX", "Amplitudes\nS", "Queries\nQ", "Both\nC"])
    axes[1].axhline(25, color="#666666", linestyle=":", linewidth=1)
    axes[1].text(.05, 27, "25% whole-vector bound", fontsize=8, color="#555555")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, .035))
    fig.suptitle("Nonlinear backbone writes remain consequential after the feedback clamp",
                 fontsize=12, x=.5, y=.99)
    fig.text(.5, .012, "Four old dependent contexts · whole-vector criteria shown; joint tests also require verdict and finite-margin accuracy",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .13, 1, .96), w_pad=2)
    fig.savefig(OUT / "backbone-feedback.png", dpi=220)
    fig.savefig(OUT / "backbone-feedback.svg", metadata={"Date": None})
    plt.close(fig)
    path = OUT / "backbone-feedback.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(dict(posthoc=True, new_model_calls=0,
        script_sha256=sha(Path(__file__)), inputs={str(p.relative_to(ROOT)): sha(p) for p in files},
        omitted_from_left="L seed:0/4 native joint passes,31.72% median whole error; present in source CSVs.",
        meaning="Every case. Whole-vector axes alone do not establish a joint pass. B/J repeat exactly between270 and271."), indent=2) + "\n")
    print("Rendered all-context backbone/controller figure from published CSVs.")


if __name__ == "__main__":
    main()
