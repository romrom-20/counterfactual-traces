"""Plot every retrospective259 score variant; no new model or fitted forecast."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .audit229 import sha
from .core225b import write_json


def main(run, figures):
    audit = json.loads((run/"audit.json").read_text())
    assert audit["status"] == "passed" and not (run/"description.json").exists()
    assert sha(run/"score-factors.csv") == audit["csv_sha256"]
    with (run/"score-factors.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    scores = ("BUW", "BU", "BW", "UW", "B", "U", "W", "DUW", "D")
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, scope in zip(axes, ("all", "verdict"), strict=True):
        for offset, arm, color in ((-.18, "C", "#217b80"), (.18, "S", "#315fc0")):
            values = []
            for score in scores:
                selected = [r for r in rows if r["arm"] == arm and r["scope"] == scope and r["score"] == score]
                assert len(selected) == 12 and all(r["eligible"] == "True" for r in selected)
                values.append(100*float(np.median([float(r["remaining_relative"]) for r in selected])))
            ax.barh(np.arange(len(scores))+offset, values, height=.34, color=color, alpha=.8, label=arm+" inputs (12)")
        ax.set_yticks(range(len(scores)), scores)
        ax.invert_yaxis()
        ax.set(title="Whole receiving write" if scope == "all" else "Verdict-position write", xlabel="Remaining same-input SiLU write gap (%)")
        ax.grid(axis="x", alpha=.15)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("259: all fixed 1% ranking variants at 24 existing forecast inputs\nB = remainder bound · U = |up| · W = column norm · D = |gate edit|³")
    hashes = {}
    for ext in ("png", "svg"):
        path = figures/("259-score-factors."+ext)
        assert not path.exists()
        fig.savefig(path, dpi=180)
        hashes[path.name] = sha(path)
    plt.close(fig)
    write_json(run/"description.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__),
        audit_sha256=sha(run/"audit.json"), csv_sha256=sha(run/"score-factors.csv"),
        all_scores=True, all_fixed_inputs=True, figure_sha256=hashes,
        limitation="Conditional local captures at existing forecast inputs, not fresh global prediction or independent samples."))
    print("Plotted all nine score variants, both scopes and both existing forecast arms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
