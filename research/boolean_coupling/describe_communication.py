"""Posthoc descriptive summaries and figures of the finite communication test."""

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


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def summary(data):
    keys = ("signed_share", "relative_error", "relative_norm", "cosine", "remaining_margin_fraction", "margin_error")
    return {k:dict(median=float(np.median([float(r[k]) for r in data])), minimum=float(min(float(r[k]) for r in data)), maximum=float(max(float(r[k]) for r in data))) for k in keys}


def main(run, figures):
    audit = json.loads((run/"audit.json").read_text())
    assert audit["status"] == "passed" and not (run/"description.json").exists()
    data = rows(run/"outcomes.csv")
    arms = ("K", "V", "B", "D", "L", "random0", "random1", "random2")
    verdict = [r for r in data if r["scope"] == "verdict"]
    pooled = {a:summary([r for r in verdict if r["arm"] == a]) for a in arms}
    strata = {f"{s}_{v}":{a:summary([r for r in verdict if r["arm"] == a and r["schema"] == s and r["reverse"] == str(v)]) for a in arms} for s in ("generic", "descriptive") for v in (0, 1)}
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), layout="constrained")
    shown = ("K", "V", "B", "D", "L")
    labels = ["Keys", "Values", "Both", "Verdict reads", "Relay reads"]
    colors = {"generic":"#257c84", "descriptive":"#bd7832"}
    for i, arm in enumerate(shown):
        selected = [r for r in verdict if r["arm"] == arm]
        for axis, metric in zip(axes, ("signed_share", "remaining_margin_fraction"), strict=True):
            vals = [float(r[metric]) for r in selected]
            axis.scatter(i+np.linspace(-.15, .15, len(vals)), vals, s=17, alpha=.65, c=[colors[r["schema"]] for r in selected])
            axis.plot([i-.23, i+.23], [np.median(vals)]*2, color="#292929", lw=2)
    for axis in axes:
        axis.axhline(0, color="#888", lw=.7)
        axis.axhline(1, color="#888", lw=.7, ls="--")
        axis.set_xticks(range(5), labels, rotation=12)
    axes[0].set(title="Remaining native verdict-vector projection", ylabel="Signed projection / native response norm²")
    axes[1].set(title="Remaining finite margin response", ylabel="Signed margin change / native margin change")
    fig.suptitle("250: remove source communication at blocks 10–28\nAll 32 old contexts; native True receiving history retained at the boundary")
    for ext in ("png", "svg"):
        fig.savefig(figures/("250-source-communication."+ext), dpi=180)
    plt.close(fig)
    write_json(run/"description.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__), audit_sha256=sha(run/"audit.json"),
        csv_hashes=audit["csv_hashes"], pooled=pooled, strata=strata,
        warning="Separately pooled medians are not additive allocations; source clamps are retrospective interventions."))
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
