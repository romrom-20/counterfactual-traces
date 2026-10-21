"""Posthoc complete-case summary and figure for the frozen 256 polynomial test."""

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
    with (run/"outcomes.csv").open() as stream:
        data = list(csv.DictReader(stream))
    assert sha(run/"outcomes.csv") == audit["csv_hashes"]["outcomes"]
    arms, metrics = ("L", "Q", "C", "R"), ("all_relative", "verdict_relative", "margin_error")
    pooled = {}
    for arm in arms:
        selected = [row for row in data if row["arm"] == arm]
        assert len(selected) == 32
        pooled[arm] = dict(passes=sum(row["passes"] == "True" for row in selected),
            medians={k:float(np.median([float(r[k]) for r in selected])) for k in metrics},
            maxima={k:max(float(r[k]) for r in selected) for k in metrics})
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), layout="constrained")
    labels = ("Linear\nL", "Quadratic\nQ", "Cubic\nC (primary)", "Random\nR")
    floored = 0
    for i, arm in enumerate(arms):
        selected = [r for r in data if r["arm"] == arm]
        for ax, metric, factor in zip(axes, metrics, (100, 100, 1), strict=True):
            values = np.array([float(r[metric])*factor for r in selected])
            assert np.all(np.isfinite(values)) and np.all(values >= 0)
            floored += int(np.sum(values < 1e-6))
            display = np.maximum(values, 1e-6)
            ax.scatter(i+np.linspace(-.13, .13, len(values)), display, s=18, alpha=.7,
                c=["#257c84" if r["schema"] == "generic" else "#bd7832" for r in selected])
            ax.plot([i-.22, i+.22], [max(float(np.median(values)), 1e-6)]*2, color="#333", lw=2)
    for ax, bound in zip(axes, (5, 5, .1), strict=True):
        ax.set_xticks(range(4), labels, fontsize=10)
        ax.set_yscale("log")
        ax.axhline(bound, color="#888", ls=":")
        ax.grid(axis="y", alpha=.15)
    axes[0].set(title="Whole receiving displacement", ylabel="Relative native-response error (%) · log scale")
    axes[1].set(title="Verdict displacement", ylabel="Relative native-response error (%) · log scale")
    axes[2].set(title="Finite verdict margin", ylabel="Absolute error (nats) · log scale")
    fig.suptitle("256: local query curvature, without a preceding correction trajectory\n32 sealed new literal pairs · finite K/V, MLPs and source computation · one pinned model")
    hashes = {}
    for ext in ("png", "svg"):
        path = figures/("256-local-query-curvature."+ext)
        assert not path.exists()
        fig.savefig(path, dpi=180)
        hashes[path.name] = sha(path)
    plt.close(fig)
    write_json(run/"description.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__), audit_sha256=sha(run/"audit.json"),
        csv_hashes=audit["csv_hashes"], pooled=pooled, figure_sha256=hashes,
        plot_floor=1e-6, values_below_plot_floor=floored, all_cases=True,
        interpretation="Single-pass query-path component test; substantial finite learned computation retained; no compact/general QB solution."))
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
