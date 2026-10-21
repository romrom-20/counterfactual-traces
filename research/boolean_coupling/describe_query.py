"""Posthoc query-generation and predicted-curvature summaries; all cases retained."""

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


def med(data):
    metrics = ("all_relative", "verdict_relative", "all_cosine", "verdict_cosine", "margin_error", "source_final_error", "verdict_signed_share")
    return {k:float(np.median([float(r[k]) for r in data])) for k in metrics}


def main(run, figures):
    audit, design = (json.loads((run/n).read_text()) for n in ("audit.json", "design.json"))
    assert audit["status"] == "passed" and not (run/"description.json").exists()
    num = design["experiment"]
    arms = ("A", "B", "C") if num == 253 else ("D0", "P1", "P2", "R")
    data, checks = rows(run/"outcomes.csv"), []
    pooled = {a:med([r for r in data if r["arm"] == a]) for a in arms}
    strata = {f"{s}_{v}":{a:med([r for r in data if r["arm"] == a and r["schema"] == s and r["reverse"] == str(v)]) for a in arms} for s in ("generic", "descriptive") for v in (0, 1)}
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout="constrained")
    labels = ["Tangent Q / live K\nA (primary)", "Tangent Q / False K\nB", "Finite Q / False K\nC"] if num == 253 else ["Tangent seed\nD0", "One correction\nP1", "Two corrections\nP2 (primary)", "Matched random\nR"]
    for i, arm in enumerate(arms):
        selected = [r for r in data if r["arm"] == arm]
        for ax, metric, factor in zip(axes, ("all_relative", "verdict_relative", "margin_error"), (100, 100, 1), strict=True):
            values = [float(r[metric])*factor for r in selected]
            ax.scatter(i+np.linspace(-.13, .13, len(values)), values, s=15, alpha=.65, c=["#257c84" if r["schema"] == "generic" else "#bd7832" for r in selected])
            ax.plot([i-.22, i+.22], [np.median(values)]*2, color="#333", lw=2)
    for ax in axes:
        ax.set_xticks(range(len(arms)), labels, fontsize=9)
    axes[0].axhline(5, color="#888", ls=":")
    axes[1].axhline(5, color="#888", ls=":")
    axes[2].axhline(.1, color="#888", ls=":")
    axes[0].set(title="Whole receiving displacement", ylabel="Relative native-response error (%)")
    axes[1].set(title="Verdict displacement", ylabel="Relative native-response error (%)")
    axes[2].set(title="Finite verdict margin", ylabel="Absolute margin error (nats)")
    fig.suptitle(f"{num}: sealed False-only forecasts on 32 new literal pairs\nFinite values, normalization, MLPs and own-state relays; one pinned model")
    for ext in ("png", "svg"):
        fig.savefig(figures/(f"{num}-query-generation."+ext), dpi=180)
    plt.close(fig)
    write_json(run/"description.json", dict(posthoc=True, model_calls=0, checks=checks, source_sha256=sha(__file__), audit_sha256=sha(run/"audit.json"),
        csv_hashes=audit["csv_hashes"], pooled=pooled, strata=strata, interpretation="New literals in dependent templates; no learned circuit or independent replication."))
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
