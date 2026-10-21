"""Posthoc complete-case native outcomes and same-input local SiLU capture plots."""

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


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(run, figures):
    audit = read(run/"audit.json")
    assert audit["status"] == "passed" and not (run/"description.json").exists()
    for name, h in audit["csv_hashes"].items():
        assert sha(run/(name+".csv")) == h
    data, local = rows(run/"outcomes.csv"), rows(run/"signed-local-vectors.csv")
    arms, metrics = ("C", "Q", "S", "R"), ("all_relative", "verdict_relative", "margin_error")
    pooled = {}
    for arm in arms:
        selected = [r for r in data if r["arm"] == arm]
        assert len(selected) == 32
        pooled[arm] = dict(passes=sum(r["passes"] == "True" for r in selected),
            medians={k:float(np.median([float(r[k]) for r in selected])) for k in metrics},
            maxima={k:max(float(r[k]) for r in selected) for k in metrics})
    count, total, maximum_fraction = 0, 0, 0.
    for entry in read(run/"forecast-status.json")["cases"]:
        with np.load(run/"forecasts"/(entry["key"]+".npz")) as f:
            for layer in range(1, 29):
                mask = f[f"S_mask_counts{layer}"]
                cells = len(mask)*8960
                assert mask.sum() == int(.01*cells)
                count += int(mask.sum())
                total += cells
                maximum_fraction = max(maximum_fraction, float(mask.sum()/cells))
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), layout="constrained")
    labels = ("Finite SiLU\nC", "Quadratic\nQ", "Ranked 1%\nS (primary)", "Random 1%\nR")
    floored = 0
    for i, arm in enumerate(arms):
        selected = [r for r in data if r["arm"] == arm]
        for ax, metric, factor in zip(axes, metrics, (100, 100, 1), strict=True):
            values = np.array([float(r[metric])*factor for r in selected])
            assert np.isfinite(values).all() and np.all(values >= 0)
            floored += int(np.sum(values < 1e-6))
            ax.scatter(i+np.linspace(-.13, .13, len(values)), np.maximum(values, 1e-6), s=18, alpha=.7,
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
    fig.suptitle("258: fixed 1% finite-SiLU budget at every receiving MLP\n32 sealed new literal pairs · cubic queries held fixed · finite source/K/V/projections")
    hashes = {}
    for ext in ("png", "svg"):
        path = figures/("258-receiver-mlp."+ext)
        assert not path.exists()
        fig.savefig(path, dpi=180)
        hashes[path.name] = sha(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    captures = {}
    colors = dict(C="#217b80", Q="#bd7832", S="#315fc0", R="#888888")
    for ax, scope in zip(axes, ("all", "verdict"), strict=True):
        for arm in arms:
            curve, eligible = [], []
            selected = [r for r in local if r["arm"] == arm and r["scope"] == scope and r["eligible"] == "True"]
            for layer in range(1, 29):
                group = [r for r in selected if r["layer"] == str(layer)]
                eligible.append(len(group))
                curve.append(float(np.median([float(r["restore_relative_gap_error"])*100 for r in group])) if group else np.nan)
            ax.plot(range(1, 29), curve, label=f"{arm} input / {'random' if arm == 'R' else 'ranked'} mask", color=colors[arm], lw=2 if arm == "S" else 1.3)
            captures[arm+"_"+scope] = dict(eligible_by_layer=eligible, median_remaining_local_gap=float(np.median([float(r["restore_relative_gap_error"]) for r in selected])),
                median_signed_share=float(np.median([float(r["restore_signed_share"]) for r in selected])))
        ax.set(title="Whole receiving write" if scope == "all" else "Verdict-position write", xlabel="Block", ylabel="Remaining same-input SiLU write gap (%)")
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle("Conditional local capture, at each arm's own input\nMedians over eligible case/layers; not native final-response error")
    for ext in ("png", "svg"):
        path = figures/("258-local-capture."+ext)
        assert not path.exists()
        fig.savefig(path, dpi=180)
        hashes[path.name] = sha(path)
    plt.close(fig)
    write_json(run/"description.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__), audit_sha256=sha(run/"audit.json"),
        csv_hashes=audit["csv_hashes"], pooled=pooled, local_capture=captures, figure_sha256=hashes,
        selected_finite_receiver_coordinates=count, receiver_coordinates=total, fraction=count/total, maximum_layer_fraction=maximum_fraction,
        plot_floor=1e-6, values_below_plot_floor=floored, all_cases=True,
        interpretation="Uniform target-blind finite-node budget; full public SiLU still executes in instrumented code, not a speedup or learned circuit."))
    print(json.dumps(dict(pooled=pooled, local_capture=captures, finite_fraction=count/total), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
