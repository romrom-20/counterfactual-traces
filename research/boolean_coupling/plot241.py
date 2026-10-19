"""Scientific figure of complete upstream vector transport and concentration."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(run, name):
    with (run / (name + ".csv")).open() as file:
        return list(csv.DictReader(file))


def main(run, out):
    energy, curve, transported = (read(run, name) for name in ("energy", "reconstruction", "transported"))
    selection = json.loads((run / "selection.json").read_text())
    nodes = list(range(2, 31))
    keys = sorted({r["key"] for r in energy})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "vectors241", "savefig.facecolor": "#fbfcfe"})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    values = np.array([[float(next(r for r in energy if r["key"] == key and int(r["node"]) == n)["error_after"]) for n in nodes] for key in keys])
    x = np.array(nodes) / 2 + 1
    ax[0, 0].plot(x, np.median(values, 0), color="#287f77", lw=2)
    ax[0, 0].fill_between(x, *np.percentile(values, [25, 75], axis=0), color="#287f77", alpha=.15)
    ax[0, 0].set(title="Error arriving before MLP16", xlabel="Block (attention then MLP)", ylabel="Whole receiving error norm (residual units)")
    terminal = {key: float(next(r for r in energy if r["key"] == key and int(r["node"]) == 30)["error_after"])**2 for key in keys}
    total = np.zeros(len(nodes))
    for name, label, color in (("transport_energy_change", "Transported-error norm change", "#526f94"),
        ("remainder_energy", "New-remainder energy", "#ba6b4b"), ("cross_energy", "Cross term", "#287f77")):
        v = np.array([[float(next(r for r in energy if r["key"] == key and int(r["node"]) == n)[name]) / terminal[key] for n in nodes] for key in keys])
        mean = np.mean(np.cumsum(v, axis=1), axis=0)
        total += mean
        ax[0, 1].plot(x, mean, label=label, color=color)
    ax[0, 1].plot(x, total, color="#30343b", lw=1, ls="--", label="Sum: cumulative squared error")
    ax[0, 1].set(title="Exact squared-error budget", xlabel="Block (attention then MLP)", ylabel="Mean cumulative term / terminal error²")
    ax[0, 1].legend(fontsize=8)
    for scope, label, color in (("all", "Whole receiving sequence", "#287f77"), ("verdict", "Verdict position", "#ba6b4b")):
        v = np.array([[float(next(r for r in curve if r["key"] == key and int(r["prefix"]) == k)[scope + "_relative"]) for k in range(1, 30)] for key in keys])
        ax[1, 0].plot(range(1, 30), np.median(v, 0), color=color, label=label)
        ax[1, 0].fill_between(range(1, 30), *np.percentile(v, [25, 75], axis=0), color=color, alpha=.12)
    ax[1, 0].axhline(.25, color="#8d3651", ls="--", lw=1)
    ax[1, 0].axvline(4, color="#68717e", ls=":", lw=1)
    ax[1, 0].set(title="Retrospective vector reconstruction", xlabel="Number of shared greedy sites", ylabel="Remaining relative input-error norm")
    ax[1, 0].legend(fontsize=8)
    order = selection["order"][:8]
    labels = [("A" if n % 2 == 0 else "M") + str(n // 2 + 1) for n in order]
    for i, n in enumerate(order):
        v = [float(r["signed_share"]) for r in transported if int(r["node"]) == n]
        ax[1, 1].scatter(i + np.linspace(-.1, .1, len(v)), v, s=9, color="#526f94", alpha=.5, linewidths=0)
        ax[1, 1].plot([i - .18, i + .18], [np.median(v)] * 2, color="#526f94", lw=2)
    ax[1, 1].set_xticks(range(len(order)), labels)
    ax[1, 1].set(title="Transported vectors in greedy selection order", xlabel="Operation (A: attention; M: MLP)", ylabel="Signed projection / terminal error²")
    for axis in ax.flat:
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("An exact error budget before MLP16", fontsize=14, weight="bold", y=.995)
    fig.text(.06, .065, "Source: experiment 241, all 32 old contexts. Curves: median/IQR, except squared-energy budget: mean of per-case normalized terms.", fontsize=9)
    fig.text(.06, .035, "Original False-affine transport and supplied native remainders. The cross term is not a unique causal fraction; selection is retrospective.", fontsize=9)
    fig.tight_layout(rect=(.01, .09, .995, .965), w_pad=2, h_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix("." + suffix), dpi=180, metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
