"""Posthoc native context/carry/source summaries and scientific figures."""

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


def med(data, keys):
    return {k:float(np.median([float(r[k]) for r in data])) for k in keys}


def main(local, carried, source, figures):
    for path in (local, carried, source):
        assert json.loads((path/"audit.json").read_text())["status"] == "passed"
        assert not (path/"description.json").exists()
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    lr, nr = rows(local/"local-effects.csv"), rows(local/"native-outcomes.csv")
    names = ("earlier", "self", "coupling", "weight", "value", "mixed")
    s247 = dict(local={n:med([r for r in lr if r["term"] == n], ("relative_error", "cosine", "relative_norm", "signed_share")) for n in names},
        final={n:med([r for r in nr if r["edit"] == n], ("verdict_remaining_ratio", "verdict_native_relative", "margin_response_reduction", "native_margin_error")) for n in (*names, "all")})
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.5), layout="constrained")
    for i, term in enumerate(("weight", "value", "mixed")):
        values = [float(r["signed_share"]) for r in lr if r["term"] == term]
        axes[0].scatter(i+np.linspace(-.12, .12, len(values)), values, s=14, alpha=.6, color="#247e82")
        axes[0].plot([i-.2, i+.2], [np.median(values)]*2, color="#174c50", lw=2)
    axes[0].axhline(0, color="#888", lw=.8)
    axes[0].set(xticks=[0,1,2], xticklabels=["Weight change", "Value change", "Product"], ylabel="Projection / native local change norm²", title="Earlier-context read at the native query")
    for i, term in enumerate(("earlier", "coupling", "all")):
        values = [float(r["native_margin_error"]) for r in nr if r["edit"] == term]
        axes[1].scatter(i+np.linspace(-.12, .12, len(values)), values, s=14, alpha=.6, color="#b87932")
        axes[1].plot([i-.2, i+.2], [np.median(values)]*2, color="#70451b", lw=2)
    axes[1].set(xticks=[0,1,2], xticklabels=["Context read", "Coupling", "All local change"], ylabel="Final margin change from native True (nats)", title="Direct removal at attention10")
    fig.suptitle("247: one native context read has a large local effect, a small final effect")
    for ext in ("png", "svg"):
        fig.savefig(figures/("247-native-context."+ext), dpi=180)
    plt.close(fig)
    carry = rows(carried/"native-outcomes.csv")
    forecast = rows(source/"native-outcomes.csv")
    context = rows(source/"context-recovery.csv")
    s248 = dict(medians={a:med([r for r in carry if r["arm"] == a], ("all_relative", "verdict_relative", "verdict_cosine", "margin_error")) for a in ("P", "R")})
    s249 = dict(native={a:med([r for r in forecast if r["arm"] == a], ("all_relative", "all_cosine", "verdict_relative", "verdict_cosine", "margin_error")) for a in ("S", "C")},
        context={a:med([r for r in context if r["arm"] == a], ("all_relative", "verdict_relative", "margin_error")) for a in ("S", "C")})
    groups = ([r for r in carry if r["arm"] == "P"], [r for r in carry if r["arm"] == "R"],
        [r for r in forecast if r["arm"] == "S"], [r for r in forecast if r["arm"] == "C"])
    labels = ["Earlier context\nOracle P", "Verdict carry\nOracle R", "Source only\nComputed S", "Other rows\nOracle C"]
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.8), layout="constrained")
    for i, group in enumerate(groups):
        for axis, metric, factor in ((axes[0], "verdict_relative", 100), (axes[1], "margin_error", 1)):
            values = [float(r[metric])*factor for r in group]
            axis.scatter(i+np.linspace(-.12, .12, len(values)), values, s=14, alpha=.6, color="#247e82" if i != 2 else "#b87932")
            axis.plot([i-.2, i+.2], [np.median(values)]*2, color="#333", lw=2)
    for axis in axes[:2]:
        axis.set_xticks(range(4), labels, fontsize=8)
    axes[0].axhline(5, color="#555", ls=":")
    axes[0].set(title="Final native verdict displacement", ylabel="Relative error (%)")
    axes[1].axhline(.1, color="#555", ls=":")
    axes[1].set(title="Finite score margin", ylabel="Absolute margin error (nats)", yscale="log")
    for i, metric in enumerate(("all_relative", "verdict_relative")):
        values = [float(r[metric])*100 for r in groups[2]]
        axes[2].scatter(i+np.linspace(-.12, .12, len(values)), values, color="#b87932", s=14, alpha=.6)
        axes[2].plot([i-.2, i+.2], [np.median(values)]*2, color="#70451b", lw=2)
    axes[2].axhline(5, color="#555", ls=":")
    axes[2].set(xticks=[0,1], xticklabels=["All receivers", "Verdict row"], title="Source-only forecast still fails", ylabel="Relative error (%)")
    fig.suptitle("248–249: source/context states regenerate much of the native verdict; strict accuracy fails")
    for ext in ("png", "svg"):
        fig.savefig(figures/("248-249-native-states."+ext), dpi=180)
    plt.close(fig)
    for path, summary in ((local, s247), (carried, s248), (source, s249)):
        summary.update(posthoc=True, model_calls=0, source_sha256=sha(__file__), audit_sha256=sha(path/"audit.json"),
            csv_hashes=json.loads((path/"audit.json").read_text())["csv_hashes"])
        write_json(path/"description.json", summary)
    print(json.dumps(dict(local=s247, carry=s248, source=s249), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-247", "input-248", "input-249", "figures"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    main(args.input_247.resolve(), args.input_248.resolve(), args.input_249.resolve(), args.figures.resolve())
