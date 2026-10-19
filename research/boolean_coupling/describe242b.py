"""Explicitly posthoc layer profiles and global policy interactions for242b."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .audit229 import checked, sha
from .audit242b import geometry, read
from .core225b import write_json

ARMS = ("100", "101", "110", "100_raw")
LABELS = ("Primary", "Finite SiLU", "Finite softmax", "Raw softmax Taylor")
COLORS = ("#315fc4", "#00846e", "#ca6b1c", "#9874a3")


def main(run, figures):
    assert not (run / "description.json").exists()
    audit, status = read(run / "audit.json"), read(run / "forecast-status.json")
    assert audit["status"] == "passed"
    checks, profiles, interactions, raw_statistics = [], [], [], []
    inputs = {}
    for ent in status["cases"]:
        key = ent["key"]
        paths = {p: run / p / (key + ".npz") for p in ("forecasts", "observations")}
        inputs[key] = {p: sha(path) for p, path in paths.items()}
        meta = read(paths["forecasts"].with_suffix(".json"))
        with np.load(paths["forecasts"]) as f, np.load(paths["observations"]) as o:
            base, true = f["false_trajectory"].astype(float), o["true_trajectory"].astype(float)
            for arm in ARMS:
                p = f[arm + "_trajectory"].astype(float)
                for layer in range(28):
                    for scope, pick in (("source", 0), ("receivers", slice(1, None)), ("verdict", -1)):
                        rel, cosine = geometry(p[layer, pick] - base[layer, pick], true[layer, pick] - base[layer, pick])
                        profiles.append(dict(key=key, **meta["group"], arm=arm, layer=layer+1, scope=scope,
                            relative=rel, cosine=cosine, absolute_error=float(np.linalg.norm(p[layer, pick]-true[layer, pick]))))
            p00, p01, p10 = (f[a + "_trajectory"][-1, 1:].astype(float) for a in ("100", "101", "110"))
            target = true[-1, 1:] - p00
            terms = dict(softmax=p10-p00, silu=p01-p00, interaction=true[-1, 1:]-p10-p01+p00)
            checked(checks, "global_policy_vector_closure", sum(terms.values()), target, 1e-10)
            for scope, pick in (("receivers", slice(None)), ("verdict", -1)):
                d = target[pick]
                den = float(np.vdot(d, d))
                for name, v in terms.items():
                    interactions.append(dict(key=key, **meta["group"], scope=scope, term=name,
                        relative_norm=float(np.linalg.norm(v[pick])/np.sqrt(den)),
                        signed_share=float(np.vdot(v[pick], d)/den)))
            raw = [s for s in meta["primitive_statistics"] if s["arm"] == "100_raw"]
            raw_statistics.append(dict(key=key, negative_entries=sum(s["negative_entries"] for s in raw),
                negative_layers=sum(s["negative_entries"] > 0 for s in raw), minimum=min(s["minimum"] for s in raw)))
    outcomes = list(csv.DictReader((run / "outcomes.csv").open()))
    medians = {arm: {k: float(np.median([float(r[k]) for r in outcomes if r["arm"] == arm]))
        for k in ("all_relative", "verdict_relative", "margin_error")} for arm in ARMS}
    for name, rows in (("layer-profiles", profiles), ("policy-interactions", interactions), ("raw-softmax-statistics", raw_statistics)):
        with (run / (name + ".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), layout="constrained")
    for arm, label, color in zip(ARMS, LABELS, COLORS, strict=True):
        y = [np.median([r["relative"] for r in profiles if r["arm"] == arm and r["layer"] == layer_number and r["scope"] == "receivers"]) for layer_number in range(1, 29)]
        axes[0].plot(range(1, 29), np.asarray(y)*100, color=color, label=label, lw=2)
    axes[0].axhline(5, color="#555", ls=":", label="Frozen 5% bound")
    axes[0].set(xlabel="Block", ylabel="Median displacement error (%)", title="Errors across receiving rows")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, loc="lower right")
    for index, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS, strict=True)):
        sub = [r for r in outcomes if r["arm"] == arm]
        offsets = np.linspace(-.14, .14, len(sub))
        for ax, key, scale in ((axes[1], "all_relative", 100), (axes[2], "margin_error", 1)):
            values = np.array([float(r[key]) for r in sub])*scale
            ax.scatter(index+offsets, values, color=color, alpha=.45, s=12)
            ax.plot([index-.2, index+.2], [np.median(values)]*2, color=color, lw=3)
    ticks = ["Primary\n100", "+finite SiLU\n101", "+finite softmax\n110", "Raw Taylor\n100_raw"]
    for ax in axes[1:]:
        ax.set_xticks(range(4), ticks, fontsize=8)
    axes[1].axhline(5, color="#555", ls=":")
    axes[1].set(title="Final residual: 0/32 passes in every arm", ylabel="Whole receiving displacement error (%)")
    axes[2].axhline(.1, color="#555", ls=":")
    axes[2].set(title="Final finite verdict readout", ylabel="Absolute margin error (nats)")
    fig.suptitle("242b: a uniform rule stays bounded but does not predict the native response", fontsize=13)
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(figures / ("242b-uniform-primitive-families." + suffix), dpi=180)
    plt.close(fig)
    write_json(run / "description.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__),
        audit_sha256=sha(run / "audit.json"), input_arrays=inputs, checks=checks, medians=medians,
        raw_negative_cases=sum(r["negative_entries"] > 0 for r in raw_statistics),
        interpretation="Layer profiles and global policy interaction identities; no unique local causal attribution."))
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
