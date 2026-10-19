"""Posthoc summaries/figure from243's separately audited signed effects."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .audit229 import sha
from .core225b import write_json


def csv_rows(path):
    return list(csv.DictReader(path.open()))


def main(run, figures):
    import json
    assert not (run / "description.json").exists()
    audit = json.loads((run / "audit.json").read_text())
    assert audit["status"] == "passed"
    native = csv_rows(run / "native-outcomes.csv")
    signed = csv_rows(run / "signed-effects.csv")
    recovery = csv_rows(run / "contrast-recovery.csv")
    medians = {}
    for arm in ("QQ", "FQ", "QF", "FF"):
        sub = [r for r in native if r["arm"] == arm and r["status"] == "complete"]
        medians[arm] = dict(completed=len(sub), **{name: float(np.median([float(r[name]) for r in sub]))
            for name in ("all_relative", "verdict_relative", "margin_error")})
    effects = {term: {name: float(np.median([float(r[name]) for r in signed if r["term"] == term and r["scope"] == "all" and r["layer"] == "28"]))
        for name in ("signed_share", "relative_norm", "relative_error", "cosine")} for term in ("source_only", "receiver_only", "interaction")}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.7), layout="constrained")
    arms, colors = ("QQ", "FQ", "QF", "FF"), ("#6d80a1", "#cc791e", "#4474c4", "#00846e")
    for index, (arm, color) in enumerate(zip(arms, colors, strict=True)):
        sub = [r for r in native if r["arm"] == arm and r["status"] == "complete"]
        for axis, metric, scale in ((axes[0], "all_relative", 100), (axes[1], "margin_error", 1)):
            values = [float(r[metric])*scale for r in sub]
            if values:
                axis.scatter(index+np.linspace(-.14, .14, len(values)), values, color=color, alpha=.5, s=13)
                axis.plot([index-.2, index+.2], [np.median(values)]*2, color=color, lw=3)
    labels = ["Quadratic\nQQ", "Source finite\nFQ", "Receivers finite\nQF", "Both finite\nFF"]
    for axis in axes[:2]:
        axis.set_xticks(range(4), labels, fontsize=8)
    axes[0].axhline(5, color="#555", ls=":")
    axes[0].set(title="Native receiving displacement", ylabel="Relative error (%)")
    axes[1].axhline(.1, color="#555", ls=":")
    axes[1].set(title="Final finite readout", ylabel="Absolute margin error (nats)")
    terms = ("source_only", "receiver_only", "interaction")
    for index, term in enumerate(terms):
        values = [float(r["signed_share"]) for r in signed if r["term"] == term and r["scope"] == "all" and r["layer"] == "28"]
        if values:
            axes[2].scatter(index+np.linspace(-.14, .14, len(values)), values, color=colors[index+1], alpha=.5, s=13)
            axes[2].plot([index-.2, index+.2], [np.median(values)]*2, color=colors[index+1], lw=3)
    axes[2].axhline(0, color="#999", lw=.8)
    axes[2].axhline(1, color="#555", ls=":")
    axes[2].set_xticks(range(3), ["Source only", "Receivers only", "Interaction"], fontsize=8)
    axes[2].set(title="Signed effect on the FF−QQ contrast", ylabel="Projection / squared contrast norm")
    fig.suptitle("243: separate SiLU at the changed source from SiLU at receiving tokens", fontsize=13)
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(figures / ("243-regional-silu." + suffix), dpi=180)
    plt.close(fig)
    write_json(run / "description.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__),
        audit_sha256=sha(run / "audit.json"), input_csv_sha256=audit["csv_hashes"], medians=medians,
        signed_effect_medians=effects, small_margin_contrast_cases=sum(abs(float(r.get("margin_target") or 0)) < .10
            for r in recovery if r["account"] == "source_only"),
        interpretation="Posthoc summaries of frozen contrasts; no feature or fresh-prediction claim."))
    print(json.dumps(dict(medians=medians, effects=effects), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
