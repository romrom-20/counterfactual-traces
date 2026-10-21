"""Presentation-only 254 figure; retain the original plot and all observations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run, figures):
    audit = json.loads((run / "audit.json").read_text())
    data_path = run / "outcomes.csv"
    assert audit["status"] == "passed"
    assert sha(data_path) == audit["csv_hashes"]["outcomes"]
    with data_path.open() as stream:
        data = list(csv.DictReader(stream))
    figures.mkdir(parents=True, exist_ok=True)
    arms = ("D0", "P1", "P2", "R")
    labels = ("Tangent\nD0", "1 correction\nP1", "2 corrections\nP2 (primary)", "Random\nR")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), layout="constrained")
    metrics = ("all_relative", "verdict_relative", "margin_error")
    for index, arm in enumerate(arms):
        selected = [row for row in data if row["arm"] == arm]
        assert len(selected) == 32
        colors = ["#257c84" if row["schema"] == "generic" else "#bd7832" for row in selected]
        for ax, metric, factor in zip(axes, metrics, (100, 100, 1), strict=True):
            values = np.array([float(row[metric]) * factor for row in selected])
            assert np.all(np.isfinite(values)) and np.all(values > 0)
            ax.scatter(index + np.linspace(-.13, .13, len(values)), values, s=18, alpha=.7, c=colors)
            ax.plot([index - .22, index + .22], [np.median(values)] * 2, color="#333", lw=2)
    for ax, bound in zip(axes, (5, 5, .1), strict=True):
        ax.set_xticks(range(4), labels, fontsize=10)
        ax.set_yscale("log")
        ax.axhline(bound, color="#888", ls=":")
        ax.grid(axis="y", alpha=.15)
    axes[0].set(title="Whole receiving displacement", ylabel="Relative native-response error (%) · log scale")
    axes[1].set(title="Verdict displacement", ylabel="Relative native-response error (%) · log scale")
    axes[2].set(title="Finite verdict margin", ylabel="Absolute error (nats) · log scale")
    fig.suptitle("254: predict the omitted query curvature from the preceding forecast\n32 sealed new literal pairs · rich finite-component budget · one pinned model")
    outputs = [figures / f"254-query-curvature.{ext}" for ext in ("png", "svg")]
    for path in outputs:
        assert not path.exists(), "Never overwrite a published figure"
        fig.savefig(path, dpi=180)
    plt.close(fig)
    provenance = dict(
        posthoc=True, model_calls=0,
        purpose="Shorter labels and explicit log axes; original plot and scientific description retained.",
        source_sha256=sha(Path(__file__)), outcomes_sha256=sha(data_path),
        audit_sha256=sha(run / "audit.json"),
        output_sha256={path.name: sha(path) for path in outputs},
        scope="All 32 cases per arm; no filtering, fit, measurement or forecast changes.",
    )
    (figures / "presentation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.figures.resolve())
