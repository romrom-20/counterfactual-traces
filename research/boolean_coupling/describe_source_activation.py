"""Posthoc all-case279 visualization and explicit materiality failures; zero model calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-activation-forecast-279"
ARMS = ("L", "Q", "C", "S", "R0", "R1", "R2")
LABELS = ("Linear", "Quad", "Cubic", "Ranked90", "Rnd0", "Rnd1", "Rnd2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def table(path, data):
    assert not path.exists()
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main(run):
    assert sha(run / "audit.json") == sha(DEST / "audit.json")
    audit = read(DEST / "audit.json")
    assert audit["status"] == "passed"
    data = {n: rows(DEST / (n + ".csv")) for n in ("native", "local", "paired", "effect", "combined")}
    eligibility = []
    for row in data["combined"]:
        e = next(r for r in data["effect"] if (r["key"], r["arm"]) == (row["key"], row["arm"]))
        vectors = all(float(e[s + "_norm"]) >= .001 and float(e[s + "_relative"]) <= .25 and float(e[s + "_cosine"]) >= .95 for s in ("all", "verdict"))
        eligibility.append(dict(**row, effect_vectors_pass=bool(vectors), material_margin_pass=abs(float(e["target_margin"])) >= .02,
            target_effect_margin=float(e["target_margin"]), effect_margin_bound_pass=float(e["margin_error"]) <= float(e["margin_bound"])))
    table(DEST / "eligibility.csv", eligibility)
    scores = rows(DEST / "forecast-scores.csv")
    native_arrays = {}
    for entry in read(run / "native-status.json")["cases"]:
        p = run / "observations" / (entry["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"]
        native_arrays[str(p.relative_to(run))] = sha(p)
        with np.load(p) as d:
            a, b = map(float, d["T_scores"])
            scores.append(dict(key=entry["key"], arm="T", false_score=str(a), true_score=str(b), margin=str(b - a)))
    table(DEST / "scores.csv", scores)
    figdir = ROOT / "docs/figures/source-activation"
    assert not figdir.exists()
    figdir.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "svg.hashsalt": "source-activation279"})
    specs = (("local", "relative", 100., "First-MLP write-change error (%)", 10.),
             ("native", "all_relative", 100., "Native receiving-vector error (%)", 5.),
             ("native", "margin_error", 1., "Native verdict margin error (nats)", .1),
             ("paired", "source_relative", 100., "Paired first-source contrast error (%)", 25.))
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    medians = {}
    for ax, (name, field, scale, title, bound) in zip(axes.ravel(), specs, strict=True):
        for i, arm in enumerate(ARMS):
            values = np.array([float(r[field]) * scale for r in data[name] if r["arm"] == arm])
            assert len(values) == 32
            median = float(np.median(values))
            medians.setdefault(arm, {})[name + "_" + field] = median / scale
            color = "#26866d" if arm == "C" else "#c29849" if arm == "S" else "#557da4" if arm == "Q" else "#9daab9"
            ax.bar(i, median, color=color, width=.68, zorder=2)
            ax.scatter(i + np.linspace(-.23, .23, 32), values, s=10, color="#253449", alpha=.4, linewidths=0, zorder=3)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(range(7), LABELS, fontsize=9)
        ax.axhline(bound, color="#a9484e", lw=1, ls="--")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Predicting a first source write from the False computation", fontsize=16, y=.985)
    fig.text(.5, .94, "Fresh literals; finite normalization/product and every later block. Cubic is primary; quadratic control also passes.", ha="center", fontsize=10.3, color="#475569")
    fig.text(.5, .024, "Bars: medians; dots: 32 directed cases / 16 dependent pairs. Ranked90 and Rnd0–2 add 90 exact source activations to cubic.\nJoint prediction: 29/32; three materiality failures retained, including for the exact reference. Sparse correction: 0/32 passes.", ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.16, hspace=.45, wspace=.24)
    for suffix in ("png", "svg"):
        fig.savefig(figdir / ("source-activation-forecast." + suffix), dpi=160, facecolor="white", metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    svg = figdir / "source-activation-forecast.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    files = ("native.csv", "local.csv", "paired.csv", "effect.csv", "combined.csv", "eligibility.csv", "scores.csv")
    provenance = dict(posthoc=True, model_calls=0, predictions_modified=False, script_sha256=sha(Path(__file__)),
        audit_sha256=sha(DEST / "audit.json"), source_csv_sha256={n: sha(DEST / n) for n in files},
        native_array_sha256=native_arrays, figure_sha256={p.name: sha(p) for p in figdir.iterdir()},
        scope="Visualization/materiality decomposition of frozen endpoints; no new hypothesis or relaxed criterion.")
    (figdir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (figdir / "summary.json").write_text(json.dumps(dict(medians=medians), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    main(args.run.resolve())
