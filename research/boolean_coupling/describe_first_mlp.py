"""Posthoc all-case figures for276/277; zero model calls or prediction changes."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = {276: "first-source-mlp-gain-276", 277: "first-source-mlp-forecast-277"}
ARMS = ("C", "A", "L", "S", "G0", "G1", "G2", "Z")
LABELS = ("Carrier\nonly", "Additive", "False\nslope", "Shifted\nslope", "G0", "G1", "G2", "No MLP\nchange")
COLORS = ("#94a3b8", "#d5a04c", "#4677b3", "#15946b", "#a8b2bf", "#a8b2bf", "#a8b2bf", "#be716e")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest = ROOT / "docs/figures/first-mlp"
    assert not dest.exists()
    dest.mkdir(parents=True)
    source_hashes, summary = {}, {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "first-source-mlp-276277"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.1))
    for i, (number, name) in enumerate(FOLDERS.items()):
        folder = ROOT / "results" / name
        audit = read(folder / "audit.json")
        assert audit["status"] == "passed"
        native, paired, gain, local = (rows(folder / (f + ".csv")) for f in ("native", "paired", "gain", "local-gain"))
        components = []
        for arm in ("S", "T" if number == 277 else "U"):
            gg, ll = ([r for r in data if r["arm"] == arm] for data in (gain, local))
            components.append(dict(arm=arm, cases=len(gg), local_passes=sum(r["passes"] == "True" for r in ll),
                downstream_vectors=sum(all(float(r[s + "_norm"]) >= .001 and float(r[s + "_relative"]) <= .25 and float(r[s + "_cosine"]) >= .95 for s in ("all", "verdict")) for r in gg),
                material_margin=sum(abs(float(r["target_margin"])) >= .02 for r in gg),
                within_margin_bound=sum(float(r["margin_error"]) <= float(r["margin_bound"]) for r in gg),
                full_effect_passes=sum(r["passes"] == "True" for r in gg)))
        summary[number] = dict(audit=audit, gain_components=components, medians={})
        fields = ((paired, "source_relative", 100., "Paired first-source contrast error (%)", 25.),
                  (native, "all_relative", 100., "Native receiving-vector error (%)", 5.),
                  (native, "margin_error", 1., "Native verdict margin error (nats)", .1))
        for j, (records, field, scale, title, bound) in enumerate(fields):
            ax = axes[i, j]
            data = [np.array([float(r[field]) * scale for r in records if r["arm"] == arm]) for arm in ARMS]
            assert all(len(d) == 32 for d in data)
            med = [float(np.median(d)) for d in data]
            ax.bar(np.arange(len(ARMS)), med, color=COLORS, width=.68, zorder=2)
            for k, d in enumerate(data):
                jitter = np.linspace(-.23, .23, len(d))
                ax.scatter(k + jitter, d, s=9, color="#253449", alpha=.4, zorder=3, linewidths=0)
            ax.axhline(bound, color="#a8474b", lw=1, linestyle="--", zorder=1)
            ax.set_xticks(np.arange(len(ARMS)), LABELS, fontsize=8)
            ax.set_title(title, fontsize=11, pad=12)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.14)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(("Old contexts" if number == 276 else "Sealed fresh literals") + "\n32 directed cases / 16 pairs", labelpad=12)
            for arm, m in zip(ARMS, med, strict=True):
                summary[number]["medians"].setdefault(arm, {})[field] = m / scale
        for name in ("native.csv", "paired.csv", "gain.csv", "local-gain.csv", "explanation.csv", "audit.json"):
            p = folder / name
            source_hashes[str(p.relative_to(ROOT))] = sha(p)
    fig.suptitle("A shifted MLP sensitivity predicts the small context contrast", fontsize=17, x=.5, y=.985)
    fig.text(.5, .94, "One source site is approximated; every later block and final readout remains finite.", ha="center", fontsize=11, color="#475569")
    fig.text(.5, .022, "Bars: medians; dots: all cases (dependent pairs). G0–G2: norm-matched random controls. Dashed lines: individual error bounds.\nThe joint prediction and stronger material correction-effect explanation have separate decisions; see all controls and failures in the report.", ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.085, right=.985, top=.865, bottom=.16, hspace=.55, wspace=.25)
    for suffix in ("png", "svg"):
        fig.savefig(dest / ("first-source-gain." + suffix), dpi=160, facecolor="white", metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    svg = dest / "first-source-gain.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    provenance = dict(posthoc=True, model_calls=0, predictions_modified=False, source_csv_sha256=source_hashes,
        script_sha256=sha(Path(__file__)), figure_sha256={p.name: sha(p) for p in dest.iterdir()},
        scope="All-case visualization of frozen endpoints; selected common arms, every omitted control retained in CSVs.")
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
