"""Posthoc signed-margin/paired-source ledgers and all-case figures for278."""

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
DEST = ROOT / "results/source-mlp-primitives-278"
BITS = ("000", "001", "010", "011", "100", "101", "110", "111")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def table(path, records):
    assert not path.exists()
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def margin(score):
    return float(score[1]) - float(score[0])


def mobius(values):
    return {bits: sum((-1) ** (bits.count("1") - b.count("1")) * values[b] for b in BITS
                     if all(y <= x for x, y in zip(bits, b, strict=True))) for bits in BITS[1:]}


def project(a, b):
    return float(np.vdot(np.ravel(a), np.ravel(b))) / max(float(np.vdot(np.ravel(b), np.ravel(b))), 1e-30)


def main(run):
    assert read(run / "audit.json")["status"] == "passed"
    assert sha(run / "audit.json") == sha(DEST / "audit.json")
    design, status = read(run / "design.json"), read(run / "status.json")
    bank, scores, margin_terms, paired_terms, raw_hashes = {}, [], [], [], {}
    for entry in status["cases"]:
        key = entry["key"]
        p = run / "cases" / (key + ".npz")
        assert sha(p) == entry["arrays_sha256"]
        raw_hashes[str(p.relative_to(run))] = sha(p)
        with np.load(p) as d:
            mf, mt, mz, ml = (margin(d[a + "_scores"]) for a in ("F", "T", "Z", "N000"))
            for arm in design["arms"]:
                m = margin(d[arm + "_scores"])
                scores.append(dict(key=key, arm=arm, false_score=float(d[arm + "_scores"][0]), true_score=float(d[arm + "_scores"][1]),
                    margin=m, response_to_F=m - mf, native_margin=mt, native_response=mt - mf,
                    native_minus_arm=mt - m, arm_minus_Z=m - mz, native_minus_Z=mt - mz,
                    arm_minus_N000=m - ml, native_minus_N000=mt - ml))
            record = dict(input=d["point_N"].astype(float) - d["point_F"].astype(float))
            for prefix in ("N", "C"):
                w = {b: d["node_" + prefix + b + "_write"].astype(float) for b in BITS}
                ww = mobius(w)
                m = {b: margin(d[prefix + b + "_scores"]) for b in BITS}
                mm = mobius(m)
                assert np.max(abs(sum(ww.values()) - (w["111"] - w["000"]))) <= 1e-10
                assert abs(sum(mm.values()) - (m["111"] - m["000"])) <= 1e-10
                for bits, term in mm.items():
                    margin_terms.append(dict(key=key, point=prefix, term=bits, signed_margin=term,
                        target_margin=m["111"] - m["000"], interpretation="Finite-output factorial; includes downstream feedback, not a unique allocation of local write effects."))
                false_w = d["node_F111_write"].astype(float)
                rounding = (d[prefix + "111_source"].astype(float) - d["point_N"].astype(float) - w["111"]
                    - (d["F_source"].astype(float) - d["point_F"].astype(float) - false_w))
                record[prefix] = dict(linear=w["000"] - false_w, terms=ww, rounding=rounding,
                    target=d[prefix + "111_source"].astype(float) - d["F_source"].astype(float))
            bank[key] = record
    for key, own in bank.items():
        other = bank[design["partners"][key]]
        for prefix in ("N", "C"):
            target = own[prefix]["target"] - other[prefix]["target"]
            parts = dict(incoming_mid=own["input"] - other["input"],
                         affine_MLP=own[prefix]["linear"] - other[prefix]["linear"])
            parts.update({b: own[prefix]["terms"][b] - other[prefix]["terms"][b] for b in BITS[1:]})
            parts["rounding"] = own[prefix]["rounding"] - other[prefix]["rounding"]
            error = float(np.max(abs(sum(parts.values()) - target)))
            assert error <= 1e-10
            for term, value in parts.items():
                paired_terms.append(dict(key=key, donor=design["partners"][key], point=prefix, term=term,
                    term_norm=float(np.linalg.norm(value)), target_norm=float(np.linalg.norm(target)),
                    signed_projection=project(value, target), closure_error=error))
    table(DEST / "margins.csv", scores)
    table(DEST / "margin-ledger.csv", margin_terms)
    table(DEST / "paired-source-ledger.csv", paired_terms)
    figdir = ROOT / "docs/figures/source-primitives"
    assert not figdir.exists()
    figdir.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "svg.hashsalt": "source-primitives278"})
    native, local, paired = (rows(DEST / (n + ".csv")) for n in ("native", "local", "paired"))
    ablations = []
    for after, before in (("N100", "N000"), ("N110", "N010"), ("N101", "N001"), ("N111", "N011"), ("N101", "N100"), ("N111", "N101")):
        left = {r["key"]: r for r in native if r["arm"] == after}
        right = {r["key"]: r for r in native if r["arm"] == before}
        for key, row in left.items():
            record = dict(key=key, before=before, after=after)
            for metric in ("all_relative", "verdict_relative", "margin_error"):
                v, u = float(row[metric]), float(right[key][metric])
                record.update({metric + "_before": u, metric + "_after": v, metric + "_improved": v < u})
            ablations.append(record)
    table(DEST / "ablation-comparisons.csv", ablations)
    specifications = ((local, "relative", 100., "First-MLP write-change error (%)", 10.),
                      (native, "all_relative", 100., "Native receiving-vector error (%)", 5.),
                      (native, "margin_error", 1., "Native verdict margin error (nats)", .1))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.1))
    medians = {}
    for i, prefix in enumerate(("N", "C")):
        for j, (data, field, scale, title, bound) in enumerate(specifications):
            ax = axes[i, j]
            for k, bits in enumerate(BITS):
                values = np.array([float(r[field]) * scale for r in data if r["arm"] == prefix + bits])
                assert len(values) == 32
                med = float(np.median(values))
                medians.setdefault(prefix + bits, {})[field] = med / scale
                color = "#2b8b70" if bits == "111" else "#ce9843" if bits == "100" else "#6e8dab"
                ax.bar(k, med, width=.68, color=color, zorder=2)
                ax.scatter(k + np.linspace(-.23, .23, 32), values, s=9, color="#253449", alpha=.4, linewidths=0, zorder=3)
            ax.set_xticks(range(8), BITS, fontsize=9)
            ax.set_title(title, fontsize=11, pad=12)
            ax.axhline(bound, color="#aa454a", ls="--", lw=1)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Native input" if prefix == "N" else "Carrier input", labelpad=12)
    fig.suptitle("First source MLP: finite normalization, SiLU and product", fontsize=17, y=.985)
    fig.text(.5, .94, "One source row is changed; native outer residual and every later block/readout remain finite.", ha="center", fontsize=11, color="#475569")
    fig.text(.5, .025, "Bits: RMSNorm / SiLU / gate–up product. 1 = finite; 0 = False-affine. Primary100 shown in amber; full111 in green.\nBars: medians; dots: all32 old directed cases /16 dependent pairs. Dashed lines are individual bounds; see joint and material-effect tests.", ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.07, right=.985, top=.86, bottom=.15, hspace=.5, wspace=.27)
    for suffix in ("png", "svg"):
        fig.savefig(figdir / ("source-mlp-factorial." + suffix), dpi=160, facecolor="white", metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    # The separate small contrast cannot be inferred from total native accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, scope, title in zip(axes, ("source", "verdict"), ("Paired first-source contrast", "Paired final-verdict contrast"), strict=True):
        for prefix, shift, color in (("N", -.17, "#4579a9"), ("C", .17, "#ba8b4e")):
            for k, bits in enumerate(BITS):
                vals = np.array([100 * float(r[scope + "_relative"]) for r in paired if r["arm"] == prefix + bits])
                ax.bar(k + shift, np.median(vals), width=.30, color=color, label=("Native input" if prefix == "N" else "Carrier input") if k == 0 else None)
                ax.scatter(k + shift + np.linspace(-.11, .11, 32), vals, s=5, color="#263449", alpha=.3, linewidths=0)
        ax.set_xticks(range(8), BITS)
        ax.set_ylabel("Relative error (%)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.axhline(25, color="#aa454a", ls="--", lw=1)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .91), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("A separate target: the difference between prompt backgrounds", fontsize=14, y=.985)
    fig.text(.5, .02, "Bits ordered RMSNorm / SiLU / product. All cases are old and paired; cosine, materiality and actual-margin bounds also apply.", ha="center", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.16, wspace=.23)
    for suffix in ("png", "svg"):
        fig.savefig(figdir / ("source-mlp-paired." + suffix), dpi=160, facecolor="white", metadata={"Date": None} if suffix == "svg" else None)
    plt.close(fig)
    for p in figdir.glob("*.svg"):
        p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    posthoc = dict(posthoc=True, model_calls=0, forecasts_modified=False, script_sha256=sha(Path(__file__)),
        frozen_audit_sha256=sha(run / "audit.json"), raw_array_sha256=raw_hashes,
        csv_sha256={name: sha(DEST / name) for name in ("margins.csv", "margin-ledger.csv", "paired-source-ledger.csv", "ablation-comparisons.csv")},
        interpretation="Supplemental signed ledgers; no new hypothesis endpoint or learned-feature attribution.")
    (DEST / "posthoc-provenance.json").write_text(json.dumps(posthoc, indent=2) + "\n")
    (figdir / "provenance.json").write_text(json.dumps(dict(posthoc=True, model_calls=0,
        script_sha256=sha(Path(__file__)), source_csv_sha256={n: sha(DEST / n) for n in ("native.csv", "local.csv", "paired.csv")},
        figure_sha256={p.name: sha(p) for p in figdir.iterdir()}), indent=2) + "\n")
    (figdir / "summary.json").write_text(json.dumps(dict(medians=medians), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve())
