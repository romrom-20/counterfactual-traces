"""Summarize audited225b outcomes and draw standalone scientific figures."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"runs/baseline-interaction-225b")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    data = json.loads((args.out/"audit.json").read_text())
    qualified = json.loads((args.out/"qualification-audit.json").read_text())
    assert data["qualified"] and qualified["qualified"]
    figures = args.figures or args.out/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, aggregate = data["rows"], []
    keys = ("E00", "E10", "E01", "E11")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, site in zip(axes, (22, 23)):
        for schema, color, marker in (("generic", ".45", "o"), ("descriptive", "#1d4f77", "s")):
            group = [r for r in rows if r["site"] == site and r["group"]["schema"] == schema]
            medians = {key: float(np.median([r[key] for r in group])) for key in keys}
            aggregate.append(dict(site=site, schema=schema, count=len(group), medians=medians,
                                  signs={key: dict(positive=sum(r[key]>0 for r in group), negative=sum(r[key]<0 for r in group)) for key in keys},
                                  K_only_reverses=sum(r["K_only_reverses"] for r in group),
                                  B_only_reverses=sum(r["B_only_reverses"] for r in group)))
            ax.plot(range(4), [medians[k] for k in keys], color=color, marker=marker, label=schema+" median")
            for index, key in enumerate(keys):
                ax.scatter(index+np.linspace(-.06, .06, len(group)), [r[key] for r in group], color=color, marker=marker, s=13, alpha=.4)
        ax.axhline(0, color=".5", lw=.8)
        ax.set(xticks=range(4), xticklabels=("Original\nE00", "Baseline only\nE10", "Interaction only\nE01", "Both\nE11"),
               ylabel="Finite-SiLU − affine-SiLU margin effect (nats)",
               xlabel="Separated product-coordinate intervention", title=f"MLP{site}: baseline B versus interaction K")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("225b: does the interaction or downstream baseline produce the sign reversal?", fontsize=13)
    fig.supxlabel("32 old contexts · finite native RMSNorm · actual down projection and full finite suffix/readout\nB and K are defined from the four actual product cells; random controls match product norms. Retrospective, not a False-only forecast.", fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(figures/f"225b-baseline-versus-interaction.{extension}", dpi=160)
    plt.close(fig)
    behavior = []
    for path in sorted((args.out/"qualification").glob("*.npz")):
        record = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as q:
            false = float(np.float64(q["false_scores"][1])-np.float64(q["false_scores"][0]))
            true = float(np.float64(q["native_scores"][1])-np.float64(q["native_scores"][0]))
        behavior.append(dict(context=record["key"], group=record["group"], false_margin=false,
                             true_margin=true, shift=true-false, same_boolean_choice=bool((false>0)==(true>0))))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for schema, color, marker in (("generic", ".45", "o"), ("descriptive", "#1d4f77", "s")):
        r = [r for r in behavior if r["group"]["schema"] == schema]
        ax.scatter([v["false_margin"] for v in r], [v["true_margin"] for v in r], color=color, marker=marker, label=schema)
    values = [r[k] for r in behavior for k in ("false_margin", "true_margin")]
    ax.plot([min(values)-1, max(values)+1], [min(values)-1, max(values)+1], color=".6", ls="--", label="unchanged margin")
    ax.axhline(0, color=".75", lw=.8);ax.axvline(0, color=".75", lw=.8)
    ax.set(xlabel="Verdict margin after metadata false (nats)", ylabel="Verdict margin after metadata true (nats)",
           title="Native conditional verdict preferences, before internal intervention")
    ax.legend(fontsize=8);ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Pinned Qwen2.5-1.5B-Instruct ·32 old contexts · supplied Boolean prefix · actual public logits\nA two-token preference comparison; no free generation or verifier-competence claim.", fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(figures/f"225b-native-boolean-coupling.{extension}", dpi=160)
    plt.close(fig)
    result = dict(aggregate=aggregate, behavior=behavior, gates=data["gates"],
                  same_boolean_choice=sum(r["same_boolean_choice"] for r in behavior),
                  source_audit_sha256=hashlib.sha256((args.out/"audit.json").read_bytes()).hexdigest(),
                  scope="Descriptive summaries of the audited retrospective experiment; no added fitted rule or fresh confirmation.")
    (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "behavior"}, indent=2))


if __name__ == "__main__":
    main()
