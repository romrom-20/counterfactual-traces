"""Posthoc split of transported individual-vector energy and pairwise interaction."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .vector241 import NODES, inputs, write


def main(run):
    assert not (run / "transport-metric.json").exists()
    _, previous, _, _ = inputs(run)
    assert json.loads((run / "audit.json").read_text())["status"] == "passed"
    with (run / "energy.csv").open() as file:
        energy = list(csv.DictReader(file))
    rows, cases, checks = [], [], []
    for ent in json.loads((run / "status.json").read_text())["cases"]:
        key = ent["key"]
        meta = json.loads((run / "cases" / (key + ".json")).read_text())
        with np.load(run / "cases" / (key + ".npz")) as d, np.load(previous / "ledger" / (key + ".npz")) as ledger:
            den = float(np.vdot(d["raw_target"], d["raw_target"]))
            before = np.empty((0, *d["target"].shape))
            diagonal_sum, pair_sum, injected_sum, cross_sum = 0., 0., 0., 0.
            for n in NODES:
                after = d[f"n{n}_tangent"][:-1]
                diagonal = float(np.sum(after * after) - np.sum(before * before))
                total = float(np.vdot(after.sum(0), after.sum(0)) - np.vdot(before.sum(0), before.sum(0)))
                pairwise = total - diagonal
                original = next(r for r in energy if r["key"] == key and int(r["node"]) == n)
                checked(checks, "transport_metric_local_identity", np.array(total), np.array(float(original["transport_energy_change"])), 1e-8)
                rows.append(dict(key=key, **meta["group"], node=n, layer=n // 2 + 1, kind="attention" if n % 2 == 0 else "mlp",
                    diagonal_change=diagonal / den, pairwise_change=pairwise / den, total_change=total / den))
                diagonal_sum += diagonal
                pair_sum += pairwise
                remainder = ledger[f"n{n}_remainder"]
                injected_sum += float(np.vdot(remainder, remainder))
                cross_sum += float(original["cross_energy"])
                before = np.concatenate((after, remainder[None]))
            gram = d["gram"]
            checked(checks, "diagonal_transport_global_identity", np.array(diagonal_sum), np.array(np.trace(gram) - injected_sum), 1e-8)
            checked(checks, "pairwise_transport_global_identity", np.array(pair_sum), np.array(gram.sum() - np.trace(gram) - cross_sum), 1e-8)
            cases.append(dict(key=key, diagonal_change=diagonal_sum / den, pairwise_change=pair_sum / den,
                total_change=(diagonal_sum + pair_sum) / den))
    with (run / "transport-metric.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = ("diagonal_change", "pairwise_change", "total_change")
    result = dict(posthoc=True, new_model_calls=0, causal_claim=False, source_sha256=sha(__file__), checks=checks,
        audit_sha256=sha(run / "audit.json"), means={k: float(np.mean([r[k] for r in cases])) for k in names},
        medians={k: float(np.median([r[k] for r in cases])) for k in names},
        counts=dict(diagonal_negative=sum(r["diagonal_change"] < 0 for r in cases), pairwise_positive=sum(r["pairwise_change"] > 0 for r in cases)),
        depth=[dict(node=n, layer=n // 2 + 1, kind="attention" if n % 2 == 0 else "mlp",
            **{k: float(np.mean([r[k] for r in rows if r["node"] == n])) for k in names}) for n in NODES])
    write(run / "transport-metric.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "depth")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
