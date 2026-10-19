"""Posthoc energy-profile arithmetic audit and descriptive transport summaries."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .vector241 import NODES, inputs, write


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def main(run):
    assert not (run / "profile-summary.json").exists()
    assert json.loads((run / "audit.json").read_text())["status"] == "passed"
    _, previous, _, _ = inputs(run)
    energy, transported = (read_csv(run / (name + ".csv")) for name in ("energy", "transported"))
    checks, rows, profiles = [], [], []
    for ent in json.loads((run / "status.json").read_text())["cases"]:
        key = ent["key"]
        with np.load(previous / "ledger" / (key + ".npz")) as ledger:
            last = ledger["n30_tangent"] + ledger["n30_remainder"]
            denominator = float(np.vdot(last, last))
            totals = dict(transport_energy_change=0., remainder_energy=0., cross_energy=0., net_energy_change=0.)
            for n in NODES:
                e, j, r = (ledger[f"n{n}_" + k] for k in ("direction", "tangent", "remainder"))
                values = dict(transport_energy_change=float(np.vdot(j, j) - np.vdot(e, e)),
                    remainder_energy=float(np.vdot(r, r)), cross_energy=float(2 * np.vdot(j, r)),
                    net_energy_change=float(np.vdot(j + r, j + r) - np.vdot(e, e)))
                record = next(row for row in energy if row["key"] == key and int(row["node"]) == n)
                for name, value in values.items():
                    checked(checks, "independent_energy_" + name, np.array(value), np.array(float(record[name])), 1e-8)
                    totals[name] += value
                rows.append(dict(key=key, node=n, **{k: v / denominator for k, v in values.items()}))
        with np.load(run / "cases" / (key + ".npz")) as d:
            terms = d["terms"]
            profiles.append(dict(key=key, **{k: v / denominator for k, v in totals.items()},
                norm_sum_to_sum_norm=float(sum(np.linalg.norm(t) for t in terms) / np.linalg.norm(terms.sum(0))),
                cast_norm=float(np.linalg.norm(d["cast"]))))
    names = ("transport_energy_change", "remainder_energy", "cross_energy", "net_energy_change")
    result = dict(status="passed", descriptive_only=True, new_model_calls=0, new_causal_interventions=False,
        source_sha256=sha(__file__), audit_sha256=sha(run / "audit.json"), checks=checks,
        aggregate_medians={k: float(np.median([r[k] for r in profiles])) for k in (*names, "norm_sum_to_sum_norm", "cast_norm")},
        aggregate_means={k: float(np.mean([r[k] for r in profiles])) for k in names},
        depth=[dict(node=n, layer=n // 2 + 1, kind="attention" if n % 2 == 0 else "mlp",
            **{k: float(np.median([r[k] for r in rows if r["node"] == n])) for k in names},
            median_local_norm=float(np.median([float(r["local_norm"]) for r in transported if int(r["node"]) == n])),
            median_terminal_norm=float(np.median([float(r["terminal_norm"]) for r in transported if int(r["node"]) == n])),
            median_norm_gain=float(np.median([float(r["norm_gain"]) for r in transported if int(r["node"]) == n]))) for n in NODES])
    write(run / "profile-summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "depth")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
