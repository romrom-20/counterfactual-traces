"""Explicitly posthoc local-energy versus terminal-Gram comparison for241."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .vector241 import write


def main(run):
    assert not (run / "metric-description.json").exists()
    audit = json.loads((run / "audit.json").read_text())
    assert audit["status"] == "passed"
    with (run / "energy.csv").open() as file:
        energy = list(csv.DictReader(file))
    checks, rows = [], []
    for ent in json.loads((run / "status.json").read_text())["cases"]:
        key = ent["key"]
        meta = json.loads((run / "cases" / (key + ".json")).read_text())
        with np.load(run / "cases" / (key + ".npz")) as d:
            gram = d["gram"]
            den = float(np.vdot(d["raw_target"], d["raw_target"]))
            upper = np.triu(gram, 1)
            diagonal, pairwise = float(np.trace(gram) / den), float(2 * upper.sum() / den)
            checked(checks, "terminal_energy_identity", np.array(diagonal + pairwise), np.array(1.), 1e-8)
            local = {name: sum(float(r[name]) for r in energy if r["key"] == key) / den
                for name in ("transport_energy_change", "remainder_energy", "cross_energy")}
            rows.append(dict(key=key, **meta["group"], **local, terminal_diagonal=diagonal, terminal_pairwise=pairwise,
                terminal_positive_pairwise=float(2 * np.maximum(upper, 0).sum() / den),
                terminal_negative_pairwise=float(2 * np.minimum(upper, 0).sum() / den)))
    with (run / "metric-description.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = ("transport_energy_change", "remainder_energy", "cross_energy", "terminal_diagonal", "terminal_pairwise",
        "terminal_positive_pairwise", "terminal_negative_pairwise")
    result = dict(posthoc=True, causal_claim=False, new_model_calls=0, source_sha256=sha(__file__), checks=checks,
        audit_sha256=sha(run / "audit.json"), means={k: float(np.mean([r[k] for r in rows])) for k in names},
        medians={k: float(np.median([r[k] for r in rows])) for k in names},
        local_cross_negative=sum(r["cross_energy"] < 0 for r in rows), terminal_pairwise_positive=sum(r["terminal_pairwise"] > 0 for r in rows))
    write(run / "metric-description.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
