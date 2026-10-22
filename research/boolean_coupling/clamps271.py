"""Posthoc exact clamp checks and finite state-increment ledger; no model calls."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .audit268 import table
from .fields267 import emit, read


def main(run, out):
    assert not out.exists() and read(run / "audit.json")["status"] == "passed"
    out.mkdir(parents=True)
    design, status = read(run / "design.json"), read(run / "status.json")
    checks, rows, inputs = [], [], {}
    counts = dict(query_equalities=0, correction_node_equalities=0,
                  correction_field_equalities=0, source_equalities=0, prefix_equalities=0,
                  state_increment_ledgers=0)
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        key = case["key"]
        path = run / "cases" / (key + ".npz")
        meta_path = path.with_suffix(".json")
        assert sha(path) == entry["arrays_sha256"]
        assert sha(meta_path) == entry["metadata_sha256"]
        inputs[key] = dict(arrays_sha256=sha(path), metadata_sha256=sha(meta_path))
        meta = read(meta_path)
        records = {(r["arm"], r["layer"]): r for r in meta["records"]}
        with np.load(path) as values:
            for layer in range(1, 29):
                b, c = (records[arm, layer] for arm in ("B", "C"))
                assert b["qinstalled_digest"] == c["qinstalled_digest"]
                counts["query_equalities"] += 1
                for tag in ("p", "d"):
                    assert b["nodes"][tag] == c["nodes"][tag]
                    counts["correction_node_equalities"] += 1
                    counts["correction_field_equalities"] += len(b["nodes"][tag])
                for name in ("source", "prefix"):
                    assert b[name + "_digest"] == c[name + "_digest"]
                    counts[name + "_equalities"] += 1
                i = layer - 1
                qb, qc, xb, xc = (values[arm + "_" + name][i].astype(float)
                                  for arm, name in (("B", "qx"), ("C", "qx"), ("B", "x"), ("C", "x")))
                ob, oc = ((values[arm + "_qx"][i + 1] if layer < 28
                           else values[arm + "_final"][1:]).astype(float) for arm in ("B", "C"))
                carry, attention, mlp, outgoing = qb - qc, (xb - qb) - (xc - qc), (ob - xb) - (oc - xc), ob - oc
                checked(checks, "finite_state_increment_ledger", carry + attention + mlp, outgoing, 1e-12)
                counts["state_increment_ledgers"] += 1
                for scope, pick in (("all", slice(None)), ("verdict", -1)):
                    target = outgoing[pick]
                    denominator = max(float(np.vdot(target, target)), 1e-30)
                    row = dict(key=key, **case["group"], layer=layer, scope=scope,
                               outgoing_norm=float(np.linalg.norm(target)))
                    for name, value in (("carry", carry), ("attention_increment", attention), ("mlp_increment", mlp)):
                        row[name + "_norm"] = float(np.linalg.norm(value[pick]))
                        row[name + "_signed_share"] = float(np.vdot(value[pick], target)) / denominator
                    rows.append(row)
    table(out / "increments.csv", rows)
    emit(out, out / "summary.json", dict(posthoc=True, new_model_calls=0, source_sha256=sha(__file__),
        status_sha256=sha(run / "status.json"), audit_sha256=sha(run / "audit.json"),
        inputs=inputs, checks=checks, counts=counts, csv_sha256=sha(out / "increments.csv"),
        meaning="B and C have identical receiving queries, correction-curve nodes, source and prefix. Other receiving-state paths still differ.",
        limitation="Increment ledger uses differences around actual Float32 residual additions and includes their arithmetic. It is not a Jacobian, pure primitive attribution, transport of local vectors to the final layer, or learned circuit."))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
