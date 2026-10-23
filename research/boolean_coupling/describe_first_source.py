"""Posthoc275 first-block signed geometry; saved arrays only, zero model calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/source-attention-reading-275"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(value, target):
    size, norm = float(np.linalg.norm(value)), float(np.linalg.norm(target))
    return dict(target_norm=norm, candidate_norm=size, norm_ratio=size / max(norm, 1e-30),
                relative=float(np.linalg.norm(value - target)) / max(norm, 1e-30),
                cosine=float(np.vdot(value, target)) / max(size * norm, 1e-30),
                signed_projection=float(np.vdot(value, target)) / max(norm ** 2, 1e-30))


def main(run):
    assert json.loads((run / "audit.json").read_text())["status"] == "passed"
    design, status = (json.loads((run / n).read_text()) for n in ("design.json", "status.json"))
    assert design["experiment"] == 275
    entries = {r["key"]: r for r in status["cases"]}
    collected, rows, pairs = {}, [], []
    for case in design["cases"]:
        key = case["key"]
        path = run / "cases" / (key + ".npz")
        assert sha(path) == entries[key]["arrays_sha256"]
        with np.load(path) as data:
            h = {a: data[a + "_states"][0, 0].astype(float) - data["F_states"][0, 0].astype(float) for a in ("N", "W", "V", "B")}
            h0 = data["N_before"][0].astype(float) - data["F_before"][0].astype(float)
            attention = data["N_attwrite"][0].astype(float) - data["F_attwrite"][0].astype(float)
            mlp = data["N_mlpwrite"][0].astype(float) - data["F_mlpwrite"][0].astype(float)
            for arm in h:
                rows.append(dict(key=key, **case["group"], arm=arm, embedding_norm=float(np.linalg.norm(h0)),
                    native_attention_write_norm=float(np.linalg.norm(attention)), native_MLP_write_norm=float(np.linalg.norm(mlp)),
                    **compare(h[arm], h["N"])))
            collected[key] = h
    for case in design["cases"]:
        key = case["key"]
        donor = next(c for c in design["cases"] if c["group"]["schema"] != case["group"]["schema"] and
            {k: v for k, v in c["group"].items() if k != "schema"} == {k: v for k, v in case["group"].items() if k != "schema"})
        target = collected[key]["N"] - collected[donor["key"]]["N"]
        for arm in ("N", "W", "V", "B"):
            pairs.append(dict(key=key, donor=donor["key"], **case["group"], arm=arm,
                **compare(collected[key][arm] - collected[donor["key"]][arm], target)))
    for name, records in (("first-source.csv", rows), ("first-source-pairs.csv", pairs)):
        path = DEST / name
        assert not path.exists()
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    path = DEST / "first-source-provenance.json"
    assert not path.exists()
    path.write_text(json.dumps(dict(posthoc=True, model_calls=0, source_sha256=sha(Path(__file__)),
        input_status_sha256=sha(run / "status.json"), input_audit_sha256=sha(run / "audit.json"),
        arrays_sha256={key: value["arrays_sha256"] for key, value in entries.items()},
        scope="Descriptive first-block native source geometry and paired contrast. Not a preregistered first-block causal rescue, fresh forecast or learned feature."), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
