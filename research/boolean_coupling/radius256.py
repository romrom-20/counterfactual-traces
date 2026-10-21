"""Posthoc local RMS Taylor radii on saved 256 states; no model or new forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .core225b import write_json


def read(path):
    return json.loads(path.read_text())


def main(run):
    assert not (run/"local-radii.json").exists() and read(run/"audit.json")["status"] == "passed"
    weights = dict(np.load(run/"weights.npz"))
    eps = float(weights["input_eps"])
    assert eps > 0
    rows, checks = [], []
    for fe, te in zip(read(run/"forecast-status.json")["cases"], read(run/"status.json")["cases"], strict=True):
        key = fe["key"]
        assert te["key"] == key
        fp, tp = (run/sub/(key+".npz") for sub in ("forecasts", "observations"))
        assert sha(fp) == fe["arrays_sha256"] and sha(tp) == te["arrays_sha256"]
        with np.load(fp) as f, np.load(tp) as t:
            for arm in ("L", "Q", "C", "R", "T"):
                trajectory = t["T_trajectory"] if arm == "T" else f[arm+"_trajectory"]
                largest_b, largest_a = 0., 0.
                for layer in range(2, 29):
                    x = f[f"F_r{layer}"].astype(float)
                    d = trajectory[layer-2, 1:].astype(float)-x
                    square = np.mean(x*x, axis=-1)+eps
                    a = np.mean(x*d, axis=-1)/square
                    b = np.mean(d*d, axis=-1)/square
                    violation = np.maximum(a*a-b*(1-eps/square), 0.)
                    checked(checks, "Cauchy_Schwarz_for_local_radius", violation, np.zeros_like(violation), 1e-12)
                    assert np.isfinite(a).all() and np.isfinite(b).all()
                    largest_b = max(largest_b, float(np.max(b)))
                    largest_a = max(largest_a, float(np.max(abs(a))))
                assert largest_b > 0
                rows.append(dict(key=key, arm=arm, maximum_b=largest_b, maximum_abs_a=largest_a,
                                 minimum_series_radius=float(1/np.sqrt(largest_b)), all_rows_inside_unit_edit=largest_b < 1))
    summary = {arm:dict(maximum_b=max(r["maximum_b"] for r in rows if r["arm"] == arm),
        minimum_series_radius=min(r["minimum_series_radius"] for r in rows if r["arm"] == arm),
        cases_inside=sum(r["all_rows_inside_unit_edit"] for r in rows if r["arm"] == arm)) for arm in ("L", "Q", "C", "R", "T")}
    write_json(run/"local-radii.json", dict(posthoc=True, model_calls=0, source_sha256=sha(__file__),
        input_audit_sha256=sha(run/"audit.json"), checks=checks, cases=rows, summary=summary,
        formula="For b>0 and eps>0, a^2<b; roots of 1+2*a*t+b*t^2 have modulus 1/sqrt(b).",
        limitation="Local smooth RMS Taylor convergence only; no truncation-error or whole-network stability theorem, learned feature or new forecast."))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve())
