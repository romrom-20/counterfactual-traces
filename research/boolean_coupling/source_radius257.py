"""Retrospective native source/receiver RMS geometry and Taylor errors; no model."""

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import write_json
from .curvature255 import terms

ROOT = Path(__file__).resolve().parents[2]
ORDERS = (1, 2, 3, 4, 6, 8, 12)


def read(path):
    return json.loads(path.read_text())


def coefficients(x, d, gamma, eps, checks):
    a = [np.mean(x*x, axis=-1, keepdims=True)+eps,
         2*np.mean(x*d, axis=-1, keepdims=True), np.mean(d*d, axis=-1, keepdims=True)]
    b = [1/np.sqrt(a[0])]
    for n in range(1, 13):
        known = np.zeros_like(b[0])
        for i in range(min(2, n)+1):
            for j in range(n-i+1):
                k = n-i-j
                if j < len(b) and k < len(b):
                    known += a[i]*b[j]*b[k]
        b.append(-known/(2*a[0]*b[0]))
        residual = np.zeros_like(b[0])
        for i in range(min(2, n)+1):
            for j in range(n-i+1):
                residual += a[i]*b[j]*b[n-i-j]
        checked(checks, "inverse_sqrt_polynomial_coefficient_identity", residual, np.zeros_like(residual), 1e-8)
    result = [gamma*(x*b[n]+d*b[n-1]) for n in range(1, 13)]
    for actual, expected in zip(result[:3], terms(x, d, gamma, eps)[:3], strict=True):
        checked(checks, "analytic_first_three_coefficients", actual, expected, 1e-10)
    return result


def main(run, out):
    assert not out.exists()
    out.mkdir(parents=True)
    design = read(run/"design.json")
    assert read(run/"audit.json")["status"] == "passed"
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    source = inspect.getsource(qwen.Qwen2Model.forward)
    assert "inputs_embeds = self.embed_tokens(input_ids)" in source
    folder, identity = locate(ROOT)
    assert identity == design["checkpoint"] and sha(run/"weights.npz") == design["weights_sha256"]
    weights = dict(np.load(run/"weights.npz"))
    eps = float(weights["input_eps"])
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        embedding = handle.get_slice("model.embed_tokens.weight")
        false, true = (embedding[i:i+1].to(torch.float32).numpy().astype(float) for i in (895, 830))
    checks, rows = [], []
    for fe, te in zip(read(run/"forecast-status.json")["cases"], read(run/"status.json")["cases"], strict=True):
        key = fe["key"]
        assert te["key"] == key
        fp, tp = (run/sub/(key+".npz") for sub in ("forecasts", "observations"))
        assert sha(fp) == fe["arrays_sha256"] and sha(tp) == te["arrays_sha256"]
        with np.load(fp) as f, np.load(tp) as t:
            for layer in range(1, 29):
                if layer > 1:
                    checked(checks, "False_receiving_state_boundary", f[f"F_r{layer}"], f["F_trajectory"][layer-2, 1:], 0.)
                for scope in ("source", "receivers"):
                    x = (false if layer == 1 else f["F_trajectory"][layer-2, :1].astype(float)) if scope == "source" else f[f"F_r{layer}"].astype(float)
                    point = (true if layer == 1 else t["T_trajectory"][layer-2, :1].astype(float)) if scope == "source" else (x if layer == 1 else t["T_trajectory"][layer-2, 1:].astype(float))
                    d, gamma = point-x, weights["input_norm"][layer-1]
                    square = np.mean(x*x, axis=-1, keepdims=True)+eps
                    alignment = np.mean(x*d, axis=-1, keepdims=True)/square
                    size = np.mean(d*d, axis=-1, keepdims=True)/square
                    violation = np.maximum(alignment**2-size*(1-eps/square), 0.)
                    checked(checks, "Cauchy_Schwarz_radius_condition", violation, np.zeros_like(violation), 1e-12)
                    def norm(value):
                        return gamma*value/np.sqrt(np.mean(value*value, axis=-1, keepdims=True)+eps)
                    target = norm(point)-norm(x)
                    magnitude, maximum_b = float(np.linalg.norm(target)), float(np.max(size))
                    row = dict(key=key, layer=layer, scope=scope, maximum_b=maximum_b,
                        minimum_series_radius=float(1/np.sqrt(maximum_b)) if maximum_b > 0 else None,
                        update_norm=magnitude, eligible=magnitude >= 1e-8)
                    partial = np.zeros_like(target)
                    for order, coefficient in enumerate(coefficients(x, d, gamma, eps, checks), 1):
                        partial += coefficient
                        if order in ORDERS:
                            row[f"degree{order}_relative"] = float(np.linalg.norm(partial-target))/max(magnitude, 1e-30)
                    rows.append(row)
        print("source/receiver radius", key[:8], flush=True)
    with (out/"source-receiver-errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    initial = [r for r in rows if r["scope"] == "source" and r["layer"] == 1]
    assert len({r["maximum_b"] for r in initial}) == 1
    first = {k:v for k, v in initial[0].items() if k != "key"}
    summary = {}
    for scope in ("source", "receivers"):
        selected = [r for r in rows if r["scope"] == scope and r["eligible"]]
        summary[scope] = dict(maximum_b=max(r["maximum_b"] for r in selected),
            minimum_series_radius=min(r["minimum_series_radius"] for r in selected),
            outside_rows=sum(r["maximum_b"] >= 1 for r in selected),
            median_errors={str(n):float(np.median([r[f"degree{n}_relative"] for r in selected])) for n in ORDERS})
    write_json(out/"audit.json", dict(status="passed", experiment=257, retrospective=True, model_calls=0,
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/257-source-receiver-radius.md"),
        input_audit_sha256=sha(run/"audit.json"), checkpoint=identity,
        csv_sha256=sha(out/"source-receiver-errors.csv"), checks=checks, initial_unique_source=first,
        source_outside_radius_hypothesis=first["maximum_b"] >= 1, summary=summary,
        limitation="One unique initial embedding comparison; native state diagnostics; not a new forecast, circuit or causal explanation of old failed policies."))
    print(json.dumps(dict(initial_source=first, summary=summary), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
