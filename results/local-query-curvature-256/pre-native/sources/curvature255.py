"""Retrospective two-scalar RMSNorm identity and local Taylor diagnostic; no model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def terms(x, d, gamma, eps):
    scale = np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)
    a = np.mean(x*d, axis=-1, keepdims=True)/scale**2
    b = np.mean(d*d, axis=-1, keepdims=True)/scale**2
    slope = gamma/scale*(d-a*x)
    quadratic = gamma/scale*((1.5*a*a-.5*b)*x-a*d)
    cubic = gamma/scale*((1.5*a*b-2.5*a**3)*x+(1.5*a*a-.5*b)*d)
    return slope, quadratic, cubic, a, b


def jet(x, d, gamma, eps):
    """Independent polynomial recurrence A(t) B(t)^2 = 1 through degree 3."""
    coefficients = [np.mean(x*x, axis=-1, keepdims=True)+eps,
                    2*np.mean(x*d, axis=-1, keepdims=True),
                    np.mean(d*d, axis=-1, keepdims=True)]
    inverse = [1/np.sqrt(coefficients[0])]
    for order in range(1, 4):
        known = np.zeros_like(inverse[0])
        for i in range(min(order, 2)+1):
            for j in range(order-i+1):
                k = order-i-j
                if j < len(inverse) and k < len(inverse):
                    known += coefficients[i]*inverse[j]*inverse[k]
        inverse.append(-known/(2*coefficients[0]*inverse[0]))
    return [gamma*(x*inverse[i]+d*inverse[i-1]) for i in range(1, 4)]


def main(run, out):
    assert not out.exists()
    out.mkdir(parents=True)
    audit = read(run/"audit.json")
    assert audit["status"] == "passed"
    weights = dict(np.load(run/"weights.npz"))
    checks, rows = [], []
    cases = read(run/"sources/cases254.json")
    firsts = {}
    for case in cases:
        firsts.setdefault((case["group"]["schema"], case["group"]["reverse"]), case["key"])
    for case, fe, te in zip(cases, read(run/"forecast-status.json")["cases"], read(run/"status.json")["cases"], strict=True):
        key = case["key"]
        assert key == fe["key"] == te["key"]
        fp, tp = (run/sub/(key+".npz") for sub in ("forecasts", "observations"))
        assert sha(fp) == fe["arrays_sha256"] and sha(tp) == te["arrays_sha256"]
        with np.load(fp) as f, np.load(tp) as t:
            for layer in range(1, 29):
                x = f[f"F_r{layer}"].astype(float)
                gamma, eps = weights["input_norm"][layer-1], float(weights["input_eps"])
                def norm(value):
                    return gamma*value/np.sqrt(np.mean(value*value, axis=-1, keepdims=True)+eps)
                native_d = np.zeros_like(x) if layer == 1 else t["T_trajectory"][layer-2, 1:].astype(float)-x
                true_slope = terms(x, native_d, gamma, eps)[0]
                native_remainder = norm(x+native_d)-norm(x)-true_slope
                for arm in ("D0", "P1", "P2", "T"):
                    trajectory = t["T_trajectory"] if arm == "T" else f[arm+"_trajectory"]
                    d = np.zeros_like(x) if layer == 1 else trajectory[layer-2, 1:].astype(float)-x
                    slope, quadratic, cubic, a, b = terms(x, d, gamma, eps)
                    remainder = norm(x+d)-norm(x)-slope
                    ratio = (1+2*a+b)**(-.5)
                    scale = np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)
                    identity = gamma/scale*((ratio-1+a)*x+(ratio-1)*d)
                    checked(checks, "exact_two_scalar_remainder", identity, remainder, 1e-10)
                    for label, actual, expected in zip(("linear", "quadratic", "cubic"), (slope, quadratic, cubic), jet(x, d, gamma, eps), strict=True):
                        checked(checks, "independent_polynomial_jet_"+label, actual, expected, 1e-10)
                    if key in firsts.values() and layer in (1, 14, 28):
                        step = 1e-3
                        fd = (norm(x+step*d)-2*norm(x)+norm(x-step*d))/(2*step**2)
                        checked(checks, "CPU64_RMS_quadratic_centered_FD", fd, quadratic, 1e-5)
                    for scope, pick in (("all", slice(None)), ("verdict", -1)):
                        target = remainder[pick]
                        magnitude = float(np.linalg.norm(target))
                        native_magnitude = float(np.linalg.norm(native_remainder[pick]))
                        row = dict(key=key, **case["group"], layer=layer, seed=arm, scope=scope,
                                   remainder_norm=magnitude, eligible=magnitude >= 1e-8,
                                   max_radial_alignment=float(np.max(abs(a))), max_relative_squared_size=float(np.max(b)))
                        for label, estimate in (("quadratic", quadratic), ("cubic", quadratic+cubic)):
                            error = float(np.linalg.norm(estimate[pick]-target))
                            row[label+"_relative"] = error/max(magnitude, 1e-30)
                            row[label+"_absolute"] = error
                        row["native_remainder_forecast_relative"] = float(np.linalg.norm(target-native_remainder[pick]))/max(native_magnitude, 1e-30)
                        rows.append(row)
        print("two-scalar diagnostic", key[:8], flush=True)
    with (out/"local-errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for arm in ("D0", "P1", "P2", "T"):
        selected = [r for r in rows if r["seed"] == arm and r["scope"] == "all" and r["eligible"]]
        summary[arm] = {k:float(np.median([r[k] for r in selected])) for k in ("quadratic_relative", "cubic_relative", "native_remainder_forecast_relative")}
    write_json(out/"audit.json", dict(status="passed", experiment=255, retrospective=True, model_calls=0, checks=checks,
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/255-local-query-curvature.md"),
        input_audit_sha256=sha(run/"audit.json"), input_forecast_seal_sha256=sha(run/"forecast-seal.json"),
        csv_sha256=sha(out/"local-errors.csv"), summary=summary,
        interpretation="Local algebra and Taylor accuracy on already observed 254 states; not prospective final-response evidence."))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
