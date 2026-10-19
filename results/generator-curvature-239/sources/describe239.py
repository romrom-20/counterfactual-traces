"""Posthoc exact SiLU-tail identity and position/concentration description for239."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import checked, sha
from .audit235 import finite
from .checkpoint238 import locate, mlp_derivative, weights


def main(run):
    assert not (run/"tail-description.json").exists()
    audit = json.loads((run/"audit.json").read_text())
    assert audit["status"] == "passed"
    root = Path(__file__).resolve().parents[2]
    design = json.loads((run/"design.json").read_text())
    folder, checkpoint = locate(root)
    assert checkpoint == design["checkpoint"]
    config = json.loads((folder/"config.json").read_text())
    previous, old = Path(design["input238"]), Path(design["input228"])
    selected = json.loads((previous/"selection.json").read_text())["selected"]
    checks, rows, raw = [], [], []
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for ent in json.loads((run/"status.json").read_text())["cases"]:
            key = ent["key"]
            d = dict(np.load(run/"cases"/(key+".npz")))
            f = dict(np.load(old/"forecasts"/(key+".npz")))
            meta = json.loads((run/"cases"/(key+".json")).read_text())
            for n in selected:
                w = weights(handle, config, n//2, "mlp")
                x = f["false_mids"][n//2].astype(float)
                z0 = x/np.sqrt(np.mean(x*x, -1, keepdims=True)+w["eps"])*w["norm"]
                g0, u0 = (z0@w[name+"_weight"].T+w.get(name+"_bias", 0) for name in ("gate", "up"))
                sigma = 1/(1+np.exp(-g0))
                phi, slope, curve = g0*sigma, sigma*(1+g0*(1-sigma)), sigma*(1-sigma)*(2+g0*(1-2*sigma))
                tails, crosses = {}, np.zeros_like(g0, dtype=bool)
                for name in ("native", "own"):
                    y = x+d[f"n{n}_{name}_direction"]
                    z = y/np.sqrt(np.mean(y*y, -1, keepdims=True)+w["eps"])*w["norm"]
                    dg, du = ((z-z0)@w[p+"_weight"].T for p in ("gate", "up"))
                    g = g0+dg
                    exact = g/(1+np.exp(-g))
                    tails[name] = (exact-phi-slope*dg-.5*curve*dg*dg)*(u0+du)
                    crosses |= g*g0 <= 0
                tail = (tails["native"]-tails["own"])@w["down_weight"].T
                pred = sum(d[f"n{n}_native_finite_{c}"]-d[f"n{n}_own_finite_{c}"] for c in ("norm", "silu", "product"))-d[f"n{n}_cast"]
                actual = d[f"n{n}_target_native"]-d[f"n{n}_target_own"]
                exact_contrast = (finite(w, {}, x+d[f"n{n}_native_direction"], "mlp")
                    -finite(w, {}, x+d[f"n{n}_own_direction"], "mlp")
                    -mlp_derivative(w, x, d[f"n{n}_native_direction"]-d[f"n{n}_own_direction"])-d[f"n{n}_cast"])
                checked(checks, "exact_missing_SiLU_tail_identity", exact_contrast-pred, tail, 1e-9)
                public_error = float(np.max(abs(actual-pred-tail)))
                raw.append(dict(key=key, node=n, error=public_error, bound=1e-4, passed=public_error <= 1e-4))
                energy = np.sum((actual-pred)**2, axis=-1)
                readout = tail/np.linalg.norm(tail)
                contribution = (tails["native"]-tails["own"])*(readout@w["down_weight"])
                flat = contribution.ravel()
                order = np.argsort(-abs(flat))
                count = max(1, int(np.ceil(.01*len(flat))))
                rows.append(dict(key=key, **meta["group"], node=n, layer=n//2+1,
                    first_row_squared_error_share=float(energy[0]/sum(energy)),
                    verdict_squared_error_share=float(energy[-1]/sum(energy)),
                    top1pct_absolute_projection_share=float(sum(abs(flat[order[:count]]))/sum(abs(flat))),
                    top1pct_signed_projection_share=float(sum(flat[order[:count]])/sum(flat)),
                    zero_crossing_signed_projection_share=float(sum(contribution[crosses])/sum(flat)),
                    target_dependent_readout=True, posthoc=True))
    with (run/"tail-description.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(posthoc=True, new_model_calls=0, causal_interventions=False, source_sha256=sha(__file__), checks=checks,
        public_bridge_checks=raw, medians={str(n):{name:float(np.median([r[name] for r in rows if r["node"] == n])) for name in
        ("first_row_squared_error_share", "verdict_squared_error_share", "top1pct_absolute_projection_share",
         "top1pct_signed_projection_share", "zero_crossing_signed_projection_share")} for n in selected})
    (run/"tail-description.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k, v in result.items() if k not in ("checks", "public_bridge_checks")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
