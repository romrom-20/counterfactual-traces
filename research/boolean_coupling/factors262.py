"""Retrospective normalized gate/up curvature: old pattern versus new direction."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import write_json
from .silu258 import predict

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def measure(value, target):
    magnitude = float(np.linalg.norm(target))
    return float(np.linalg.norm(value-target))/max(magnitude, 1e-30), float(np.vdot(value, target))/max(float(np.linalg.norm(value))*magnitude, 1e-30)


def main(run, qualified, out):
    assert not out.exists() and read(qualified/"audit.json")["status"] == "passed"
    out.mkdir(parents=True)
    design, status, audit = (read(p) for p in (run/"design.json", run/"status.json", qualified/"audit.json"))
    folder, checkpoint = locate(ROOT)
    assert checkpoint == audit["checkpoint"]
    with np.load(run/"norm-weights.npz") as w:
        weights = {name:w[name].astype(float) for name in w.files}
    inputs = []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        path = run/"cases"/(entry["key"]+".npz")
        vp = qualified/("vectors-"+entry["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(vp) == audit["vectors_sha256"][vp.name]
        inputs.append((case, np.load(path), np.load(vp)))
    rows, scalars, checks = [], [], []
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            matrices = {branch:handle.get_tensor(f"model.layers.{layer-1}.mlp.{branch}_proj.weight").float().numpy().astype(float) for branch in ("gate", "up")}
            gamma, eps = weights["mlp_norm"][layer-1], float(weights["eps"])
            for case, values, vectors in inputs:
                key, p0 = case["key"], f"F_{layer}_"
                x0 = values[p0+"x"].astype(float)
                scale = np.sqrt(np.mean(x0*x0, axis=-1, keepdims=True)+eps)
                base_patterns = {branch:(gamma*x0/scale)@matrix.T for branch, matrix in matrices.items()}
                for arm in ("B", "T"):
                    prefix = f"{arm}_{layer}_"
                    d = values[prefix+"x"].astype(float)-x0
                    a = np.mean(x0*d, axis=-1, keepdims=True)/scale**2
                    b = np.mean(d*d, axis=-1, keepdims=True)/scale**2
                    rho = (1+2*a+b)**(-.5)
                    actual_norm = gamma*(x0+d)/np.sqrt(np.mean((x0+d)**2, axis=-1, keepdims=True)+eps)
                    checked(checks, "exact_scalar_normalization_identity", gamma/scale*(x0+d)*rho, actual_norm, 1e-10)
                    for row in range(len(a)):
                        scalars.append(dict(key=key, arm=arm, layer=layer, row=row, a=float(a[row, 0]), b=float(b[row, 0]), rho=float(rho[row, 0]),
                            radial_alignment=abs(float(a[row, 0]))/max(np.sqrt(float(b[row, 0])), 1e-30)))
                    mask = predict(values[p0+"gate"].astype(float), values[p0+"phi"].astype(float), values[prefix+"gate"].astype(float),
                        values[prefix+"up"].astype(float), weights["columns"][layer-1], key, layer)[1]
                    if arm == "B":
                        checked(checks, "original_input_conditional_mask", mask.astype(int), values[prefix+"mask"].astype(int), 0.)
                    for branch in matrices:
                        h0 = base_patterns[branch]
                        linear, curvature, public, arithmetic = (vectors[f"{arm}_{layer}_{branch}_{name}"] for name in ("linear", "curvature", "public", "arithmetic"))
                        direction = linear+a*h0
                        base = (rho-1+a)*h0
                        directional = (rho-1)*direction
                        energy = -.5*b*h0
                        checked(checks, "exact_curvature_factor_identity", base+directional, curvature, 1e-9)
                        for name, candidate in (("base", base), ("direction", directional), ("energy", energy)):
                            for scope, pick in (("all", slice(None)), ("selected", mask), ("verdict", -1)):
                                c, target = candidate[pick], curvature[pick]
                                relative, cosine = measure(c, target)
                                norm = float(np.linalg.norm(target))
                                pnorm, anorm = float(np.linalg.norm(public[pick])), float(np.linalg.norm(arithmetic[pick]))
                                remainder = float(np.linalg.norm(target-c))
                                lower = max(0., remainder-anorm)/max(pnorm, 1e-30)
                                upper = (remainder+anorm)/max(pnorm, 1e-30)
                                actual_error = float(np.linalg.norm((linear+candidate-public)[pick]))/max(pnorm, 1e-30)
                                checked(checks, "public_triangle_lower", np.array(max(0., lower-actual_error)), np.array(0.), 1e-9)
                                checked(checks, "public_triangle_upper", np.array(max(0., actual_error-upper)), np.array(0.), 1e-9)
                                rows.append(dict(key=key, arm=arm, layer=layer, branch=branch, term=name, scope=scope,
                                    eligible=norm >= 1e-8, curvature_norm=norm, relative_error=relative, cosine=cosine,
                                    signed_share=float(np.vdot(c, target))/max(norm*norm, 1e-30), candidate_norm=float(np.linalg.norm(c)),
                                    passes=norm >= 1e-8 and relative <= .25 and cosine >= .95,
                                    public_norm=pnorm, public_error_lower=lower, public_error_upper=upper,
                                    robust_public_pass=pnorm >= 1e-8 and upper <= .05,
                                    robust_public_reject=pnorm >= 1e-8 and lower > .05,
                                    public_ambiguous=pnorm >= 1e-8 and lower <= .05 < upper))
    for _, values, vectors in inputs:
        values.close()
        vectors.close()
    for name, data in (("factors.csv", rows), ("scalars.csv", scalars)):
        with (out/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = {}
    for arm in ("B", "T"):
        for scope in ("all", "selected"):
            for term in ("base", "direction", "energy"):
                group = [r for r in rows if r["arm"] == arm and r["scope"] == scope and r["branch"] == "gate" and r["term"] == term and r["eligible"]]
                summary[arm+"_"+scope+"_"+term] = dict(eligible=len(group), passes=sum(r["passes"] for r in group),
                    meets_90=bool(group and sum(r["passes"] for r in group) >= .9*len(group)),
                    median_relative=float(np.median([r["relative_error"] for r in group])),
                    median_signed_share=float(np.median([r["signed_share"] for r in group])),
                    robust_public_passes=sum(r["robust_public_pass"] for r in group),
                    robust_public_rejects=sum(r["robust_public_reject"] for r in group), public_ambiguous=sum(r["public_ambiguous"] for r in group))
    write_json(out/"audit.json", dict(status="passed", experiment=262, retrospective=True, model_calls=0,
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/262-gate-curvature-factors.md"),
        input_qualified_audit_sha256=sha(qualified/"audit.json"), input_status_sha256=sha(run/"status.json"),
        checkpoint=checkpoint, checks=checks, summary=summary,
        csv_sha256={name:sha(out/name) for name in ("factors.csv", "scalars.csv")},
        limitation="Conditional local factorization at old native/forecast inputs, not an upstream state prediction, primitive intervention or learned feature."))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.qualified.resolve(), args.out.resolve())
