"""Independent Torch primitive allocations, frozen local contrast scoring for239."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from .audit229 import checked, sha
from .checkpoint238 import locate, weights

MODELS = ("quadratic", "finite")
COMPONENTS = ("norm", "silu", "product")


def independent(w, x, d):
    p = {k:torch.from_numpy(np.asarray(v)) for k, v in w.items()}
    t, v = torch.from_numpy(x), torch.from_numpy(d)

    def linear(z, name, bias=False):
        return F.linear(z, p[name+"_weight"], p.get(name+"_bias") if bias else None)

    def norm(z):
        return p["norm"]*z*torch.rsqrt(z.square().mean(-1, keepdim=True)+p["eps"])

    z, z1 = torch.func.jvp(norm, (t,), (v,))
    def first(z):
        return torch.func.jvp(norm, (z,), (v,))[1]
    _, z2 = torch.func.jvp(first, (t,), (v,))
    z2 = z2/2
    g, u = linear(z, "gate", True), linear(z, "up", True)
    ones = torch.ones_like(g)
    phi, slope = torch.func.jvp(F.silu, (g,), (ones,))
    def gate_first(a):
        return torch.func.jvp(F.silu, (a,), (ones,))[1]
    _, curve = torch.func.jvp(gate_first, (g,), (ones,))
    g1, u1 = linear(z1, "gate"), linear(z1, "up")
    g2, u2 = linear(z2, "gate"), linear(z2, "up")
    dz = norm(t+v)-z
    dg, du = linear(dz, "gate"), linear(dz, "up")
    fields = dict(quadratic_norm=slope*g2*u+phi*u2,
        quadratic_silu=.5*curve*g1*g1*u, quadratic_product=slope*g1*u1,
        finite_norm=slope*(dg-g1)*u+phi*(du-u1),
        finite_silu=.5*curve*dg*dg*u,
        finite_product=(slope*dg+.5*curve*dg*dg)*du)
    return {name:linear(value, "down").detach().numpy() for name, value in fields.items()}


def geometry(a, b):
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a-b)/den) if den else 0., float(np.vdot(a, b)/(np.linalg.norm(a)*den)) if np.linalg.norm(a)*den else 0.


def main(run):
    assert not (run/"audit.json").exists()
    root = Path(__file__).resolve().parents[2]
    design, status = (json.loads((run/name).read_text()) for name in ("design.json", "status.json"))
    assert status["status"] == "complete" and status["ordinary_models"] == 0
    for name, h in design["source_hashes"].items():
        assert sha(root/name) == sha(run/"sources"/Path(name).name) == h
    previous = Path(design["input238"])
    assert sha(previous/"audit.json") == design["audit238_sha256"]
    assert sha(previous/"selection.json") == design["selection_sha256"]
    old = Path(design["input228"])
    folder, checkpoint = locate(root)
    assert checkpoint == design["checkpoint"]
    config = json.loads((folder/"config.json").read_text())
    selected = json.loads((previous/"selection.json").read_text())["selected"]
    checks, rows, decision_rows, component_rows, raw = [], [], [], [], []
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for entry in status["cases"]:
            key = entry["key"]
            path = run/"cases"/(key+".npz")
            meta = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == meta["arrays_sha256"] == entry["arrays_sha256"]
            assert sha(path.with_suffix(".json")) == entry["record_sha256"]
            assert sha(previous/"ledger"/path.name) == meta["ledger238_sha256"]
            assert sha(previous/"cases"/path.name) == meta["intervention238_sha256"]
            assert all(c["error"] <= c["bound"] for c in meta["checks"])
            raw += meta["raw_float64_bridges"]
            data = dict(np.load(path))
            f = dict(np.load(old/"forecasts"/path.name))
            for n in selected:
                w = weights(handle, config, n//2, "mlp")
                x = f["false_mids"][n//2].astype(float)
                for name in ("native", "source", "own"):
                    parts = independent(w, x, data[f"n{n}_{name}_direction"])
                    for component, value in parts.items():
                        checked(checks, "independent_primitive_allocation", value, data[f"n{n}_{name}_{component}"], 1e-9)
                    quadratic = sum(parts[k] for k in ("quadratic_norm", "quadratic_silu", "quadratic_product"))
                    checked(checks, "independent_Hhalf", quadratic, data[f"n{n}_{name}_Hhalf"], 1e-9)
                    checked(checks, "independent_second_FD", quadratic, data[f"n{n}_{name}_secondFD"])
                target = data[f"n{n}_target_native"]-data[f"n{n}_target_own"]
                for model in MODELS:
                    native = sum(data[f"n{n}_native_{model}_{c}"]-data[f"n{n}_source_{model}_{c}"] for c in COMPONENTS)
                    own = sum(data[f"n{n}_own_{model}_{c}"]-data[f"n{n}_source_{model}_{c}"] for c in COMPONENTS)+data[f"n{n}_cast"]
                    predicted = native-own
                    parts = {c:data[f"n{n}_native_{model}_{c}"]-data[f"n{n}_own_{model}_{c}"] for c in COMPONENTS}
                    checked(checks, "contrast_component_closure", sum(parts.values())-data[f"n{n}_cast"], predicted, 1e-9)
                    passes = []
                    for scope in ("all", "verdict"):
                        pick = slice(None) if scope == "all" else -1
                        r, cos = geometry(predicted[pick], target[pick])
                        eligible = np.linalg.norm(target[pick]) >= .001
                        passes.append(bool(eligible and r <= .25 and cos >= .95))
                        nr, nc = geometry(native[pick], data[f"n{n}_target_native"][pick])
                        pr, pc = geometry(own[pick], data[f"n{n}_target_own"][pick])
                        rows.append(dict(key=key, **meta["group"], node=n, layer=n//2+1, model=model, scope=scope,
                            target_norm=float(np.linalg.norm(target[pick])), relative_error=r, cosine=cos, eligible=eligible,
                            native_relative=nr, native_cosine=nc, own_relative=pr, own_cosine=pc,
                            cast_norm=float(np.linalg.norm(data[f"n{n}_cast"][pick]))))
                        den = float(np.vdot(target[pick], target[pick]))
                        for name, v in parts.items():
                            component_rows.append(dict(key=key, **meta["group"], node=n, layer=n//2+1, model=model,
                                scope=scope, component=name, norm=float(np.linalg.norm(v[pick])),
                                signed_share=float(np.vdot(v[pick], target[pick])/den) if den else 0.))
                    decision_rows.append(dict(key=key, **meta["group"], node=n, model=model, passes=all(passes)))
            print("curvature audit", key[:8], flush=True)
    for name, records in (("contrasts", rows), ("decisions", decision_rows), ("components", component_rows)):
        with (run/(name+".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    decisions = {}
    for model in MODELS:
        decisions[model] = {}
        for n in selected:
            subset = [r for r in decision_rows if r["node"] == n and r["model"] == model]
            count = sum(r["passes"] for r in subset)
            strata = {f"{s}_{rev}":sum(r["passes"] for r in subset if r["schema"] == s and r["reverse"] == rev)
                for s in ("generic", "descriptive") for rev in (0, 1)}
            decisions[model][str(n)] = dict(passes=count, strata=strata, meets_rule=count >= 24 and all(c >= 6 for c in strata.values()))
        decisions[model]["all_four_passes"] = sum(all(r["passes"] for r in decision_rows if r["key"] == ent["key"] and r["model"] == model)
            for ent in status["cases"])
    summary = dict(status="passed" if all(r["passed"] for r in raw) else "passed_declared_domain_with_raw_bridge_failures",
        experiment=239, retrospective=True, ordinary_models=0, source_hashes=design["source_hashes"], checks=checks,
        raw_float64_bridges=raw, decisions=decisions,
        medians={model:{str(n):{scope:{name:float(np.median([r[name] for r in rows if r["model"] == model and r["node"] == n
            and r["scope"] == scope])) for name in ("relative_error", "cosine", "native_relative", "own_relative")}
            for scope in ("all", "verdict")} for n in selected} for model in MODELS})
    (run/"audit.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k, v in summary.items() if k not in ("checks", "source_hashes", "raw_float64_bridges")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    target = parser.parse_args().run.resolve()
    try:
        main(target)
    except Exception as exc:
        (target/"failure-audit.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
