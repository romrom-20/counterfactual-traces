"""Frozen diagnostic of native/own local generator differences using False curvature."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha
from .audit235 import finite
from .checkpoint238 import locate, mlp_derivative, weights
from .curvature239 import anchor, formula, torch_function

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/p for p in ("research/boolean_coupling/run239.py", "research/boolean_coupling/audit239.py",
    "research/boolean_coupling/curvature239.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit235.py",
    "research/boolean_coupling/settings.json", "docs/experiments/239-curvature-generator-contrast.md")]
PROGRESS = {}


def main(previous, out):
    assert not out.exists()
    assert json.loads((previous/"audit.json").read_text())["status"] == "passed"
    assert json.loads((previous/"numpy-fd-intervene.json").read_text())["status"] == "passed"
    design238 = json.loads((previous/"design.json").read_text())
    old = Path(design238["input228"])
    selected = json.loads((previous/"selection.json").read_text())["selected"]
    assert sorted(selected) == [31, 33, 35, 37]
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design238["checkpoint"]
    config = json.loads((folder/"config.json").read_text())
    out.mkdir(parents=True)
    for name in ("cases", "sources"):
        (out/name).mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    (out/"design.json").write_text(json.dumps(dict(experiment=239, retrospective=True, input238=str(previous),
        input228=str(old), checkpoint=checkpoint, source_hashes=sources,
        audit238_sha256=sha(previous/"audit.json"), selection_sha256=sha(previous/"selection.json")), indent=2)+"\n")
    entries, count = [], 0
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for entry in json.loads((previous/"status.json").read_text())["cases"]:
            key = entry["key"]
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            native = dict(np.load(previous/"ledger"/(key+".npz")))
            own = dict(np.load(previous/"cases"/(key+".npz")))
            meta = json.loads((previous/"cases"/(key+".json")).read_text())
            checks, raw, data = [], [], {}
            PROGRESS.update(key=key, checks=checks, raw_float64_bridges=raw, data=data)
            for n in selected:
                i = n//2
                w = weights(handle, config, i, "mlp")
                x = f["false_mids"][i].astype(float)
                a = anchor(w, x)
                function = torch_function(w)
                source = x+f["source_mid"][i]
                inputs = dict(native=o["true_mids"][i].astype(float), source=source,
                    own=own[f"own4_n{n}_finite_input"].astype(float))
                public = dict(false=f["false_boundaries"][i+1], source=f["false_boundaries"][i+1].astype(float)+f["source_out"][i],
                    native=o["true_states"][i], own=own[f"own4_n{n}_finite_output"])
                for name, value in dict(false=x, **inputs).items():
                    with np.errstate(over="ignore"):
                        single = finite({k:v.astype(np.float32) for k, v in w.items()}, {}, value.astype(np.float32), "mlp")
                    checked(checks, "public_float32_primal", single, public[name])
                    error = float(np.max(abs(finite(w, {}, value, "mlp")-public[name])))
                    raw.append(dict(node=n, input=name, error=error, bound=1e-4, passed=error <= 1e-4))
                for name, value in inputs.items():
                    d = value-x
                    parts = formula(w, x, d, a)
                    for component, v in parts.items():
                        data[f"n{n}_{name}_{component}"] = v
                    data[f"n{n}_{name}_direction"] = d
                    quadratic = sum(parts[k] for k in ("quadratic_norm", "quadratic_silu", "quadratic_product"))
                    t, direction = torch.from_numpy(x), torch.from_numpy(d)
                    def first(v, fn=function, vector=direction):
                        return torch.func.jvp(fn, (v,), (vector,))[1]
                    _, second = torch.func.jvp(first, (t,), (direction,))
                    checked(checks, "quadratic_AD", quadratic, second.detach().numpy()/2, 1e-9)
                    step = 3e-4
                    base = function(t).detach().numpy()
                    plus, minus = (function(t+sign*step*direction).detach().numpy() for sign in (1, -1))
                    fd = (plus-2*base+minus)/(2*step**2)
                    checked(checks, "quadratic_FD", quadratic, fd)
                    data[f"n{n}_{name}_Hhalf"] = second.detach().numpy()/2
                    data[f"n{n}_{name}_secondFD"] = fd
                    data[f"n{n}_{name}_fd_hashes"] = np.array([hashlib.sha256(v.tobytes()).hexdigest() for v in (plus, minus)])
                    count += 1
                data[f"n{n}_cast"] = mlp_derivative(w, x, inputs["own"]-own[f"own4_n{n}_input"])
                data[f"n{n}_target_native"] = native[f"n{n}_remainder"]
                data[f"n{n}_target_own"] = own[f"own4_n{n}_finite_output"].astype(float)-own[f"own4_n{n}_linear"]
                del function, first
            path = out/"cases"/(key+".npz")
            np.savez_compressed(path, **data)
            record = dict(key=key, group=meta["group"], arrays_sha256=sha(path), checks=checks, raw_float64_bridges=raw,
                ledger238_sha256=sha(previous/"ledger"/(key+".npz")), intervention238_sha256=sha(previous/"cases"/(key+".npz")))
            path.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
            entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
            print("curvature generator", len(entries), "/32", flush=True)
    assert count == 384
    (out/"status.json").write_text(json.dumps(dict(status="complete", cases=entries, ordinary_models=0,
        hessian_AD_comparisons=count, finite_second_difference_endpoints=2*count), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-238", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_238.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/"partial.npz", **PROGRESS["data"])
                (args.out/"partial.json").write_text(json.dumps({k:v for k, v in PROGRESS.items() if k != "data"}, indent=2)+"\n")
            (args.out/"failure.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
