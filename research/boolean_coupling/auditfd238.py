"""Additional independent NumPy finite-endpoint qualification for every238 direction."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import build, checked, sha
from .audit235 import finite
from .checkpoint238 import locate, weights


def main(run, phase):
    target = run/("numpy-fd-"+phase+".json")
    assert not target.exists()
    root = Path(__file__).resolve().parents[2]
    design = json.loads((run/"design.json").read_text())
    folder, provenance = locate(root)
    assert provenance == design["checkpoint"]
    config = json.loads((folder/"config.json").read_text())
    old = Path(design["input228"])
    state = json.loads((run/("ledger-status.json" if phase == "ledger" else "status.json")).read_text())
    checks, count = [], 0
    subfolder = "ledger" if phase == "ledger" else "cases"
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for entry in state["cases"]:
            key = entry["key"]
            path = run/subfolder/(key+".npz")
            meta = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == entry["arrays_sha256"] == meta["arrays_sha256"]
            assert sha(path.with_suffix(".json")) == entry["record_sha256"]
            data = dict(np.load(path))
            ledger = data if phase == "ledger" else dict(np.load(run/"ledger"/(key+".npz")))
            f = dict(np.load(old/"forecasts"/(key+".npz")))
            start = meta["start"]
            context = {k:ledger[k].astype(float) for k in ("mask", "rope_cos", "rope_sin")}
            for n in range(2, 39):
                i, kind = n//2, "attention" if n % 2 == 0 else "mlp"
                w = weights(handle, config, i, kind)
                labels = ([f"n{n}", *([f"n{n}_dual"] if f"n{n}_dual_direction" in data else [])]
                    if phase == "ledger" else [f"{name}_n{n}" for name in ("oracle", "own4", "random4")])
                for label in labels:
                    direction = data[label+"_direction"]
                    if kind == "attention":
                        x = ledger[f"a{i}_false"].astype(float)
                        d = np.zeros_like(x)
                        d[start:] = direction
                        plus_x, minus_x = x+1e-5*d, x-1e-5*d
                        plus = (plus_x+build(w, context, plus_x)["write"])[start:]
                        minus = (minus_x+build(w, context, minus_x)["write"])[start:]
                    else:
                        x = f["false_mids"][i].astype(float)
                        plus, minus = (finite(w, {}, x+sign*1e-5*direction, "mlp") for sign in (1, -1))
                    fd = (plus-minus)/2e-5
                    checked(checks, "independent_numpy_FD_tangent", fd, data[label+"_tangent"])
                    checked(checks, "independent_numpy_FD_witness", fd, data[label+"_fd"])
                    count += 1
            print("independent finite endpoints", phase, key[:8], flush=True)
    assert count == state["counts"]["jvps"]
    result = dict(status="passed", experiment=238, phase=phase, qualification_only=True, model_calls=0,
        source_sha256=sha(__file__), protocol_sha256=sha(root/"docs/experiments/238d-independent-fd.md"),
        original_design_sha256=sha(run/"design.json"), checks=checks,
        independent_FD_comparisons=count, independent_finite_endpoints=2*count)
    target.write_text(json.dumps(result, indent=2)+"\n")
    print("NumPy finite endpoints passed", phase, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("ledger", "intervene"), required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.phase)
    except Exception as exc:
        (args.run/("failure-numpy-fd-"+args.phase+".json")).write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
