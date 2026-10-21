"""Pre-native projection precision investigation; public weights, saved inputs, no model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import sha
from .checkpoint238 import locate
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main(run, out):
    assert not out.exists() and not (run/"status.json").exists()
    cases = {c["key"]:c for c in read(run/"sources/cases258.json")}
    folder, identity = locate(ROOT)
    assert identity == read(run/"design.json")["checkpoint"]
    matrices, cpu = {}, {}
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in (1, 14, 28):
            matrix = handle.get_tensor(f"model.layers.{layer-1}.mlp.down_proj.weight").float()
            matrices[layer] = matrix.to("mps")
            cpu[layer] = matrix.numpy().astype(float)
    records = []
    with torch.no_grad():
        for entry in read(run/"forecast-status.json")["cases"]:
            if not entry["witness_sha256"]:
                continue
            path = run/"witnesses-forecast"/(entry["key"]+".npz")
            assert sha(path) == entry["witness_sha256"]
            case = cases[entry["key"]]
            with np.load(path) as w:
                for arm in ("F", "Z", "C", "Q", "S", "R"):
                    for layer in (1, 14, 28):
                        prefix = f"{arm}_{layer}_mlp_"
                        product, write = (w[prefix+n].astype(np.float32) for n in ("product", "write"))
                        tensor = torch.tensor(product[None], device="mps")
                        full = torch.zeros((1, len(case["false_ids"]), product.shape[-1]), device="mps", dtype=torch.float32)
                        full[:, case["position"]+1:] = tensor
                        entire = torch.nn.functional.linear(full, matrices[layer])[0, case["position"]+1:].cpu().numpy()
                        sliced = torch.nn.functional.linear(tensor, matrices[layer])[0].cpu().numpy()
                        exact = product.astype(float)@cpu[layer].T
                        records.append(dict(key=case["key"], arm=arm, layer=layer,
                            full_shape_error=float(np.max(abs(entire.astype(float)-write.astype(float)))),
                            sliced_shape_error=float(np.max(abs(sliced.astype(float)-write.astype(float)))),
                            CPU64_error=float(np.max(abs(exact-write.astype(float)))), original_bound=1e-4))
    report = dict(source_sha256=sha(__file__), input_forecast_status_sha256=sha(run/"forecast-status.json"), model_calls=0,
        stage="before_native_measurements", checkpoint=identity, records=records,
        maximum_full_shape_error=max(r["full_shape_error"] for r in records),
        maximum_sliced_shape_error=max(r["sliced_shape_error"] for r in records),
        maximum_CPU64_error=max(r["CPU64_error"] for r in records),
        original_CPU64_failures=sum(r["CPU64_error"] > r["original_bound"] for r in records),
        limitation="Numerical replay of a defined primitive; no independent whole-model reconstruction or derivative recertification.")
    write_json(out, report)
    print(json.dumps({k:v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
