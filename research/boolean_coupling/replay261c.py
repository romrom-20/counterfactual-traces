"""Replay every261b gate/up primal at full shape; retain original CPU64 failures."""

import argparse
import csv
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
    assert not out.exists()
    out.mkdir(parents=True)
    design, status = read(run/"design.json"), read(run/"status.json")
    folder, checkpoint = locate(ROOT)
    cases = []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        path = run/"cases"/(entry["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"]
        cases.append((case, np.load(path)))
    records = []
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            for branch in ("gate", "up"):
                matrix32 = handle.get_tensor(f"model.layers.{layer-1}.mlp.{branch}_proj.weight").float().numpy()
                matrix = matrix32.astype(float)
                mps = torch.tensor(matrix32, device="mps")
                for case, values in cases:
                    for arm in ("F", "Z", "B", "L", "C", "R", "T"):
                        prefix = f"{arm}_{layer}_"
                        if prefix+"write" not in values:
                            continue
                        point = values[prefix+"installed_z"].astype(float)
                        target = values[prefix+branch].astype(float)
                        raw = float(np.max(abs(point@matrix.T-target)))
                        with torch.no_grad():
                            full = torch.zeros((1, len(case["false_ids"]), point.shape[-1]), dtype=torch.float32, device="mps")
                            full[:, case["position"]+1:] = torch.tensor(point.astype(np.float32)[None], device="mps")
                            output = torch.nn.functional.linear(full, mps)[0, case["position"]+1:].cpu().numpy().astype(float)
                        error = float(np.max(abs(output-target)))
                        records.append(dict(key=case["key"], arm=arm, layer=layer, branch=branch,
                            CPU64_error=raw, original_bound=1e-4, original_pass=raw <= 1e-4,
                            full_shape_error=error, exact_replay=error == 0., maximum_output=float(np.max(abs(target)))))
            print("gate/up full-shape replay", layer, "/28", flush=True)
    for _, values in cases:
        values.close()
    with (out/"projection-replays.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(out/"investigation.json", dict(status="complete", source_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/"docs/experiments/261c-projection-replay.md"),
        input_design_sha256=sha(run/"design.json"), input_status_sha256=sha(run/"status.json"), checkpoint=checkpoint,
        comparisons=len(records), CPU64_failures=sum(not r["original_pass"] for r in records),
        maximum_CPU64_error=max(r["CPU64_error"] for r in records), exact_replays=sum(r["exact_replay"] for r in records),
        maximum_full_shape_error=max(r["full_shape_error"] for r in records), csv_sha256=sha(out/"projection-replays.csv"),
        limitation="Numerical primitive replay only, not derivative or behavior qualification; CPU64 failures remain failures."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
