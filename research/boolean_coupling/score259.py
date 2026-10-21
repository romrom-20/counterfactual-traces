"""Retrospective fixed-input ranking factors; no new model or behavioral forecast."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main(run, out):
    assert not out.exists() and read(run/"audit.json")["status"] == "passed"
    out.mkdir(parents=True)
    folder, identity = locate(ROOT)
    assert identity == read(run/"design.json")["checkpoint"]
    matrices = {}
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in (1, 14, 28):
            matrices[layer] = handle.get_tensor(f"model.layers.{layer-1}.mlp.down_proj.weight").float().numpy().astype(float)
    rows, checks, inputs = [], [], []
    for entry in read(run/"forecast-status.json")["cases"]:
        if not entry["witness_sha256"]:
            continue
        wp = run/"witnesses-forecast"/(entry["key"]+".npz")
        fp = run/"forecasts"/(entry["key"]+".npz")
        assert sha(wp) == entry["witness_sha256"] and sha(fp) == entry["arrays_sha256"]
        with np.load(wp) as w, np.load(fp) as f:
            for arm in ("C", "S"):
                for layer in (1, 14, 28):
                    prefix = f"{arm}_{layer}_mlp_"
                    base, gate, phi, quadratic, up, bound = (w[prefix+n].astype(float) for n in ("base", "gate", "finite_phi", "quadratic", "up", "bound"))
                    matrix = matrices[layer]
                    columns = np.sqrt(np.sum(matrix*matrix, axis=0))
                    scores = dict(BUW=bound*abs(up)*columns, BU=bound*abs(up), BW=bound*columns,
                        UW=abs(up)*columns, B=bound, U=abs(up), W=np.broadcast_to(columns, up.shape),
                        DUW=abs(gate-base)**3*abs(up)*columns, D=abs(gate-base)**3)
                    tail = phi-quadratic
                    target = f[f"{arm}_gap{layer}"].astype(float)
                    complete = (tail*up).astype(np.float32).astype(float)@matrix.T
                    checked(checks, "qualified_complete_component_gap", complete, target, 1e-4)
                    reference = w[prefix+"rank"].astype(bool)
                    for name, score in scores.items():
                        mask = np.zeros(score.size, dtype=bool)
                        mask[np.argsort(score.ravel(), kind="stable")[-int(.01*score.size):]] = True
                        mask = mask.reshape(score.shape)
                        if name == "BUW":
                            checked(checks, "original_rank_mask", mask.astype(int), reference.astype(int), 0.)
                        value = (np.where(mask, tail, 0.)*up).astype(np.float32).astype(float)@matrix.T
                        if name == "BUW":
                            checked(checks, "original_ranked_component", value, f[f"{arm}_restore{layer}"], 1e-4)
                        for scope, pick in (("all", slice(None)), ("verdict", -1)):
                            v, g = value[pick], target[pick]
                            norm = float(np.linalg.norm(g))
                            rows.append(dict(key=entry["key"], arm=arm, layer=layer, score=name, scope=scope,
                                eligible=norm >= 1e-8, gap_norm=norm, remaining_relative=float(np.linalg.norm(v-g))/max(norm, 1e-30),
                                signed_share=float(np.vdot(v, g)/max(np.vdot(g, g), 1e-30)),
                                restored_norm=float(np.linalg.norm(v)), mask_overlap=float(np.sum(mask & reference)/np.sum(reference))))
                    inputs.append(dict(key=entry["key"], arm=arm, layer=layer))
    assert len(inputs) == 24 and len(rows) == 432
    with (out/"score-factors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for arm in ("C", "S"):
        for scope in ("all", "verdict"):
            selected = [r for r in rows if r["arm"] == arm and r["scope"] == scope and r["eligible"]]
            summary[arm+"_"+scope] = {name:dict(median_remaining=float(np.median([r["remaining_relative"] for r in selected if r["score"] == name])),
                median_overlap=float(np.median([r["mask_overlap"] for r in selected if r["score"] == name]))) for name in scores}
    local = [r for r in rows if r["scope"] == "all"]
    dropped_w = 0
    for item in inputs:
        pair = {r["score"]:r for r in local if all(r[k] == v for k, v in item.items())}
        dropped_w += bool(pair["BU"]["eligible"] and pair["BUW"]["eligible"] and pair["BU"]["remaining_relative"]-pair["BUW"]["remaining_relative"] <= .05)
    all_buw = [r["remaining_relative"] for r in local if r["score"] == "BUW" and r["eligible"]]
    all_uw = [r["remaining_relative"] for r in local if r["score"] == "UW" and r["eligible"]]
    result = dict(status="passed", experiment=259, retrospective=True, model_calls=0, checks=checks, fixed_inputs=inputs,
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/259-silu-score-factors.md"),
        input_audit_sha256=sha(run/"audit.json"), input_forecast_seal_sha256=sha(run/"forecast-seal.json"),
        csv_sha256=sha(out/"score-factors.csv"), summary=summary,
        dropped_bound_median_ratio=float(np.median(all_uw)/np.median(all_buw)),
        dropped_bound_meets_2x=bool(np.median(all_uw) >= 2*np.median(all_buw)),
        dropped_column_norm_within_05=dropped_w, dropped_column_norm_meets_18=bool(dropped_w >= 18),
        limitation="24 existing forecast-input witnesses, not a new native forecast, simplified-rule confirmation or learned-feature account.")
    write_json(out/"audit.json", result)
    print(json.dumps({k:v for k, v in result.items() if k not in ("checks", "fixed_inputs")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
