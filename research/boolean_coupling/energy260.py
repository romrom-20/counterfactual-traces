"""Retrospective SiLU write-energy concentration before and after cancellation."""

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
        inputs.append(dict(key=entry["key"], witness_sha256=sha(wp), forecast_sha256=sha(fp)))
        with np.load(wp) as w, np.load(fp) as f:
            for arm in ("C", "S"):
                for layer in (1, 14, 28):
                    prefix = f"{arm}_{layer}_mlp_"
                    phi, quadratic, up, bound = (w[prefix+n].astype(float) for n in ("finite_phi", "quadratic", "up", "bound"))
                    matrix = matrices[layer]
                    column_energy = np.sum(matrix*matrix, axis=0)
                    score = bound*abs(up)*np.sqrt(column_energy)
                    mask = np.zeros(score.size, dtype=bool)
                    mask[np.argsort(score.ravel(), kind="stable")[-int(.01*score.size):]] = True
                    mask = mask.reshape(score.shape)
                    checked(checks, "original_rank_mask", mask.astype(int), w[prefix+"rank"].astype(int), 0.)
                    coefficients = ((phi-quadratic)*up).astype(np.float32).astype(float)
                    selected = np.where(mask, coefficients, 0.)
                    unselected = np.where(mask, 0., coefficients)
                    complete, restored, remaining = (c@matrix.T for c in (coefficients, selected, unselected))
                    checked(checks, "qualified_complete_component", complete, f[f"{arm}_gap{layer}"], 1e-4)
                    checked(checks, "qualified_selected_component", restored, f[f"{arm}_restore{layer}"], 1e-4)
                    checked(checks, "complete_vector_closure", restored+remaining, complete, 1e-10)
                    individual = coefficients*coefficients*column_energy
                    for scope, pick in (("all", slice(None)), ("verdict", -1)):
                        total = float(np.sum(individual[pick]))
                        retained = float(np.sum(np.where(mask, individual, 0.)[pick]))
                        omitted = float(np.sum(np.where(mask, 0., individual)[pick]))
                        g, v, u = (value[pick] for value in (complete, restored, remaining))
                        gn, vn, un = (float(np.vdot(value, value)) for value in (g, v, u))
                        cross = 2*float(np.vdot(v, u))
                        checked(checks, "summed_vector_energy_identity", np.array(gn), np.array(vn+un+cross), 1e-8)
                        checked(checks, "individual_energy_partition", np.array(total), np.array(retained+omitted), 1e-8)
                        rows.append(dict(key=entry["key"], arm=arm, layer=layer, scope=scope,
                            eligible=gn >= 1e-16 and total >= 1e-20, individual_energy=total,
                            selected_individual_energy=retained, unselected_individual_energy=omitted,
                            selected_fraction=retained/max(total, 1e-30), gap_squared_norm=gn,
                            selected_squared_norm=vn, unselected_squared_norm=un, cross_energy=cross,
                            complete_coherence=gn/max(total, 1e-30),
                            selected_coherence=vn/max(retained, 1e-30), unselected_coherence=un/max(omitted, 1e-30),
                            cross_relative=cross/max(gn, 1e-30), remaining_relative=np.sqrt(un/max(gn, 1e-30)),
                            signed_share=float(np.vdot(v, g))/max(gn, 1e-30)))
    assert len(inputs) == 4 and len(rows) == 48
    with (out/"energy-concentration.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for scope in ("all", "verdict"):
        group = [r for r in rows if r["scope"] == scope]
        eligible = [r for r in group if r["eligible"]]
        summary[scope] = dict(total=len(group), eligible=len(eligible),
            above_90=sum(r["eligible"] and r["selected_fraction"] >= .9 for r in group),
            median_selected_fraction=float(np.median([r["selected_fraction"] for r in eligible])),
            minimum_selected_fraction=min(r["selected_fraction"] for r in eligible),
            negative_cross=sum(r["cross_energy"] < 0 for r in eligible),
            median_coherence=float(np.median([r["complete_coherence"] for r in eligible])),
            median_cross_relative=float(np.median([r["cross_relative"] for r in eligible])),
            median_remaining_relative=float(np.median([r["remaining_relative"] for r in eligible])))
    result = dict(status="passed", experiment=260, retrospective=True, model_calls=0,
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/260-silu-energy-concentration.md"),
        input_audit_sha256=sha(run/"audit.json"), checkpoint=identity, inputs=inputs, checks=checks,
        csv_sha256=sha(out/"energy-concentration.csv"), summary=summary,
        primary_concentration_account=summary["all"]["above_90"] >= 18,
        limitation="Local concentration at existing forecast inputs, not upstream displacement prediction, global causal allocation or a learned circuit.")
    write_json(out/"audit.json", result)
    print(json.dumps({k:v for k, v in result.items() if k not in ("checks", "inputs")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
