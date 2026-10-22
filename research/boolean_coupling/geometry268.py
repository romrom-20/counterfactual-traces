"""Posthoc input-space geometry of268's omitted directional correction; no model calls."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from .amplitude268 import normalization
from .audit229 import checked, sha
from .fields267 import ROOT, emit, read


def perpendicular(vector, axis):
    value = vector-np.sum(vector*axis, axis=-1, keepdims=True)*axis
    size = np.linalg.norm(value, axis=-1, keepdims=True)
    return value/np.where(size > 1e-12, size, 1.), size


def main(run, reference, out):
    assert not out.exists() and read(run/"audit.json")["status"] == "passed"
    out.mkdir(parents=True)
    design, status = read(run/"design.json"), read(run/"status.json")
    assert design["experiment"] == 268
    assert sha(reference/"norm-weights.npz") == design["reference_hashes"]["norm-weights.npz"]
    checks, rows, ledger = [], [], []
    with np.load(reference/"norm-weights.npz") as weights:
        gammas, eps = weights["mlp_norm"], float(weights["eps"])
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        key = case["key"]
        path = run/"cases"/(key+".npz")
        assert sha(path) == entry["arrays_sha256"]
        with np.load(path) as values:
            for index in range(28):
                args = values["F_x"][index], values["F_z"][index], gammas[index], eps
                star = normalization(args[0], values["L_x"][index], *args[1:])
                unit = star["unit"]
                false_axis, sizes = perpendicular(star["false_pattern"], unit)
                random = []
                for row in range(len(unit)):
                    seed = hashlib.sha256(f"{key}:{index+1}:{row}:geometry268".encode()).hexdigest()
                    draw = np.random.default_rng(int(seed[:16], 16)).normal(size=unit.shape[-1])
                    random.append(draw)
                random_axis, _ = perpendicular(np.array(random), unit)
                for arm in ("A11", "D11", "P11"):
                    parts = normalization(args[0], values[arm+"_x"][index], *args[1:])
                    angular = parts["direction"]-parts["amplitude"]*unit
                    line = np.sum(angular*unit, axis=-1, keepdims=True)*unit
                    second = np.sum(angular*false_axis, axis=-1, keepdims=True)*false_axis
                    random_second = np.sum(angular*random_axis, axis=-1, keepdims=True)*random_axis
                    remainder = angular-line-second
                    checked(checks, "orthogonal_input_geometry_closure", line+second+remainder, angular, 1e-10)
                    checked(checks, "orthogonal_plane_residual", np.sum(remainder*false_axis, axis=-1), np.zeros(len(unit)), 1e-9)
                    for row in range(len(unit)):
                        total = float(np.sum(angular[row]**2))
                        def fraction(value):
                            return float(np.sum(value[row]**2))/max(total, 1e-30)
                        rows.append(dict(key=key, **case["group"], arm=arm, layer=index+1, row=row,
                            verdict=row == len(unit)-1, eligible=total >= 1e-16, angular_energy=total,
                            line_fraction=fraction(line), false_plane_fraction=fraction(line+second),
                            random_plane_fraction=fraction(line+random_second),
                            false_plane_error=float(np.linalg.norm(remainder[row]))/max(np.sqrt(total), 1e-30),
                            second_axis_norm=float(sizes[row, 0])))
            full = values["A11_final"].astype(float)-values["P11_final"].astype(float)-values["D11_final"].astype(float)+values["L_final"].astype(float)
            amp = values["A10_final"].astype(float)-values["P10_final"].astype(float)-values["D10_final"].astype(float)+values["L_final"].astype(float)
            a = values["A10_final"].astype(float)-values["A11_final"].astype(float)
            d = values["D10_final"].astype(float)-values["D11_final"].astype(float)
            checked(checks, "final_interaction_error_identity", a-d, amp-full, 1e-10)
            for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                ledger.append(dict(key=key, **case["group"], scope=scope,
                    A_change_norm=float(np.linalg.norm(a[pick])), D_change_norm=float(np.linalg.norm(d[pick])),
                    error_norm=float(np.linalg.norm((a-d)[pick])), target_norm=float(np.linalg.norm(full[pick])),
                    A_D_cosine=float(np.vdot(a[pick], d[pick]))/max(float(np.linalg.norm(a[pick])*np.linalg.norm(d[pick])), 1e-30)))
    for name, data in (("geometry.csv", rows), ("final-ledger.csv", ledger)):
        with (out/name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summaries = []
    for arm in ("A11", "D11", "P11"):
        for schema in ("generic", "descriptive"):
            for scope in ("all", "verdict"):
                selected = [r for r in rows if r["arm"] == arm and r["schema"] == schema
                            and (scope == "all" or r["verdict"]) and r["eligible"]]
                total = sum(r["angular_energy"] for r in selected)
                summaries.append(dict(arm=arm, schema=schema, scope=scope, eligible=len(selected),
                    **{f"weighted_{mode}": sum(r["angular_energy"]*r[mode+"_fraction"] for r in selected)/max(total, 1e-30)
                       for mode in ("line", "false_plane", "random_plane")},
                    median_false_plane_error=float(np.median([r["false_plane_error"] for r in selected]))))
    emit(out, out/"summary.json", dict(status="complete", experiment="268-geometry", posthoc=True,
        model_calls=0, source_sha256=sha(__file__), helper_sha256=sha(ROOT/"research/boolean_coupling/amplitude268.py"),
        input_hashes={n: sha(run/n) for n in ("design.json", "status.json", "audit.json")},
        norm_weights_sha256=sha(reference/"norm-weights.npz"), checks=checks, summaries=summaries,
        csv_sha256={n: sha(out/n) for n in ("geometry.csv", "final-ledger.csv")},
        limitation="Retrospective input geometry and final-vector accounting, no causal plane intervention or learned-feature claim."))
    for row in summaries:
        print(row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.reference.resolve(), args.out.resolve())
