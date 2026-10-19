"""Independent NumPy four-primitive/row reconstruction and fixed scoring for 230."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import build, checked, sha, softmax

BITS = tuple(f"{i:04b}" for i in range(16))
NAMES = ("N", "K", "S", "V")


def corner(weights, data, original, bits, rows=None):
    x, y = data["x"], data["receiver"]
    dx = y-x
    dz = weights["attn_norm"]*(dx/original["rms"]-x*np.mean(x*dx, -1, keepdims=True)/original["rms"]**3)
    affine_z = original["z"]+dz
    finite_z = y/np.sqrt(np.mean(y**2, -1, keepdims=True)+weights["attn_eps"])*weights["attn_norm"]
    z = finite_z if bits[0] == "1" else affine_z
    if rows is not None:
        z = affine_z.copy()
        z[rows] = finite_z[rows]
    width, groups = int(weights["width"]), int(weights["groups"])

    def project(name):
        value = (z@weights[name+"_weight"].T+weights.get(name+"_bias", 0)).reshape(len(x), -1, width).transpose(1, 0, 2)
        if name != "v":
            a, b = np.split(value, 2, -1)
            value = value*data["rope_cos"][0]+np.concatenate((-b, a), -1)*data["rope_sin"][0]
        return np.repeat(value, groups, 0) if name != "q" else value

    q, k, v = (project(n) for n in ("q", "k", "v"))
    dl = ((q-original["q"])@original["k"].transpose(0, 2, 1)
          +original["q"]@(k-original["k"]).transpose(0, 2, 1))*weights["scaling"]
    logits = q@k.transpose(0, 2, 1)*weights["scaling"] if bits[1] == "1" else original["logits"]+dl
    dl = logits-original["logits"]
    dp = original["p"]*(dl-np.sum(original["p"]*dl, -1, keepdims=True))
    p = softmax(logits+data["mask"][0]) if bits[2] == "1" else original["p"]+dp
    weighted = p@v if bits[3] == "1" else (original["weighted"]
        +(p-original["p"])@original["v"]+original["p"]@(v-original["v"]))
    flat = weighted.transpose(1, 0, 2).reshape(x.shape)
    return y+flat@weights["o_weight"].T+weights.get("o_bias", 0)


def normalized_paths(weights, data, original, dz):
    width, groups = int(weights["width"]), int(weights["groups"])
    def project(name):
        value = (dz@weights[name+"_weight"].T).reshape(len(dz), -1, width).transpose(1, 0, 2)
        if name != "v":
            a, b = np.split(value, 2, -1)
            value = value*data["rope_cos"][0]+np.concatenate((-b, a), -1)*data["rope_sin"][0]
        return np.repeat(value, groups, 0) if name != "q" else value
    dq, dk, dv = (project(n) for n in ("q", "k", "v"))
    p, v = original["p"], original["v"]
    logits = dict(query=dq@original["k"].transpose(0, 2, 1)*weights["scaling"],
                  key=original["q"]@dk.transpose(0, 2, 1)*weights["scaling"])
    contexts = {n:(p*(dl-np.sum(p*dl, -1, keepdims=True)))@v for n, dl in logits.items()}
    contexts["value"] = p@dv
    return {n:c.transpose(1, 0, 2).reshape(dz.shape)@weights["o_weight"].T for n, c in contexts.items()}


def cosine(a, b):
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.vdot(a, b)/den) if den > 1e-20 else 0.


def main(run):
    assert not (run/"audit.json").exists(), "Never overwrite evidence."
    design, status = (json.loads((run/n).read_text()) for n in ("design.json", "status.json"))
    assert status["status"] == "complete" and status["counts"] == design["expected"]
    previous = Path(design["input_root"])
    root = Path(__file__).resolve().parents[2]
    for name, digest in design["source_hashes"].items():
        assert sha(root/name) == sha(run/"sources"/Path(name).name) == digest
    for name, digest in design["inputs"].items():
        assert sha(previous/name) == digest
    weights = {k:v.astype(float) for k, v in dict(np.load(previous/"weights.npz")).items()}
    checks, cases, edges, positions = [], [], [], []
    derivative_count = 0
    for ent in status["cases"]:
        path = run/"cases"/(ent["key"]+".npz")
        record = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == ent["arrays_sha256"] == record["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == ent["record_sha256"]
        assert all(c["error"] <= c["bound"] for c in record["checks"])
        old = {k:v.astype(float) for k, v in dict(np.load(previous/"cases"/path.name)).items()}
        data = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        start = record["start"]
        original = build(weights, old, old["x"])
        for bits in BITS:
            value = corner(weights, old, original, bits)[start:]
            checked(checks, "numpy_factorial", value, data["smooth_"+bits], 1e-9)
        groups = dict(first=[start], middle=list(range(start+1, len(old["x"])-1)), verdict=[len(old["x"])-1])
        for name, rows in groups.items():
            checked(checks, "numpy_row_graph", corner(weights, old, original, "1000", rows)[start:], data["smooth_N_"+name], 1e-9)
        eta = (old["receiver"]/np.sqrt(np.mean(old["receiver"]**2, -1, keepdims=True)+weights["attn_eps"])*weights["attn_norm"]
            -original["z"]-weights["attn_norm"]*((old["receiver"]-old["x"])/original["rms"]
            -old["x"]*np.mean(old["x"]*(old["receiver"]-old["x"]), -1, keepdims=True)/original["rms"]**3))
        checked(checks, "numpy_norm_curvature", eta[start:], data["norm_curvature"], 1e-9)
        first = np.zeros_like(eta)
        first[start] = eta[start]
        for name, value in normalized_paths(weights, old, original, first).items():
            checked(checks, "numpy_normalized_path", value[start:], data["first_"+name+"_path"], 1e-9)
        for d in record["derivatives"]:
            assert d["ad_error"] <= 1e-9 and d["fd_error"] <= 1e-4
            prefix = d["witness_key"]
            checked(checks, "stored_derivative_FD", data[prefix+"_tangent"],
                    (data[prefix+"_plus"]-data[prefix+"_minus"])/2e-5)
        derivative_count += len(record["derivatives"])
        target = data["old_R"]
        denom = float(np.linalg.norm(target[-1]))
        row = dict(key=ent["key"], **record["group"], R_norm=denom, eligible=denom >= .001,
                   R_all_receiving_norm=float(np.linalg.norm(target)))
        for i, name in enumerate(NAMES):
            only = "0"*i+"1"+"0"*(3-i)
            without = "1"*i+"0"+"1"*(3-i)
            value = data["public_"+only]-data["public_0000"]
            error = value-target
            ratio = float(np.linalg.norm(error[-1])/denom)
            cos = cosine(value[-1], target[-1])
            left = float(np.linalg.norm((data["public_"+without]-data["public_0000"])[-1])/denom)
            row.update({name+"_relative":ratio, name+"_cosine":cos, name+"_without_ratio":left,
                name+"_sufficient":bool(row["eligible"] and ratio <= .25 and cos >= .95),
                name+"_necessary":bool(row["eligible"] and left <= .25),
                name+"_both":bool(row["eligible"] and ratio <= .25 and cos >= .95 and left <= .25)})
        for name in groups:
            value = data["public_N_"+name]-data["public_0000"]
            ratio, cos = float(np.linalg.norm((value-target)[-1])/denom), cosine(value[-1], target[-1])
            row.update({"N_"+name+"_relative":ratio, "N_"+name+"_cosine":cos,
                "N_"+name+"_sufficient":bool(row["eligible"] and ratio <= .25 and cos >= .95)})
        n_all = data["public_1000"]-data["public_0000"]
        n_first = data["public_N_first"]-data["public_0000"]
        n_denom = float(np.linalg.norm(n_all[-1]))
        row["N_first_without_normalization_ratio"] = float(np.linalg.norm((n_all-n_first)[-1])/n_denom) if n_denom else None
        row["N_first_necessary"] = bool(n_denom >= .001 and row["N_first_without_normalization_ratio"] <= .25)
        ordered = dict(N=data["smooth_1111"]-data["smooth_0111"], K=data["smooth_0111"]-data["smooth_0011"],
                       S=data["smooth_0011"]-data["smooth_0001"], V=data["smooth_0001"]-data["smooth_0000"])
        checked(checks, "ordered_closure", sum(ordered.values())+data["rounding_R"], target, 1e-9)
        for name, value in ordered.items():
            row[name+"_ordered_share"] = float(np.vdot(value[-1], target[-1])/denom**2)
            row[name+"_ordered_all_share"] = float(np.vdot(value, target)/np.vdot(target, target))
        for name in ("query", "key", "value"):
            value = data["first_"+name+"_path"]
            row["first_"+name+"_share"] = float(np.vdot(value[-1], target[-1])/denom**2)
        for i, name in enumerate(NAMES):
            for bits in BITS:
                if bits[i] == "0":
                    to = bits[:i]+"1"+bits[i+1:]
                    v = data["public_"+to]-data["public_"+bits]
                    edges.append(dict(key=ent["key"], **record["group"], primitive=name, background=bits,
                        norm=float(np.linalg.norm(v[-1])), signed_share=float(np.vdot(v[-1], target[-1])/denom**2)))
        for j in range(len(target)):
            positions.append(dict(key=ent["key"], **record["group"], offset=j, verdict=j == len(target)-1,
                input_relative=float(np.linalg.norm(data["input_delta"][j])/np.linalg.norm(data["false_input"][j])),
                norm_curvature=float(np.linalg.norm(data["norm_curvature"][j])), R_norm=float(np.linalg.norm(target[j])),
                **{n+"_share":float(np.vdot(v[j], target[j])/np.vdot(target[j], target[j])) for n, v in ordered.items()}))
        cases.append(row)
    assert derivative_count == status["smooth_derivatives"]
    for name, rows in (("cases", cases), ("edges", edges), ("positions", positions)):
        with (run/(name+".csv")).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    decision_keys = [k for k in cases[0] if k.endswith(("_sufficient", "_necessary", "_both"))]
    strata = {f"{s}_{o}":{k:sum(r[k] for r in cases if r["schema"] == s and r["reverse"] == o)
              for k in decision_keys} for s in ("generic", "descriptive") for o in (0, 1)}
    passes = {k:sum(r[k] for r in cases) for k in decision_keys}
    medians = {k:float(np.median([r[k] for r in cases])) for k in cases[0] if k not in
               ("key", "block", "assignment", "target", "schema", "reverse")}
    maximums = {}
    for ent in status["cases"]:
        record = json.loads((run/"cases"/(ent["key"]+".json")).read_text())
        for c in record["checks"]:
            maximums[c["name"]] = max(maximums.get(c["name"], 0), c["error"])
    audit = dict(status="passed", retrospective=True, experiment=230, model_calls=0, cases=len(cases),
        edges=len(edges), positions=len(positions), checks=checks, passes=passes, strata=strata, medians=medians,
        whole={k:passes[k] >= 24 and all(s[k] >= 6 for s in strata.values()) for k in decision_keys},
        runner_maxima=maximums, derivatives=derivative_count, design_sha256=sha(run/"design.json"))
    (run/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:audit[k] for k in ("status", "passes", "whole", "medians", "derivatives")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    main(parser.parse_args().run.resolve())
