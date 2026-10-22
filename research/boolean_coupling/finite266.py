"""Retrospective finite-SiLU control of same-input normalization interactions."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha
from .audit261b import metrics
from .checkpoint238 import locate, read_head
from .local265 import ROOT, anchor, emit, finite_scores, read, table
from .transmission265 import corners, project

SOURCES = ("research/boolean_coupling/finite266.py", "docs/experiments/266-finite-silu-interaction-control.md",
    "research/boolean_coupling/transmission265.py", "research/boolean_coupling/local265.py",
    "research/boolean_coupling/input263.py", "research/boolean_coupling/audit261b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/silu258.py")
ACTIVE = None


def main(reference, local, out):
    global ACTIVE
    assert not out.exists()
    out.mkdir(parents=True)
    ACTIVE = out
    (out/"sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/Path(name).name)
    design, status, la = (read(p) for p in (reference/"design.json", reference/"status.json", local/"audit.json"))
    assert la["status"] == "passed" and la["counts"]["exact_old_full_shape_MLP_replays"] == 224
    folder, checkpoint = locate(ROOT)
    assert checkpoint == read(local/"design.json")["checkpoint"]
    head = read_head(folder)
    with np.load(reference/"norm-weights.npz") as w:
        columns, eps = w["columns"], float(w["eps"])
    local_records = {r["key"]:r for r in la["records"]}
    inputs, saved = [], {}
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        path = reference/"cases"/(case["key"]+".npz")
        lp = local/(case["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(lp) == local_records[case["key"]]["arrays_sha256"]
        inputs.append((case, np.load(path), np.load(lp)))
        saved[case["key"]] = {}
    emit(out, out/"design.json", dict(experiment=266, retrospective=True, model_calls=0, cases=design["cases"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={name:sha(ROOT/name) for name in SOURCES}, reference_status_sha256=sha(reference/"status.json"),
        input_local_audit_sha256=sha(local/"audit.json"), checkpoint=checkpoint))
    checks, rows, alignments, margins, raw = [], [], [], [], []
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            def weight(name):
                return handle.get_tensor(f"model.layers.{layer-1}."+name).float().numpy()
            gamma = weight("post_attention_layernorm.weight")
            weights = {name:weight("mlp."+name+"_proj.weight") for name in ("gate", "up", "down")}
            matrices = {name:torch.tensor(value, device="mps") for name, value in weights.items()}
            down = weights["down"].astype(float)
            for case, values, previous in inputs:
                key, length, start = case["key"], len(case["false_ids"]), case["position"]
                for background in ("H", "L"):
                    prefix = f"{background}_{layer}_"
                    x = values[prefix+"x"]
                    nodes = corners(anchor(values, layer), x, gamma, eps, matrices, columns[layer-1], key, layer, length, start)
                    for name in ("gate", "up", "phi"):
                        checked(checks, "exact_old_"+name, nodes[background][name], values[prefix+name], 0.)
                    writes = {}
                    for mode in ("adaptive", "fixed", "finite"):
                        products = {corner:(node["phi"]*node["up"] if mode == "finite" else node["product_"+mode]) for corner, node in nodes.items()}
                        writes[mode] = {corner:project(value, matrices["down"], length, start) for corner, value in products.items()}
                        for corner, value in products.items():
                            error = float(np.max(abs(value.astype(float)@down.T-writes[mode][corner])))
                            raw.append(dict(key=key, background=background, layer=layer, mode=mode, corner=corner,
                                error=error, original_bound=.0001, passes=error <= .0001))
                    mixed = {mode:w["H"].astype(float)-w["P"].astype(float)-w["D"].astype(float)+w["L"].astype(float) for mode, w in writes.items()}
                    for mode in ("adaptive", "fixed"):
                        checked(checks, "exact265_mixed_"+mode, mixed[mode], previous[prefix+mode], 0.)
                    for mode, value in mixed.items():
                        saved[key][prefix+mode] = value
                        for scope, pick in (("all", slice(None)), ("verdict", -1)):
                            m = metrics(value[pick], mixed["finite"][pick])
                            rows.append(dict(key=key, **case["group"], background=background, layer=layer, mode=mode, scope=scope,
                                eligible=m["norm"] >= .001, **m, passes=m["norm"] >= .001 and m["relative"] <= .25 and m["cosine"] >= .95))
                    for mode, w in writes.items():
                        alone, conditional = w["D"].astype(float)-w["L"].astype(float), w["H"].astype(float)-w["P"].astype(float)
                        saved[key][prefix+mode+"_direction_alone"] = alone
                        saved[key][prefix+mode+"_direction_conditional"] = conditional
                        for scope, pick in (("all", slice(None)), ("verdict", -1)):
                            m = metrics(alone[pick], conditional[pick])
                            alignments.append(dict(key=key, **case["group"], background=background, layer=layer, mode=mode, scope=scope, **m,
                                alone_norm=float(np.linalg.norm(alone[pick]))))
                        if layer == 28:
                            for corner, value in w.items():
                                score = finite_scores((x+value).astype(np.float32)[-1], head)
                                margins.append(dict(key=key, **case["group"], background=background, mode=mode, corner=corner,
                                    margin=float(score[1]-score[0])))
            print("finite activation interaction control", layer, "/28", flush=True)
    names = ("comparison.csv", "alignment.csv", "margins.csv", "raw-projections.csv")
    for name, data in zip(names, (rows, alignments, margins, raw), strict=True):
        table(out/name, data)
    assert len(rows) == len(alignments) == 4*2*28*3*2 and len(margins) == 4*2*3*4
    records = []
    for case, values, previous in inputs:
        path = out/(case["key"]+".npz")
        np.savez_compressed(path, **saved[case["key"]])
        records.append(dict(key=case["key"], arrays_sha256=sha(path), bytes=path.stat().st_size))
        values.close()
        previous.close()
    decisions = {}
    for background in ("H", "L"):
        for mode in ("adaptive", "fixed"):
            group = [r for r in rows if r["background"] == background and r["mode"] == mode and r["scope"] == "all" and r["eligible"]]
            decisions[background+"_"+mode] = dict(eligible=len(group), passes=sum(r["passes"] for r in group),
                meets_90=bool(group and sum(r["passes"] for r in group) >= .9*len(group)))
    emit(out, out/"audit.json", dict(status="passed", experiment=266, retrospective=True, model_calls=0,
        source_sha256=sha(__file__), design_sha256=sha(out/"design.json"), checks=checks, records=records, decisions=decisions,
        primary_sparse_interaction_fidelity=decisions["H_adaptive"]["meets_90"] and decisions["L_adaptive"]["meets_90"],
        raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        csv_sha256={name:sha(out/name) for name in names},
        limitation="Same-input primitive finite-SiLU control on old approximate trajectories; no fresh/native whole-response validation or learned circuit."))
    print(decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference.resolve(), args.local.resolve(), args.out.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/"failure.json", dict(error=repr(exc)))
        raise
