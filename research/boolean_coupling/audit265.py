"""Qualify direct265 product interventions and test the frozen global account."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha
from .audit261b import expansion, metrics, norm
from .checkpoint238 import locate, read_head
from .input263 import plans
from .local265 import ROOT, anchor, emit, finite_scores, read, table
from .rollout265 import ARMS
from .transmission265 import additive, corners, digest, full, project, random_write

ACTIVE = None


def main(reference, local, run):
    global ACTIVE
    assert not (run/"audit.json").exists() and not (run/"failure-audit.json").exists()
    design, status = read(run/"design.json"), read(run/"status.json")
    assert design["experiment"] == 265 and design["stage"] == "rollout" and status["status"] == "complete"
    ACTIVE = run
    assert design["reference_status_sha256"] == sha(reference/"status.json")
    assert design["reference_design_sha256"] == sha(reference/"design.json")
    assert design["reference_audit_sha256"] == sha(reference/"audit.json")
    assert design["local_audit_sha256"] == sha(local/"audit.json")
    for folder in (run, local):
        for name, value in read(folder/"design.json")["source_hashes"].items():
            assert sha(ROOT/name) == sha(folder/"sources"/Path(name).name) == value
    local_audit = read(local/"audit.json")
    assert local_audit["status"] == "passed" and local_audit["counts"]["exact_old_full_shape_MLP_replays"] == 224
    for c in local_audit["checks"]:
        assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
    for name, value in local_audit["csv_sha256"].items():
        assert sha(local/name) == value
    folder, checkpoint = locate(ROOT)
    assert checkpoint == read(local/"design.json")["checkpoint"]
    head = read_head(folder)
    with np.load(reference/"norm-weights.npz") as w:
        columns, eps = w["columns"], float(w["eps"])
    refs = {r["key"]:r for r in read(reference/"status.json")["cases"]}
    inputs = []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        path = run/"cases"/(case["key"]+".npz")
        old = reference/"cases"/(case["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        assert sha(old) == refs[case["key"]]["arrays_sha256"]
        meta = read(path.with_suffix(".json"))
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        lookup = {(r["arm"], r["layer"]):r for r in meta["node_records"]}
        assert len(lookup) == len(meta["node_records"])
        inputs.append((case, np.load(path), np.load(old), meta, lookup))
    checks, raw, outcomes, margins, counts = [], [], [], [], Counter()
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            def weight(name):
                return handle.get_tensor(f"model.layers.{layer-1}."+name).float().numpy()
            gamma, qgamma = (weight(name) for name in ("post_attention_layernorm.weight", "input_layernorm.weight"))
            weights = {name:weight("mlp."+name+"_proj.weight") for name in ("gate", "up", "down")}
            matrices = {name:torch.tensor(value, device="mps") for name, value in weights.items()}
            for case, values, old, meta, lookup in inputs:
                key, start, length = case["key"], case["position"], len(case["false_ids"])
                base = anchor(old, layer)
                checked(checks, "exact_False_MLP_input", values[f"F_{layer}_x"], base["x"], 0.)
                checked(checks, "exact_False_normalized_input", values[f"F_{layer}_installed_z"], base["z"], 0.)
                qx0, qz0 = (values[f"F_{layer}_"+n].astype(float) for n in ("qx", "qoriginal"))
                for arm in ARMS:
                    prefix = f"{arm}_{layer}_"
                    if prefix+"write" not in values:
                        assert meta["statuses"][arm]["status"] == "guard_failed"
                        continue
                    record = lookup[arm, layer]
                    x, zo, zi = (values[prefix+n] for n in ("x", "original_z", "installed_z"))
                    checked(checks, "public_MLP_norm_primal", norm(x.astype(float), gamma.astype(float), eps), zo.astype(float), .0001)
                    linear, _, _ = expansion(base["x"].astype(float), x.astype(float), base["z"].astype(float), gamma.astype(float), eps, checks, counts)
                    point_plans = plans(base["x"].astype(float), x.astype(float), base["z"].astype(float), gamma.astype(float), eps)
                    checked(checks, "independent_MLP_tangent", linear, point_plans["linear"], 1e-10)
                    expected_z = point_plans["linear"]+point_plans["base"]+point_plans["direction"]
                    checked(checks, "smooth_MLP_factorization", expected_z, base["z"]+norm(x.astype(float), gamma.astype(float), eps)-norm(base["x"].astype(float), gamma.astype(float), eps), 1e-10)
                    checked(checks, "exact_normalized_input_installation", zi, (expected_z.astype(np.float32) if arm not in ("F", "T") else zo), 0.)
                    assert digest(zi) == record["gate_z_digest"] == record["up_z_digest"]
                    qx, qo, qi = (values[prefix+n].astype(float) for n in ("qx", "qoriginal", "qinstalled"))
                    checked(checks, "public_query_norm_primal", norm(qx, qgamma.astype(float), eps), qo, .0001)
                    qlinear, qcubic, _ = expansion(qx0, qx, qz0, qgamma.astype(float), eps, checks, counts)
                    checked(checks, "exact_query_installation", qi, ((qlinear+qcubic).astype(np.float32).astype(float) if arm not in ("F", "T") else qo), 0.)
                    if arm in ("F", "T"):
                        gate, up = (project(zi, matrices[name], length, start) for name in ("gate", "up"))
                        with torch.no_grad():
                            phi = torch.nn.functional.silu(full(gate, length, start))[0, start+1:].cpu().numpy()
                        product = phi*up
                        nodes = dict(H=dict(z=zi, gate=gate, up=up))
                    else:
                        nodes = corners(base, x, gamma, eps, matrices, columns[layer-1], key, layer, length, start)
                        product = nodes["H"]["product_adaptive"]
                        assert digest(product) == record["H_product_digest"]
                        if record["corner_digest"] is not None:
                            assert record["corner_digest"] == {corner:{name:digest(node[name]) for name in ("z", "gate", "up", "phi", "mask", "product_adaptive", "product_fixed")} for corner, node in nodes.items()}
                        if arm in ("Z", "A", "K"):
                            product = additive(nodes, "fixed" if arm == "K" else "adaptive")
                    assert digest(product) == record["product_digest"]
                    raw_write = project(product, matrices["down"], length, start)
                    checked(checks, "exact_public_down_replay", raw_write, values[prefix+"write_raw"], 0.)
                    expected_write = raw_write
                    if arm == "R":
                        alternative_write = project(additive(nodes, "adaptive"), matrices["down"], length, start)
                        checked(checks, "exact_alternative_write", alternative_write, values[prefix+"alternative_write"], 0.)
                        random = random_write(raw_write.astype(float)-alternative_write.astype(float), key, layer)
                        checked(checks, "exact_random_control", random, values[prefix+"random"], 0.)
                        expected_write = raw_write-random
                    checked(checks, "exact_write_installation", expected_write, values[prefix+"write"], 0.)
                    if meta["statuses"][arm]["status"] == "complete":
                        checked(checks, "exact_block_residual_addition", x+expected_write, values[arm+"_trajectory"][layer-1, 1:], 0.)
                    for corner, node in nodes.items():
                        for name in ("gate", "up"):
                            error = float(np.max(abs(node["z"].astype(float)@weights[name].astype(float).T-node[name])))
                            raw.append(dict(key=key, arm=arm, layer=layer, corner=corner, node=name,
                                error=error, original_bound=.0001, passes=error <= .0001))
                    error = float(np.max(abs(product.astype(float)@weights["down"].astype(float).T-raw_write)))
                    raw.append(dict(key=key, arm=arm, layer=layer, corner="installed", node="down",
                        error=error, original_bound=.0001, passes=error <= .0001))
                    counts["audited_MLP_points"] += 1
            print("continuation primitive audit", layer, "/28", flush=True)
    for case, values, old, meta, _ in inputs:
        key = case["key"]
        for arm in ("F", "H", "T"):
            for name in ("trajectory", "scores"):
                checked(checks, "old263_replay_"+arm, values[arm+"_"+name], old[arm+"_"+name], 0.)
        native = values["T_trajectory"].astype(float)-values["F_trajectory"].astype(float)
        baseline = values["H_trajectory"][-1].astype(float)
        add = old["P_trajectory"][-1].astype(float)+old["D_trajectory"][-1].astype(float)-old["L_trajectory"][-1].astype(float)
        interaction = baseline-add
        mbase, madd = (float(np.diff(finite_scores(value[-1], head))[0]) for value in (baseline, add))
        signed_vectors = dict(target=interaction, additive_final_state=add)
        for arm in ("H", "A", "K", "R"):
            complete = meta["statuses"][arm]["status"] == "complete"
            row = dict(key=key, **case["group"], arm=arm, status=meta["statuses"][arm]["status"], interaction_pass=False, native_pass=False)
            if complete:
                absolute = values[arm+"_trajectory"][-1].astype(float)
                candidate = baseline-absolute
                signed_vectors[arm] = candidate
                pred = absolute-values["F_trajectory"][-1].astype(float)
                interaction_ok, native_ok = [], []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    im, nm = metrics(candidate[pick], interaction[pick]), metrics(pred[pick], native[-1, pick])
                    row.update({scope+"_interaction_"+k:v for k, v in im.items()})
                    row.update({scope+"_native_"+k:v for k, v in nm.items()})
                    interaction_ok.append(im["norm"] >= .001 and im["relative"] <= .25 and im["cosine"] >= .95)
                    native_ok.append(nm["norm"] >= .001 and nm["relative"] <= .05 and nm["cosine"] >= .99)
                cpu_scores = finite_scores(absolute[-1], head)
                checked(checks, "finite_head", cpu_scores, values[arm+"_scores"], .005)
                margin = float(cpu_scores[1]-cpu_scores[0])
                row["interaction_margin_error"] = abs(margin-madd)
                row["interaction_margin_bound"] = .005+.25*abs(mbase-madd)
                row["native_margin_error"] = abs(float(np.diff(values[arm+"_scores"].astype(float))[0])-float(np.diff(values["T_scores"].astype(float))[0]))
                row["interaction_pass"] = bool(all(interaction_ok) and row["interaction_margin_error"] <= row["interaction_margin_bound"])
                row["native_pass"] = bool(all(native_ok) and row["native_margin_error"] <= .1)
                margins.append(dict(key=key, **case["group"], arm=arm, H_margin=mbase, additive_margin=madd, arm_margin=margin,
                    target_effect=mbase-madd, predicted_effect=mbase-margin))
            outcomes.append(row)
        path = run/("signed-interactions-"+key+".npz")
        assert not path.exists()
        np.savez_compressed(path, **signed_vectors)
        values.close()
        old.close()
    for name, data in (("outcomes.csv", outcomes), ("margins.csv", margins), ("raw-projections.csv", raw)):
        table(run/name, data)
    decisions = {arm:dict(total=4, interaction_passes=sum(r["interaction_pass"] for r in outcomes if r["arm"] == arm),
        native_passes=sum(r["native_pass"] for r in outcomes if r["arm"] == arm),
        guard_failed=sum(r["status"] == "guard_failed" for r in outcomes if r["arm"] == arm)) for arm in ("H", "A", "K", "R")}
    emit(run, run/"audit.json", dict(status="passed", experiment=265, stage="rollout", retrospective=True,
        source_sha256=sha(__file__), design_sha256=sha(run/"design.json"), status_sha256=sha(run/"status.json"),
        checks=checks, counts=dict(counts), decisions=decisions,
        primary_global_sufficiency=decisions["A"]["interaction_passes"] >= 3 and decisions["R"]["interaction_passes"] <= 1,
        raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        csv_sha256={name:sha(run/name) for name in ("outcomes.csv", "margins.csv", "raw-projections.csv")},
        vectors_sha256={p.name:sha(p) for p in run.glob("signed-interactions-*.npz")},
        numerical_boundary="Matched full-shape publicFloat32 nodes, exact installations and checked smooth RMS components; original CPU64 failures retained, no whole-model real-arithmetic certificate.",
        limitation="Retrospective same-input product intervention through evolving states; no fresh forecast, learned circuit or complete Question B solution."))
    print(decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference.resolve(), args.local.resolve(), args.run.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/"failure-audit.json", dict(error=repr(exc)))
        raise
