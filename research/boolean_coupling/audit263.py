"""Qualify263 branch interventions at the declared Float32 boundary and score all arms."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha
from .audit261b import expansion, metrics, norm
from .checkpoint238 import locate, read_head
from .core225b import write_json
from .silu258 import predict, sigmoid

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "Z", "B", "L", "G", "U", "P", "D", "E", "H", "T")


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(run):
    assert not (run/"audit.json").exists()
    design, status = read(run/"design.json"), read(run/"status.json")
    assert design["experiment"] == 263 and design["primary"] == "E" and design["retrospective"]
    for name, digest in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == digest
    assert sha(run/"norm-weights.npz") == status["norm_weights_sha256"]
    folder, checkpoint = locate(ROOT)
    head = read_head(folder)
    with np.load(run/"norm-weights.npz") as w:
        weights = {k:w[k].astype(float) for k in w.files}
    eps = float(weights["eps"])
    assert eps == read(folder/"config.json")["rms_norm_eps"]
    inputs = []
    checks, raw, work, outcomes, rounding, interactions = [], [], Counter(), [], [], []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        p = run/"cases"/(case["key"]+".npz")
        assert sha(p) == entry["arrays_sha256"] and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(p.with_suffix(".json"))
        for c in meta["checks"]:
            assert np.isfinite(c["error"]) and c["error"] <= c["bound"]
        inputs.append((case, np.load(p), meta))
    assert len(inputs) == 4
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            def weight(name):
                return handle.get_tensor(f"model.layers.{layer-1}."+name).float().numpy()
            gamma, qgamma = (weight(name).astype(float) for name in ("post_attention_layernorm.weight", "input_layernorm.weight"))
            checked(checks, "checkpoint_MLP_gamma", gamma, weights["mlp_norm"][layer-1], 0.)
            checked(checks, "checkpoint_query_gamma", qgamma, weights["input_norm"][layer-1], 0.)
            matrices32 = {branch:weight("mlp."+branch+"_proj.weight") for branch in ("gate", "up")}
            matrices = {branch:v.astype(float) for branch, v in matrices32.items()}
            device_matrices = {branch:torch.tensor(v, device="mps") for branch, v in matrices32.items()}
            for case, values, meta in inputs:
                key, p0 = case["key"], f"F_{layer}_"
                x0, z0, qx0, qz0 = (values[p0+n].astype(float) for n in ("x", "original_z", "qx", "qoriginal"))
                scale = np.sqrt(np.mean(x0*x0, axis=-1, keepdims=True)+eps)
                f0 = gamma*x0/scale
                for arm in ARMS:
                    prefix = f"{arm}_{layer}_"
                    if prefix+"write" not in values:
                        assert meta["statuses"][arm]["status"] == "guard_failed"
                        continue
                    x, original = (values[prefix+n].astype(float) for n in ("x", "original_z"))
                    checked(checks, "public_MLP_norm_primal", norm(x, gamma, eps), original, 1e-4)
                    linear, _, _ = expansion(x0, x, z0, gamma, eps, checks, work)
                    d = x-x0
                    a, b = (np.mean(v, axis=-1, keepdims=True)/scale**2 for v in (x0*d, d*d))
                    rho = (1+2*a+b)**(-.5)
                    plans = dict(linear=linear, base=(rho-1+a)*f0, direction=(rho-1)*gamma*d/scale, energy=-.5*b*f0)
                    if arm not in ("F", "T"):
                        for name, v in plans.items():
                            checked(checks, "exact_saved_plan_"+name, v, values[prefix+"plan_"+name], 0.)
                    checked(checks, "complete_factorization_identity", linear+plans["base"]+plans["direction"], z0+norm(x, gamma, eps)-f0, 1e-10)
                    qx, qo, qi = (values[prefix+n].astype(float) for n in ("qx", "qoriginal", "qinstalled"))
                    checked(checks, "public_query_norm_primal", norm(qx, qgamma, eps), qo, 1e-4)
                    qlin, qcurv, _ = expansion(qx0, qx, qz0, qgamma, eps, checks, work)
                    qexpected = (qlin+qcurv).astype(np.float32).astype(float) if arm not in ("F", "T") else qo
                    checked(checks, "exact_query_installation", qi, qexpected, 0.)
                    for branch in ("gate", "up"):
                        expected = original
                        if arm in ("Z", "L") or (arm == "G" and branch == "up") or (arm == "U" and branch == "gate"):
                            expected = linear
                        elif arm in ("P", "D", "E", "H"):
                            expected = linear
                            for name in {"P":("base",), "D":("direction",), "E":("energy",), "H":("base", "direction")}[arm]:
                                expected = expected+plans[name]
                        installed = values[prefix+branch+"_z"].astype(float)
                        checked(checks, "exact_branch_input_installation", installed, expected.astype(np.float32).astype(float), 0.)
                        target = values[prefix+branch].astype(float)
                        error = float(np.max(abs(installed@matrices[branch].T-target)))
                        raw.append(dict(key=key, arm=arm, layer=layer, branch=branch, error=error, original_bound=1e-4, passes=error <= 1e-4))
                        with torch.no_grad():
                            full = torch.zeros((1, len(case["false_ids"]), installed.shape[-1]), device="mps", dtype=torch.float32)
                            full[:, case["position"]+1:] = torch.tensor(installed.astype(np.float32)[None], device="mps")
                            replay = torch.nn.functional.linear(full, device_matrices[branch])[0, case["position"]+1:].cpu().numpy()
                        checked(checks, "exact_full_shape_public_"+branch+"_replay", replay, target, 0.)
                        work["full_shape_Float32_projections"] += 1
                        work["CPU64_projections"] += 1
                    gate, phi, up = (values[prefix+n] for n in ("gate", "phi", "up"))
                    checked(checks, "public_SiLU_primal", gate.astype(float)*sigmoid(gate.astype(float)), phi, 1e-4)
                    used = phi
                    if arm not in ("F", "T"):
                        quad, mask, _, bound, _, _ = predict(values[p0+"gate"].astype(float), values[p0+"phi"].astype(float), gate.astype(float), up.astype(float), weights["columns"][layer-1], key, layer)
                        checked(checks, "exact_quadratic_SiLU", quad, values[prefix+"quadratic"], 0.)
                        checked(checks, "exact_ranked_mask", mask.astype(int), values[prefix+"mask"].astype(int), 0.)
                        checked(checks, "exact_SiLU_bound", bound, values[prefix+"bound"], 0.)
                        used = np.where(mask, phi, quad.astype(np.float32)).astype(np.float32)
                    checked(checks, "exact_SiLU_installation", used, values[prefix+"used_phi"], 0.)
                    checked(checks, "exact_product_installation", used*up, values[prefix+"product"], 0.)
            print("branch/energy primitive audit", layer, "/28", flush=True)
    for case, values, meta in inputs:
        key = case["key"]
        truth = values["T_trajectory"].astype(float)-values["F_trajectory"].astype(float)
        def margin(arm):
            return float(np.diff(values[arm+"_scores"].astype(float))[0])
        for arm in ("B", "L", "G", "U", "P", "D", "E", "H"):
            row = dict(key=key, **case["group"], arm=arm, status=meta["statuses"][arm]["status"], passes=False)
            if row["status"] == "complete":
                pred = values[arm+"_trajectory"].astype(float)-values["F_trajectory"].astype(float)
                last = values[arm+"_trajectory"][-1, -1].astype(float)
                z = last/np.sqrt(np.mean(last*last)+head["final_eps"])*head["final_norm"]
                checked(checks, "finite_head", z@head["final_head"].T, values[arm+"_scores"], .005)
                ok = []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    m = metrics(pred[-1, pick], truth[-1, pick])
                    row.update({scope+"_"+k:v for k, v in m.items()})
                    ok.append(m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99)
                row["margin_error"] = abs(margin(arm)-margin("T"))
                row["passes"] = bool(all(ok) and row["margin_error"] <= .1)
            outcomes.append(row)
        hpass = meta["statuses"]["H"]["status"] == "complete"
        rr = dict(key=key, passes=False)
        if hpass:
            diff = values["H_trajectory"].astype(float)-values["B_trajectory"].astype(float)
            for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                rr[scope+"_relative"] = float(np.linalg.norm(diff[-1, pick]))/max(float(np.linalg.norm(truth[-1, pick])), 1e-30)
            rr["margin_difference"] = abs(margin("H")-margin("B"))
            rr["passes"] = bool(rr["all_relative"] <= .001 and rr["verdict_relative"] <= .001 and rr["margin_difference"] <= .005)
        rounding.append(rr)
        if all(meta["statuses"][a]["status"] == "complete" for a in ("B", "G", "U", "L")):
            interaction = values["B_trajectory"].astype(float)-values["G_trajectory"].astype(float)-values["U_trajectory"].astype(float)+values["L_trajectory"].astype(float)
            np.savez_compressed(run/("interaction-"+key+".npz"), vector=interaction)
            for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                target = truth[-1, pick]
                interactions.append(dict(key=key, scope=scope, norm=float(np.linalg.norm(interaction[-1, pick])),
                    relative_native_norm=float(np.linalg.norm(interaction[-1, pick]))/max(float(np.linalg.norm(target)), 1e-30),
                    signed_native_share=float(np.vdot(interaction[-1, pick], target))/max(float(np.vdot(target, target)), 1e-30),
                    margin_interaction=margin("B")-margin("G")-margin("U")+margin("L")))
        values.close()
    for name, data in (("outcomes.csv", outcomes), ("rounding-control.csv", rounding), ("interactions.csv", interactions), ("raw-projections.csv", raw)):
        table(run/name, data)
    decisions = {arm:dict(passes=sum(r["passes"] for r in outcomes if r["arm"] == arm), total=4,
        guard_failed=sum(r["status"] == "guard_failed" for r in outcomes if r["arm"] == arm)) for arm in ("B", "L", "G", "U", "P", "D", "E", "H")}
    write_json(run/"audit.json", dict(status="passed", experiment=263, retrospective=True, primary="E", checks=checks,
        source_sha256=sha(__file__), design_sha256=sha(run/"design.json"), status_sha256=sha(run/"status.json"),
        checkpoint=checkpoint, extra_work=dict(work), decisions=decisions, rounding_control_passes=sum(r["passes"] for r in rounding),
        raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=1e-4),
        csv_sha256={n:sha(run/n) for n in ("outcomes.csv", "rounding-control.csv", "interactions.csv", "raw-projections.csv")},
        interaction_sha256={p.name:sha(p) for p in run.glob("interaction-*.npz")},
        numerical_boundary="Exact matched-shape publicFloat32 projections and independently checked smooth normalization; raw CPU64 failures retained. Old displacement bridges/derivatives not recertified.",
        limitation="Four old contexts; own-state component interventions, not fresh generalization, cheap prediction or learned features."))
    print(json.dumps(dict(decisions=decisions, rounding_control_passes=sum(r["passes"] for r in rounding)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve())
    except Exception as exc:
        path = args.run/"failure-audit.json"
        assert not path.exists()
        write_json(path, dict(error=repr(exc)))
        raise
