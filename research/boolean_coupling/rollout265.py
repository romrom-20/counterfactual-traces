"""Remove same-input MLP interaction while retaining changing receiving states."""

import argparse
import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .core225b import array, load_model, observe
from .input261b import Instability, polynomial
from .input263 import plans
from .local265 import ROOT, bind, emit, read
from .mlp258 import runtime
from .silu258 import predict
from .transmission265 import additive, corners, digest, project, random_write

ACTIVE = None
ARMS = ("F", "Z", "H", "A", "K", "R", "T")


def execute(model, case, columns):
    start, length, key = case["position"], len(case["false_ids"]), case["key"]
    bank, live, saved, checks, records, handles, statuses, counts = {}, {}, {}, [], [], [], {}, Counter()
    arm = "F"
    eps = float(model.config.rms_norm_eps)
    qgamma = [array(layer.input_layernorm.weight).astype(float) for layer in model.model.layers]
    gamma = [array(layer.post_attention_layernorm.weight) for layer in model.model.layers]

    def install(index, layer):
        number = index+1
        matrices = dict(gate=layer.mlp.gate_proj.weight, up=layer.mlp.up_proj.weight)

        def query_raw(module, args):
            live[index] = dict(qx=array(args[0][0, start+1:]))
            if arm == "F":
                bank[index] = dict(qx=live[index]["qx"].astype(float))

        def query_input(module, args):
            old = array(args[0][0, start+1:])
            changed = args[0]
            if arm == "F":
                bank[index]["qz"] = old.astype(float)
            if arm not in ("F", "T"):
                linear, cubic = polynomial(bank[index]["qx"], live[index]["qx"].astype(float), bank[index]["qz"], qgamma[index], eps)
                value = (linear+cubic).astype(np.float32)
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(value[None], device="mps")
                checked(checks, "query_installation", array(changed[0, start+1:]), value, 0.)
                checked(checks, "query_source_prefix", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
            live[index].update(qoriginal=old, qinstalled=array(changed[0, start+1:]))
            return (changed,)

        def mlp_raw(module, args):
            live[index]["x"] = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["x"] = live[index]["x"]

        def normalized(module, args, output):
            old = array(output[0, start+1:])
            live[index]["original_z"] = old
            if arm == "F":
                bank[index]["z"] = old
            if arm not in ("F", "T"):
                parts = plans(bank[index]["x"].astype(float), live[index]["x"].astype(float), bank[index]["z"].astype(float), gamma[index].astype(float), eps)
                live[index]["desired_z"] = (parts["linear"]+parts["base"]+parts["direction"]).astype(np.float32)

        def branch_input(branch):
            def hook(module, args):
                changed = args[0]
                if arm not in ("F", "T"):
                    changed = args[0].clone()
                    changed[:, start+1:] = torch.tensor(live[index]["desired_z"][None], device="mps")
                    checked(checks, "branch_prefix_source", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                live[index][branch+"_z_digest"] = digest(array(changed[0, start+1:]))
                live[index]["installed_z"] = array(changed[0, start+1:])
                return (changed,)
            return hook

        def branch_output(branch):
            def hook(module, args, output):
                live[index][branch] = array(output[0, start+1:])
            return hook

        def activation(module, args, output):
            live[index]["phi"] = array(output[0, start+1:])
            if arm == "F":
                bank[index].update(gate=live[index]["gate"], phi=live[index]["phi"])

        def product(module, args):
            data = live[index]
            checked(checks, "original_product", array(args[0][0, start+1:]), data["phi"]*data["up"], 0.)
            installed = data["phi"]*data["up"]
            if arm not in ("F", "T"):
                quad, mask, _, _, _, _ = predict(bank[index]["gate"].astype(float), bank[index]["phi"].astype(float),
                    data["gate"].astype(float), data["up"].astype(float), columns[index], key, number)
                installed = np.where(mask, data["phi"], quad.astype(np.float32)).astype(np.float32)*data["up"]
                data["H_product_digest"] = digest(installed)
                counts["ranked_masks"] += 1
                if arm in ("Z", "A", "K", "R"):
                    nodes = corners(bank[index], data["x"], gamma[index], eps, matrices, columns[index], key, number, length, start)
                    for name in ("gate", "up", "phi"):
                        checked(checks, "exact_own_H_"+name, nodes["H"][name], data[name], 0.)
                    checked(checks, "exact_own_H_product", nodes["H"]["product_adaptive"], installed, 0.)
                    mode = "fixed" if arm == "K" else "adaptive"
                    alternative = additive(nodes, mode)
                    if arm == "R":
                        data["alternative_write"] = project(alternative, layer.mlp.down_proj.weight, length, start)
                    else:
                        installed = alternative
                    data["corner_digest"] = {corner:{name:digest(node[name]) for name in ("z", "gate", "up", "phi", "mask", "product_adaptive", "product_fixed")} for corner, node in nodes.items()}
                    counts["pointwise_corner_factorials"] += 1
            data["product_digest"] = digest(installed)
            changed = args[0]
            if arm not in ("F", "T"):
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(installed[None], device="mps")
                checked(checks, "product_installation", array(changed[0, start+1:]), installed, 0.)
                checked(checks, "product_prefix_source", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
            return (changed,)

        def write(module, args, output):
            data = live[index]
            raw = array(output[0, start+1:])
            changed = output
            if arm == "R":
                difference = raw.astype(float)-data["alternative_write"].astype(float)
                random = random_write(difference, key, number)
                expected = raw-random
                changed = output.clone()
                changed[:, start+1:] = torch.tensor(expected[None], device="mps")
                checked(checks, "random_installation", array(changed[0, start+1:]), expected, 0.)
                checked(checks, "random_source_prefix", array(changed[0, :start+1]), array(output[0, :start+1]), 0.)
                sizes = np.linalg.norm(difference, axis=-1)
                error = abs(np.linalg.norm(random.astype(float), axis=-1)-sizes)/np.maximum(sizes, 1e-8)
                checked(checks, "matched_random_write_norm", error, np.zeros_like(error), 1e-5)
                saved[f"{arm}_{number}_random"] = random
                saved[f"{arm}_{number}_alternative_write"] = data["alternative_write"]
            for name in ("x", "original_z", "installed_z", "qx", "qoriginal", "qinstalled"):
                saved[f"{arm}_{number}_"+name] = data[name]
            saved[f"{arm}_{number}_write_raw"] = raw
            saved[f"{arm}_{number}_write"] = array(changed[0, start+1:])
            records.append(dict(arm=arm, layer=number, product_digest=data["product_digest"],
                H_product_digest=data.get("H_product_digest"), gate_z_digest=data["gate_z_digest"],
                up_z_digest=data["up_z_digest"], corner_digest=data.get("corner_digest")))
            return changed

        def guard(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            tail = array(value[0, start:])
            if arm == "F":
                saved[f"F_tail{number}"] = tail
                bank[index]["prefix"] = hashlib.sha256(array(value[0, :start]).tobytes()).hexdigest()
            else:
                ratio = float(np.linalg.norm(tail.astype(float))/max(np.linalg.norm(saved[f"F_tail{number}"].astype(float)), .001))
                if not np.isfinite(tail).all() or ratio > 100:
                    raise Instability(dict(arm=arm, layer=number, ratio=ratio))
                assert hashlib.sha256(array(value[0, :start]).tobytes()).hexdigest() == bank[index]["prefix"]

        return ((layer.input_layernorm, query_raw, True), (layer.self_attn.q_proj, query_input, True),
            (layer.post_attention_layernorm, mlp_raw, True), (layer.post_attention_layernorm, normalized, False),
            (layer.mlp.gate_proj, branch_input("gate"), True), (layer.mlp.gate_proj, branch_output("gate"), False),
            (layer.mlp.up_proj, branch_input("up"), True), (layer.mlp.up_proj, branch_output("up"), False),
            (layer.mlp.act_fn, activation, False), (layer.mlp.down_proj, product, True),
            (layer.mlp.down_proj, write, False), (layer, guard, False))

    for index, layer in enumerate(model.model.layers):
        for module, function, before in install(index, layer):
            handles.append((module.register_forward_pre_hook if before else module.register_forward_hook)(function))
    try:
        for arm in ARMS:
            try:
                states, _, scores = observe(model, torch.tensor([case["false_ids" if arm in ("F", "Z") else "true_ids"]], device="mps"))
                saved[arm+"_trajectory"] = np.stack([array(value[0, start:]) for value in states])
                saved[arm+"_scores"] = array(scores)
                assert np.isfinite(saved[arm+"_scores"]).all()
                statuses[arm] = dict(status="complete")
            except Instability as exc:
                assert arm in ("A", "K", "R")
                statuses[arm] = dict(status="guard_failed", reason=str(exc))
            counts["full_forwards_attempted"] += 1
    finally:
        for handle in handles:
            handle.remove()
    checked(checks, "zero_trajectory", saved["Z_trajectory"], saved["F_trajectory"], 0.)
    checked(checks, "zero_scores", saved["Z_scores"], saved["F_scores"], 0.)
    for name in ("H", "A", "K", "R"):
        if statuses[name]["status"] == "complete":
            checked(checks, "known_edit_source_exact", saved[name+"_trajectory"][:, 0], saved["T_trajectory"][:, 0], 0.)
    return saved, checks, records, statuses, counts


def main(reference, local, out):
    global ACTIVE
    assert not out.exists() and read(local/"audit.json")["status"] == "passed"
    assert sha(qwen.__file__) == read(ROOT/"results/receiver-mlp-curvature-258/design.json")["qwen_source_sha256"]
    out.mkdir(parents=True)
    ACTIVE = out
    (out/"cases").mkdir()
    sources = bind(out)
    design, status = (read(reference/n) for n in ("design.json", "status.json"))
    assert read(local/"design.json")["reference_status_sha256"] == sha(reference/"status.json")
    emit(out, out/"design.json", dict(experiment=265, stage="rollout", retrospective=True,
        source_hashes=sources, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        cases=design["cases"], reference_design_sha256=sha(reference/"design.json"), reference_status_sha256=sha(reference/"status.json"),
        reference_audit_sha256=sha(reference/"audit.json"), local_audit_sha256=sha(local/"audit.json"), arms=ARMS))
    model = load_model(True)
    assert runtime(model) == read(reference/"runtime.json")
    emit(out, out/"runtime.json", runtime(model))
    with np.load(reference/"norm-weights.npz") as w:
        columns = w["columns"]
    entries, counts, total_bytes = [], Counter(), sum(r["bytes"] for r in read(local/"audit.json")["records"])
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        saved, checks, records, statuses, work = execute(model, case, columns)
        old_path = reference/"cases"/(case["key"]+".npz")
        assert sha(old_path) == entry["arrays_sha256"] and sha(old_path.with_suffix(".json")) == entry["metadata_sha256"]
        with np.load(old_path) as old:
            for arm in ("F", "H", "T"):
                for name in ("trajectory", "scores"):
                    checked(checks, "old263_"+arm+"_"+name, saved[arm+"_"+name], old[arm+"_"+name], 0.)
        assert shutil.disk_usage(out).free >= 512*1024**2
        path = out/"cases"/(case["key"]+".npz")
        np.savez_compressed(path, **saved)
        total_bytes += path.stat().st_size
        assert total_bytes <= 300*1024**2
        emit(out, path.with_suffix(".json"), dict(key=case["key"], arrays_sha256=sha(path), checks=checks,
            node_records=records, statuses=statuses, counts=dict(work)))
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        counts.update(work)
        print("local-mixing continuation", len(entries), "/4", statuses, flush=True)
    emit(out, out/"status.json", dict(status="complete", cases=entries, counts=dict(counts), combined_raw_bytes=total_bytes))


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
