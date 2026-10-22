"""Retrospective gate/up normalization factorial and residual-energy rescaling."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .core225b import array, load_model, observe, write_json
from .input261b import Instability, polynomial
from .mlp258 import runtime
from .silu258 import predict

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "Z", "B", "L", "G", "U", "P", "D", "E", "H", "T")
SOURCES = ("research/boolean_coupling/input263.py", "research/boolean_coupling/audit263.py",
    "docs/experiments/263-normalization-branches-and-energy.md", "research/boolean_coupling/input261b.py",
    "research/boolean_coupling/audit261b.py",
    "research/boolean_coupling/mlp258.py", "research/boolean_coupling/silu258.py",
    "research/boolean_coupling/curvature255.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/settings.json")
ACTIVE_OUTPUT = None


def read(path):
    return json.loads(path.read_text())


def emit(out, path, value):
    assert path.resolve().is_relative_to(out.resolve())
    assert not path.exists(), path
    write_json(path, value)


def plans(base, point, anchor, gamma, eps):
    d = point-base
    scale = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+eps)
    j = gamma*(d/scale-base*np.mean(base*d, axis=-1, keepdims=True)/scale**3)
    a, b = (np.mean(v, axis=-1, keepdims=True)/scale**2 for v in (base*d, d*d))
    rho = (1+2*a+b)**(-.5)
    f0 = gamma*base/scale
    return dict(linear=anchor+j, base=(rho-1+a)*f0, direction=(rho-1)*gamma*d/scale, energy=-.5*b*f0)


def execute(model, case, columns, out):
    start, key = case["position"], case["key"]
    saved, bank, live, statuses, checks, handles, counts = {}, {}, {}, {}, [], [], Counter()
    arm = "F"
    eps = float(model.config.rms_norm_eps)
    qgamma = [array(layer.input_layernorm.weight).astype(float) for layer in model.model.layers]
    gamma = [array(layer.post_attention_layernorm.weight).astype(float) for layer in model.model.layers]

    def install(index, layer):
        number = index+1

        def query_raw(module, args):
            live[index] = dict(qx=array(args[0][0, start+1:]))
            if arm == "F":
                bank[index] = dict(qx=live[index]["qx"].astype(float))

        def query_input(module, args):
            old = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["qz"] = old.astype(float)
            changed = args[0]
            if arm not in ("F", "T"):
                lin, curv = polynomial(bank[index]["qx"], live[index]["qx"].astype(float), bank[index]["qz"], qgamma[index], eps)
                value = (lin+curv).astype(np.float32)
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(value[None], device="mps")
                checked(checks, "query_installation", array(changed[0, start+1:]), value, 0.)
                checked(checks, "query_source_prefix_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                counts["query_cubic_maps"] += 1
            live[index].update(qoriginal=old, qinstalled=array(changed[0, start+1:]))
            return (changed,)

        def mlp_raw(module, args):
            live[index]["x"] = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["x"] = live[index]["x"].astype(float)

        def normalized(module, args, output):
            live[index]["original_z"] = array(output[0, start+1:])
            if arm == "F":
                bank[index]["z"] = live[index]["original_z"].astype(float)
            if arm not in ("F", "T"):
                values = plans(bank[index]["x"], live[index]["x"].astype(float), bank[index]["z"], gamma[index], eps)
                live[index].update({"plan_"+k:v for k, v in values.items()})
                counts["normalization_plans"] += 1

        def branch_input(branch):
            def hook(module, args):
                data = live[index]
                value = data["original_z"]
                if arm in ("Z", "L") or (arm == "G" and branch == "up") or (arm == "U" and branch == "gate"):
                    value = data["plan_linear"]
                elif arm in ("P", "D", "E", "H"):
                    value = data["plan_linear"]
                    for term in {"P":("base",), "D":("direction",), "E":("energy",), "H":("base", "direction")}[arm]:
                        value = value+data["plan_"+term]
                changed = args[0]
                if arm not in ("F", "T", "B"):
                    value = value.astype(np.float32)
                    changed = args[0].clone()
                    changed[:, start+1:] = torch.tensor(value[None], device="mps")
                    checked(checks, "branch_input_installation", array(changed[0, start+1:]), value, 0.)
                    checked(checks, "branch_prefix_source_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                data[branch+"_z"] = array(changed[0, start+1:])
                return (changed,)
            return hook

        def branch_output(branch):
            def hook(module, args, output):
                live[index][branch] = array(output[0, start+1:])
            return hook

        def activation(module, args, output):
            live[index]["phi"] = array(output[0, start+1:])
            if arm == "F":
                bank[index].update(gate=live[index]["gate"].astype(float), phi=live[index]["phi"].astype(float))

        def product(module, args):
            data = live[index]
            checked(checks, "original_product", array(args[0][0, start+1:]), data["phi"]*data["up"], 0.)
            changed, used = args[0], data["phi"]
            if arm not in ("F", "T"):
                quadratic, mask, _, bound, _, _ = predict(bank[index]["gate"], bank[index]["phi"], data["gate"].astype(float), data["up"].astype(float), columns[index], key, number)
                used = np.where(mask, data["phi"], quadratic.astype(np.float32)).astype(np.float32)
                installed = used*data["up"]
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(installed[None], device="mps")
                checked(checks, "installed_product", array(changed[0, start+1:]), installed, 0.)
                checked(checks, "product_source_prefix_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                assert int(mask.sum()) == int(.01*mask.size)
                data.update(mask=mask, bound=bound, quadratic=quadratic)
                counts["ranked_SiLU_maps"] += 1
            data.update(used_phi=used, product=array(changed[0, start+1:]))
            return (changed,)

        def write(module, args, output):
            live[index]["write"] = array(output[0, start+1:])
            saved.update({f"{arm}_{number}_{k}":v for k, v in live[index].items()})

        def guard(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            tail = array(state[0, start:])
            if arm != "F":
                ratio = float(np.linalg.norm(tail.astype(float))/max(np.linalg.norm(saved[f"F_tail{number}"].astype(float)), .001))
                if not np.isfinite(tail).all() or ratio > 100:
                    raise Instability(dict(arm=arm, layer=number, ratio=ratio))
            saved[f"{arm}_tail{number}"] = tail
            digest = hashlib.sha256(array(state[0, :start]).tobytes()).hexdigest()
            if arm == "F":
                bank[index]["prefix_hash"] = digest
            assert digest == bank[index]["prefix_hash"]

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
        with torch.no_grad():
            for arm in ARMS:
                try:
                    states, _, scores = observe(model, torch.tensor([case["false_ids" if arm in ("F", "Z") else "true_ids"]], device="mps"))
                    saved[arm+"_trajectory"] = np.stack([array(state[0, start:]) for state in states])
                    saved[arm+"_scores"] = array(scores)
                    assert np.isfinite(saved[arm+"_scores"]).all()
                    statuses[arm] = dict(status="complete")
                except Instability as exc:
                    assert arm in ("G", "U", "P", "D", "E", "H")
                    statuses[arm] = dict(status="guard_failed", reason=str(exc))
                counts["full_forwards_attempted"] += 1
    except Exception:
        np.savez_compressed(out/("partial-"+key+".npz"), **saved)
        raise
    finally:
        for handle in handles:
            handle.remove()
    checked(checks, "zero_trajectory", saved["Z_trajectory"], saved["F_trajectory"], 0.)
    checked(checks, "zero_scores", saved["Z_scores"], saved["F_scores"], 0.)
    for name in ARMS:
        if name not in ("F", "Z", "T") and statuses[name]["status"] == "complete":
            checked(checks, "known_edit_source_exact", saved[name+"_trajectory"][:, 0], saved["T_trajectory"][:, 0], 0.)
    return saved, checks, statuses, counts


def main(original, reference, factors, out):
    global ACTIVE_OUTPUT
    assert not out.exists()
    out.mkdir(parents=True)
    ACTIVE_OUTPUT = out
    (out/"sources").mkdir()
    (out/"cases").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/Path(name).name)
    old = read(original/"design.json")
    assert read(original/"audit.json")["status"] == read(factors/"audit.json")["status"] == "passed"
    assert sha(original/"weights.npz") == old["weights_sha256"] and sha(qwen.__file__) == old["qwen_source_sha256"]
    design = read(reference/"design.json")
    emit(out, out/"design.json", dict(experiment=263, retrospective=True, primary="E", cases=design["cases"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p:sha(ROOT/p) for p in SOURCES}, original_design_sha256=sha(original/"design.json"),
        original_audit_sha256=sha(original/"audit.json"), linear_reference_status_sha256=sha(reference/"status.json"), factors_audit_sha256=sha(factors/"audit.json")))
    with np.load(original/"weights.npz") as w:
        columns = w["down_columns"]
    model = load_model(True)
    assert runtime(model) == read(original/"runtime.json")
    emit(out, out/"runtime.json", runtime(model))
    np.savez_compressed(out/"norm-weights.npz", input_norm=np.stack([array(layer.input_layernorm.weight) for layer in model.model.layers]),
        mlp_norm=np.stack([array(layer.post_attention_layernorm.weight) for layer in model.model.layers]), eps=np.array(model.config.rms_norm_eps), columns=columns)
    refs = {sub:{e["key"]:e for e in read(folder/file)["cases"]} for sub, folder, file in (
        ("F", original, "forecast-status.json"), ("T", original, "status.json"), ("L", reference, "status.json"))}
    entries, work = [], Counter()
    for number, case in enumerate(design["cases"], 1):
        saved, checks, statuses, counts = execute(model, case, columns, out)
        assert shutil.disk_usage(out).free > 512*1024**2
        case_path = out/"cases"/(case["key"]+".npz")
        np.savez_compressed(case_path, **saved)
        for arm, target_arm, folder, sub, ref in (("F", "F", original, "forecasts", refs["F"]), ("B", "S", original, "forecasts", refs["F"]),
            ("T", "T", original, "observations", refs["T"]), ("L", "L", reference, "cases", refs["L"])):
            before_path = folder/sub/(case["key"]+".npz")
            entry = ref[case["key"]]
            assert sha(before_path) == entry["arrays_sha256"]
            assert sha(before_path.with_suffix(".json")) == entry.get("record_sha256", entry.get("metadata_sha256"))
            with np.load(before_path) as before:
                for suffix in ("trajectory", "scores"):
                    checked(checks, "original_reference_"+arm+"_"+suffix, saved[arm+"_"+suffix], before[target_arm+"_"+suffix], 0.)
        emit(out, case_path.with_suffix(".json"), dict(key=case["key"], statuses=statuses, checks=checks, arrays_sha256=sha(case_path), counts=dict(counts)))
        entries.append(dict(key=case["key"], arrays_sha256=sha(case_path), metadata_sha256=sha(case_path.with_suffix(".json"))))
        work.update(counts)
        print("normalization branch/energy test", number, "/4", statuses, flush=True)
    emit(out, out/"status.json", dict(status="complete", cases=entries, counts=dict(work), norm_weights_sha256=sha(out/"norm-weights.npz")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.original.resolve(), args.reference.resolve(), args.factors.resolve(), args.out.resolve())
    except Exception as exc:
        if ACTIVE_OUTPUT is not None:
            emit(ACTIVE_OUTPUT, ACTIVE_OUTPUT/"failure-run.json", dict(error=repr(exc)))
        raise
