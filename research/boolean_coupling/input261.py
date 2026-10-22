"""Retrospective receiving MLP normalization and endogenous gate displacement."""

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .core225b import array, load_model, observe, write_json
from .mlp258 import runtime
from .silu258 import predict

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("research/boolean_coupling/input261.py", "research/boolean_coupling/audit261.py",
    "docs/experiments/261-mlp-input-origin.md", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/mlp258.py", "research/boolean_coupling/silu258.py",
    "research/boolean_coupling/curvature255.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/settings.json")
ARMS = ("F", "Z", "B", "L", "C", "R", "T")


def read(path):
    return json.loads(path.read_text())


def polynomial(base, point, anchor, gamma, eps):
    d = point-base
    scale = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+eps)
    linear = gamma*(d/scale-base*np.mean(base*d, axis=-1, keepdims=True)/scale**3)
    a = np.mean(base*d, axis=-1, keepdims=True)/scale**2
    b = np.mean(d*d, axis=-1, keepdims=True)/scale**2
    second = gamma/scale*((1.5*a*a-.5*b)*base-a*d)
    third = gamma/scale*((1.5*a*b-2.5*a**3)*base+(1.5*a*a-.5*b)*d)
    return anchor+linear, second+third


def random_correction(value, key, layer):
    result = []
    for row, vector in enumerate(value):
        seed = int(hashlib.sha256(f"{key}:{layer}:{row}:mlp-input-curvature-control".encode()).hexdigest()[:16], 16)
        v = np.random.default_rng(seed).normal(size=vector.shape)
        result.append(v/max(float(np.linalg.norm(v)), 1e-30)*float(np.linalg.norm(vector)))
    return np.stack(result)


class Instability(Exception):
    pass


def run_case(model, case, columns, partial):
    start, key = case["position"], case["key"]
    saved, bank, checks, handles, statuses, counts = {}, {}, [], [], {}, Counter()
    arm, live = "F", {}
    eps = float(model.config.rms_norm_eps)
    query_gamma = [array(layer.input_layernorm.weight).astype(float) for layer in model.model.layers]
    mlp_gamma = [array(layer.post_attention_layernorm.weight).astype(float) for layer in model.model.layers]

    def hooks(index, layer):
        i = index+1

        def query_raw(module, args):
            live[index] = dict(qx=array(args[0][0, start+1:]))
            if arm == "F":
                bank[index] = dict(qx=live[index]["qx"].astype(float))

        def query_input(module, args):
            actual = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["qz"] = actual.astype(float)
            changed = args[0]
            if arm in ("Z", "B", "L", "C", "R"):
                linear, correction = polynomial(bank[index]["qx"], live[index]["qx"].astype(float), bank[index]["qz"], query_gamma[index], eps)
                value = (linear+correction).astype(np.float32)
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(value[None], device="mps")
                checked(checks, "installed_query_input", array(changed[0, start+1:]), value, 0.)
                checked(checks, "query_prefix_source_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                counts["query_cubic_maps"] += 1
            live[index].update(qoriginal=actual, qinstalled=array(changed[0, start+1:]))
            return (changed,)

        def mlp_raw(module, args):
            live[index]["x"] = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["x"] = live[index]["x"].astype(float)

        def mlp_normalized(module, args, output):
            actual = array(output[0, start+1:])
            if arm == "F":
                bank[index]["z"] = actual.astype(float)
            changed = output
            if arm in ("Z", "L", "C", "R"):
                linear, correction = polynomial(bank[index]["x"], live[index]["x"].astype(float), bank[index]["z"], mlp_gamma[index], eps)
                value = linear if arm in ("Z", "L") else linear+(correction if arm == "C" else random_correction(correction, key, i))
                value = value.astype(np.float32)
                changed = output.clone()
                changed[:, start+1:] = torch.tensor(value[None], device="mps")
                checked(checks, "installed_MLP_normalization", array(changed[0, start+1:]), value, 0.)
                checked(checks, "MLP_norm_prefix_source_unchanged", array(changed[0, :start+1]), array(output[0, :start+1]), 0.)
                counts["MLP_normalization_maps"] += 1
            live[index].update(original_z=actual, installed_z=array(changed[0, start+1:]))
            return changed

        def gate(module, args, output):
            live[index]["gate"] = array(output[0, start+1:])

        def activation(module, args, output):
            live[index]["phi"] = array(output[0, start+1:])
            if arm == "F":
                bank[index].update(gate=live[index]["gate"].astype(float), phi=live[index]["phi"].astype(float))

        def up(module, args, output):
            live[index]["up"] = array(output[0, start+1:])

        def product(module, args):
            v = live[index]
            checked(checks, "original_product", array(args[0][0, start+1:]), v["phi"]*v["up"], 0.)
            changed, used = args[0], v["phi"]
            if arm in ("Z", "B", "L", "C", "R"):
                quadratic, mask, _, bound, _, _ = predict(bank[index]["gate"], bank[index]["phi"], v["gate"].astype(float), v["up"].astype(float), columns[index], key, i)
                used = np.where(mask, v["phi"], quadratic.astype(np.float32)).astype(np.float32)
                installed = used*v["up"]
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(installed[None], device="mps")
                checked(checks, "installed_neuron_product", array(changed[0, start+1:]), installed, 0.)
                checked(checks, "product_prefix_source_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                assert int(mask.sum()) == int(.01*mask.size)
                v.update(mask=mask, bound=bound, quadratic=quadratic)
                counts["ranked_SiLU_maps"] += 1
            v.update(used_phi=used, product=array(changed[0, start+1:]))
            return (changed,)

        def write(module, args, output):
            live[index]["write"] = array(output[0, start+1:])
            saved.update({f"{arm}_{i}_{name}":value for name, value in live[index].items()})

        def guard(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            tail = array(value[0, start:])
            if arm != "F":
                ratio = float(np.linalg.norm(tail.astype(float))/max(np.linalg.norm(saved[f"F_tail{i}"].astype(float)), .001))
                if not np.isfinite(tail).all() or ratio > 100:
                    raise Instability(dict(arm=arm, layer=i, ratio=ratio))
            saved[f"{arm}_tail{i}"] = tail
            prefix = hashlib.sha256(array(value[0, :start]).tobytes()).hexdigest()
            if arm == "F":
                bank[index]["prefix_hash"] = prefix
            assert prefix == bank[index]["prefix_hash"]

        return ((layer.input_layernorm, query_raw, True), (layer.self_attn.q_proj, query_input, True),
            (layer.post_attention_layernorm, mlp_raw, True), (layer.post_attention_layernorm, mlp_normalized, False),
            (layer.mlp.gate_proj, gate, False), (layer.mlp.act_fn, activation, False),
            (layer.mlp.up_proj, up, False), (layer.mlp.down_proj, product, True),
            (layer.mlp.down_proj, write, False), (layer, guard, False))

    for index, layer in enumerate(model.model.layers):
        for module, function, before in hooks(index, layer):
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
                    assert arm in ("L", "C", "R")
                    statuses[arm] = dict(status="guard_failed", reason=str(exc))
                counts["full_forwards_attempted"] += 1
    except Exception:
        np.savez_compressed(partial, **saved)
        raise
    finally:
        for handle in handles:
            handle.remove()
    checked(checks, "zero_trajectory", saved["Z_trajectory"], saved["F_trajectory"], 0.)
    checked(checks, "zero_scores", saved["Z_scores"], saved["F_scores"], 0.)
    for name in ("B", "L", "C", "R"):
        if statuses[name]["status"] == "complete":
            checked(checks, "source_trajectory_exact", saved[name+"_trajectory"][:, 0], saved["T_trajectory"][:, 0], 0.)
    return saved, checks, statuses, counts


def main(original, out):
    assert not out.exists() and read(original/"audit.json")["status"] == "passed"
    out.mkdir(parents=True)
    (out/"sources").mkdir()
    (out/"cases").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/Path(name).name)
    firsts = {}
    for case in read(ROOT/"research/boolean_coupling/cases258.json"):
        firsts.setdefault((case["group"]["schema"], case["group"]["reverse"]), case)
    cases = list(firsts.values())
    assert len(cases) == 4
    old_design = read(original/"design.json")
    assert sha(original/"weights.npz") == old_design["weights_sha256"] and sha(qwen.__file__) == old_design["qwen_source_sha256"]
    with np.load(original/"weights.npz") as w:
        columns = w["down_columns"]
    write_json(out/"design.json", dict(experiment=261, retrospective=True, new_literals=False,
        created_utc=datetime.now(timezone.utc).isoformat(), source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={p:sha(ROOT/p) for p in SOURCES}, cases=cases,
        input_design_sha256=sha(original/"design.json"), input_audit_sha256=sha(original/"audit.json"),
        input_forecast_status_sha256=sha(original/"forecast-status.json"), input_native_status_sha256=sha(original/"status.json")))
    model = load_model(True)
    assert runtime(model) == read(original/"runtime.json")
    write_json(out/"runtime.json", runtime(model))
    np.savez_compressed(out/"norm-weights.npz", input_norm=np.stack([array(layer.input_layernorm.weight) for layer in model.model.layers]),
        mlp_norm=np.stack([array(layer.post_attention_layernorm.weight) for layer in model.model.layers]), eps=np.array(model.config.rms_norm_eps), columns=columns)
    entries, counts = [], Counter()
    fdict = {e["key"]:e for e in read(original/"forecast-status.json")["cases"]}
    tdict = {e["key"]:e for e in read(original/"status.json")["cases"]}
    for number, case in enumerate(cases, 1):
        values, checks, statuses, work = run_case(model, case, columns, out/("partial-"+case["key"]+".npz"))
        key = case["key"]
        assert shutil.disk_usage(out).free > 512*1024**2
        path = out/"cases"/(key+".npz")
        np.savez_compressed(path, **values)
        for name, sub, ent in (("F", "forecasts", fdict[key]), ("B", "forecasts", fdict[key]), ("T", "observations", tdict[key])):
            path = original/sub/(key+".npz")
            assert sha(path) == ent["arrays_sha256"]
            with np.load(path) as before:
                old_arm = "S" if name == "B" else name
                checked(checks, "original258_trajectory_"+name, values[name+"_trajectory"], before[old_arm+"_trajectory"], 0.)
                checked(checks, "original258_scores_"+name, values[name+"_scores"], before[old_arm+"_scores"], 0.)
        write_json(path.with_suffix(".json"), dict(key=key, statuses=statuses, checks=checks, arrays_sha256=sha(path), counts=dict(work)))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json"))))
        counts.update(work)
        print("MLP input origin", number, "/4", statuses, flush=True)
    write_json(out/"status.json", dict(status="complete", cases=entries, counts=dict(counts), norm_weights_sha256=sha(out/"norm-weights.npz")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.original.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            write_json(args.out/"failure-run.json", dict(error=repr(exc)))
        raise
