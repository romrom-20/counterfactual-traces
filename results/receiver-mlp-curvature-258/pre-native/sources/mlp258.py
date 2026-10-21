"""Fresh fixed1% receiver SiLU budget with the qualified cubic query rule."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate, read_head
from .core225b import array, load_model, observe, write_json
from .reference242b import digest
from .silu258 import predict, sigmoid

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/p for p in (
    "research/boolean_coupling/mlp258.py", "research/boolean_coupling/audit258.py",
    "research/boolean_coupling/fixtures258.py", "research/boolean_coupling/silu258.py", "research/boolean_coupling/audit256.py", "research/boolean_coupling/curvature255.py", "research/boolean_coupling/cases258.json",
    "research/boolean_coupling/historical-input-hashes-258.json", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/audit242b.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/core225b.py", "research/boolean_coupling/reference242b.py",
    "research/boolean_coupling/settings.json", "docs/experiments/258-receiver-mlp-curvature.md")]
PROGRESS = {}


def read(path):
    return json.loads(path.read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_check(run):
    design = read(run/"design.json")
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(run/"sources"/Path(p).name) == h
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    return design


def runtime(model):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)


def execute_case(model, case, phase, forecast=None, columns=None):
    start, key = case["position"], case["key"]
    saved, witnesses, checks, counts, banks = {}, {}, [], Counter(), {}
    original = qwen.eager_attention_forward
    arm = "F" if phase == "forecast" else "T"
    cases = read(ROOT/"research/boolean_coupling/cases258.json")
    firsts = {}
    for c in cases:
        firsts.setdefault((c["group"]["schema"], c["group"]["reverse"]), c["key"])
    assert len(firsts) == 4
    witnessed = key in firsts.values()
    PROGRESS.update(key=key, phase=phase, data=saved, checks=checks)

    live, handles = {}, []
    gamma = [array(layer.input_layernorm.weight).astype(float) for layer in model.model.layers]
    epsilon = float(model.config.rms_norm_eps)

    def hooks(index):
        def norm_input(module, args):
            live[index] = dict(x=array(args[0][0, start+1:]))
            if arm == "F":
                banks.setdefault(index, {})["x"] = live[index]["x"]
                saved[f"F_r{index+1}"] = live[index]["x"]

        def query_input(module, args):
            normalized = array(args[0][0, start+1:])
            live[index]["original_z"] = normalized
            if arm == "F":
                banks[index]["z"] = normalized
                saved[f"F_z{index+1}"] = normalized
            changed = args[0]
            if arm in ("Z", "C", "Q", "S", "R"):
                base = banks[index]["x"].astype(float)
                displacement = live[index]["x"].astype(float)-base
                scale = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+epsilon)
                increment = gamma[index]*(displacement/scale-base*np.mean(base*displacement, axis=-1, keepdims=True)/scale**3)
                a = np.mean(base*displacement, axis=-1, keepdims=True)/scale**2
                b = np.mean(displacement*displacement, axis=-1, keepdims=True)/scale**2
                quadratic = gamma[index]/scale*((1.5*a*a-.5*b)*base-a*displacement)
                correction = quadratic+gamma[index]/scale*((1.5*a*b-2.5*a**3)*base+(1.5*a*a-.5*b)*displacement)
                counts["CPU64_quadratic_maps"] += 1
                counts["CPU64_cubic_maps"] += 1
                live[index]["correction"] = correction
                predicted = (banks[index]["z"].astype(float)+increment+correction).astype(np.float32)
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(predicted[None], device="mps")
                actual = array(changed[0, start+1:])
                saved[f"{arm}_z_hash{index+1}"] = np.frombuffer(hashlib.sha256(actual.tobytes()).digest(), dtype=np.uint8)
                checked(checks, "actual_tangent_query_input_installation", actual, predicted, 0.)
                checked(checks, "prefix_source_query_input_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                counts["CPU64_tangent_maps"] += 1
            live[index]["installed_z"] = array(changed[0, start+1:])
            return (changed,)

        def query_output(module, args, output):
            if witnessed and index+1 in (1, 14, 28):
                live[index]["q_linear"] = array(output[0, start+1:])

        def rotary_context(module, args, kwargs):
            if witnessed and index+1 in (1, 14, 28):
                live[index]["cos"], live[index]["sin"] = (array(v[0, start+1:]) for v in kwargs["position_embeddings"])
        return norm_input, query_input, query_output, rotary_context

    for index, layer in enumerate(model.model.layers):
        nh, qi, qo, rc = hooks(index)
        handles.extend((layer.input_layernorm.register_forward_pre_hook(nh), layer.self_attn.q_proj.register_forward_pre_hook(qi),
            layer.self_attn.q_proj.register_forward_hook(qo), layer.self_attn.register_forward_pre_hook(rc, with_kwargs=True)))

    def attention(module, query, keys, values, mask, scaling, dropout=0., **kwargs):
        output, prob = original(module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs)
        index = module.layer_idx
        assert dropout == 0 and not module.training
        used_k = keys
        if arm in ("F", "T"):
            banks.setdefault(index, {})["k"] = keys[:, :, start+1:].detach().clone()
            saved[arm+f"_k{index+1}"] = array(banks[index]["k"][0])
        if arm in ("Z", "O"):
            used_k = keys.clone()
            checked(checks, "source_prefix_keys_live", array(used_k[:, :, :start+1]), array(keys[:, :, :start+1]), 0.)
            raw_used = (query@qwen.repeat_kv(used_k, module.num_key_value_groups).transpose(2, 3))*scaling
            proposed_p = torch.softmax(raw_used+mask[:, :, :, :keys.shape[-2]], dim=-1, dtype=torch.float32).to(query.dtype)
            proposed_y = (proposed_p@qwen.repeat_kv(values, module.num_key_value_groups)).transpose(1, 2).contiguous()
            changed, changed_p = output.clone(), prob.clone()
            changed[:, start+1:] = proposed_y[:, start+1:]
            changed_p[:, :, start+1:] = proposed_p[:, :, start+1:]
            checked(checks, "actual_attention_output_installation", array(changed[:, start+1:]), array(proposed_y[:, start+1:]), 0.)
            checked(checks, "prefix_source_attention_unchanged", array(changed[:, :start+1]), array(output[:, :start+1]), 0.)
            output, prob = changed, changed_p
            counts["extra_full_QK_softmax_PV_each"] += 1
        if witnessed and index+1 in (1, 14, 28):
            raw = (query@qwen.repeat_kv(used_k, module.num_key_value_groups).transpose(2, 3))*scaling
            prefix = f"{arm}_{index+1}_"
            data = dict(q=array(query[0, :, start+1:]), k=array(used_k[0]), v=array(values[0]), raw=array(raw[0, :, start+1:]),
                mask=array(mask[0, :, start+1:, :keys.shape[-2]]), installed_p=array(prob[0, :, start+1:]),
                y=array(output[0, start+1:]).transpose(1, 0, 2), current_k=array(keys[0]),
                groups=np.array(module.num_key_value_groups), scaling=np.array(scaling), **live[index])
            witnesses.update({prefix+n:v for n, v in data.items()})
            counts["witness_extra_QK"] += 1
        return output, prob


    mlive = {}

    def mlp_hooks(index, layer):
        def activation(module, args, output):
            gate, phi = array(args[0][0, start+1:]), array(output[0, start+1:])
            mlive[index] = dict(gate=gate, finite_phi=phi)
            if arm == "F":
                banks[index]["gate"], banks[index]["phi"] = gate, phi
                counts["False_mlp_anchors"] += 1

        def up(module, args, output):
            mlive[index]["up"] = array(output[0, start+1:])

        def product(module, args):
            data = mlive[index]
            finite_phi, up_value = data["finite_phi"], data["up"]
            checked(checks, "original_MLP_product", array(args[0][0, start+1:]), finite_phi*up_value, 0.)
            used, changed = finite_phi, args[0]
            if arm in ("Z", "C", "Q", "S", "R"):
                base, anchor = banks[index]["gate"].astype(float), banks[index]["phi"].astype(float)
                point = data["gate"].astype(float)
                quadratic, mask, rank, bound, first, second = predict(base, anchor, point, up_value.astype(float), columns[index], key, index+1, random=arm == "R")
                delta = point-base
                smooth_remainder = point*sigmoid(point)-base*sigmoid(base)-first*delta-.5*second*delta*delta
                checked(checks, "smooth_quadratic_bound_excess", np.maximum(abs(smooth_remainder)-bound, 0.), np.zeros_like(bound), 1e-8)
                checked(checks, "public_False_SiLU_anchor", base*sigmoid(base), anchor, 1e-4)
                direction = delta/max(1., float(np.max(abs(delta))))
                step = 1e-5
                fd1 = ((base+step*direction)*sigmoid(base+step*direction)-(base-step*direction)*sigmoid(base-step*direction))/(2*step)
                checked(checks, "False_SiLU_first_FD", fd1, first*direction, 1e-5)
                step = 1e-3
                fd2 = ((base+step*direction)*sigmoid(base+step*direction)-2*base*sigmoid(base)+(base-step*direction)*sigmoid(base-step*direction))/(2*step**2)
                checked(checks, "False_SiLU_second_coefficient_FD", fd2, .5*second*direction**2, 1e-5)
                assert int(mask.sum()) == int(rank.sum()) == int(.01*mask.size)
                checked(checks, "selected_per_row_count", mask.sum(axis=-1), rank.sum(axis=-1), 0.)
                for label, value in (("mask_counts", mask.sum(axis=-1)), ("rank_counts", rank.sum(axis=-1))):
                    saved[f"{arm}_{label}{index+1}"] = value.astype(np.int32)
                saved[f"{arm}_mask_hash{index+1}"] = np.frombuffer(hashlib.sha256(mask.tobytes()).digest(), dtype=np.uint8)
                counts["CPU64_SiLU_selections"] += 1
                counts["CPU64_SiLU_FD_checks"] += 2
                if arm in ("Q", "S", "R", "Z"):
                    used = quadratic.astype(np.float32)
                    if arm != "Q":
                        used = np.where(mask, finite_phi, used).astype(np.float32)
                    changed = args[0].clone()
                    installed = used*up_value
                    changed[:, start+1:] = torch.tensor(installed[None], device="mps")
                    checked(checks, "actual_MLP_product_installation", array(changed[0, start+1:]), installed, 0.)
                data.update(base=base.astype(np.float32), anchor=anchor.astype(np.float32), columns=columns[index],
                    quadratic=quadratic, mask=mask, rank=rank, bound=bound, used_phi=used)
                if arm != "Z":
                    assert module.bias is None
                    def project(value):
                        return array(torch.nn.functional.linear(torch.tensor(value.astype(np.float32)[None], device="mps"), module.weight, None)[0])
                    tail = finite_phi.astype(float)-quadratic
                    restored = project(np.where(mask, tail, 0.)*up_value.astype(float))
                    remaining = project(np.where(mask, 0., tail)*up_value.astype(float))
                    finite_write = project(finite_phi*up_value)
                    quadratic_write = project(quadratic.astype(np.float32)*up_value)
                    gap = finite_write.astype(float)-quadratic_write.astype(float)
                    checked(checks, "signed_down_vector_closure", restored.astype(float)+remaining.astype(float), gap, 1e-4)
                    for label, value in (("restore", restored), ("remain", remaining), ("gap", gap.astype(np.float32))):
                        saved[f"{arm}_{label}{index+1}"] = value
                    data.update(finite_write=finite_write, quadratic_write=quadratic_write)
                    counts["extra_receiving_down_projections"] += 4
            checked(checks, "MLP_product_source_prefix_unchanged", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
            data.update(used_phi=used, product=array(changed[0, start+1:]))
            if witnessed and index+1 in (1, 14, 28):
                prefix = f"{arm}_{index+1}_mlp_"
                witnesses.update({prefix+n:v for n, v in data.items()})
            return (changed,)

        def write(module, args, output):
            if witnessed and index+1 in (1, 14, 28):
                witnesses[f"{arm}_{index+1}_mlp_write"] = array(output[0, start+1:])
        return activation, up, product, write

    for index, layer in enumerate(model.model.layers):
        ah, uh, ph, wh = mlp_hooks(index, layer)
        handles.extend((layer.mlp.act_fn.register_forward_hook(ah), layer.mlp.up_proj.register_forward_hook(uh),
            layer.mlp.down_proj.register_forward_pre_hook(ph), layer.mlp.down_proj.register_forward_hook(wh)))

    qwen.eager_attention_forward = attention
    try:
        prefix = None if phase == "forecast" else forecast["false_prefix_hashes"]
        with torch.no_grad():
            for arm in (("F", "Z", "C", "Q", "S", "R") if phase == "forecast" else ("T", "O")):
                ids = case["false_ids" if arm in ("F", "Z") else "true_ids"]
                states, _, scores = observe(model, torch.tensor([ids], device="mps"))
                counts["full_forwards"] += 1
                tail = np.stack([array(state[0, start:]) for state in states])
                if arm == "F":
                    prefix = [digest(array(state[0, :start])) for state in states]
                for i, state in enumerate(states):
                    assert digest(array(state[0, :start])) == prefix[i]
                    checked(checks, "causal_prefix_exact", np.array(0.), np.array(0.), 0.)
                    if arm in ("C", "Q", "S", "R"):
                        ratio = np.linalg.norm(tail[i].astype(float))/max(np.linalg.norm(saved["F_trajectory"][i].astype(float)), .001)
                        assert np.isfinite(tail[i]).all() and ratio <= 100, (arm, i+1, float(ratio))
                saved[arm+"_trajectory"], saved[arm+"_scores"] = tail, array(scores)
                assert np.isfinite(tail).all() and np.isfinite(array(scores)).all()
                if arm in ("Z", "O"):
                    target = "F" if arm == "Z" else "T"
                    checked(checks, "zero_or_oracle_full_trajectory_replay", tail, saved[target+"_trajectory"], 0.)
                    checked(checks, "zero_or_oracle_score_replay", array(scores), saved[target+"_scores"], 0.)
                if arm == "T":
                    for forecast_arm in ("C", "Q", "S", "R"):
                        checked(checks, "known_edit_source_computation", tail[:, 0], forecast[forecast_arm+"_trajectory"][:, 0], 0.)
                        if forecast_arm == "C":
                            checked(checks, "native_first_block_tail", tail[0], forecast[forecast_arm+"_trajectory"][0], 0.)
    finally:
        for handle in handles:
            handle.remove()
        qwen.eager_attention_forward = original
    return saved, witnesses, checks, counts, prefix


def forecast(run, replication):
    assert not run.exists(), "Never overwrite evidence"
    run.mkdir(parents=True)
    for name in ("forecasts", "observations", "sources", "witnesses-forecast", "witnesses-native"):
        (run/name).mkdir()
    for p in SOURCES:
        (run/"sources"/p.name).write_bytes(p.read_bytes())
    folder, checkpoint = locate(ROOT)
    weights = read_head(folder)
    down_columns = []
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for i in range(28):
            matrix = array(handle.get_tensor(f"model.layers.{i}.mlp.down_proj.weight").float()).astype(float)
            down_columns.append(np.sqrt(np.sum(matrix*matrix, axis=0)))
        weights["down_columns"] = np.stack(down_columns)
        del matrix
        weights["input_norm"] = np.stack([array(handle.get_tensor(f"model.layers.{i}.input_layernorm.weight").float()).astype(float) for i in range(28)])
    weights["input_eps"] = np.array(read(folder/"config.json")["rms_norm_eps"])
    np.savez_compressed(run/"weights.npz", **weights)
    write_json(run/"design.json", dict(experiment=258, primary="S", fresh_literals=True, dependent_templates=True, replication=replication,
        checkpoint=checkpoint, qwen_source_sha256=sha(qwen.__file__), created_utc=stamp(),
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}, weights_sha256=sha(run/"weights.npz")))
    model = load_model(True)
    write_json(run/"runtime.json", runtime(model))
    counts, entries = Counter(), []
    for number, case in enumerate(read(run/"sources/cases258.json"), 1):
        saved, witness, checks, used, prefix = execute_case(model, case, "forecast", columns=weights["down_columns"])
        counts.update(used)
        assert shutil.disk_usage(run).free > 258*1024**2
        path = run/"forecasts"/(case["key"]+".npz")
        np.savez_compressed(path, **saved)
        whash = None
        if witness:
            wp = run/"witnesses-forecast"/path.name
            np.savez_compressed(wp, **witness)
            whash = sha(wp)
        write_json(path.with_suffix(".json"), dict(key=case["key"], group=case["group"], arrays_sha256=sha(path), witness_sha256=whash,
            checks=checks, false_prefix_hashes=prefix, ordinary_true=0))
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")), witness_sha256=whash))
        PROGRESS.clear()
        write_json(run/"progress-forecast.json", dict(cases=number, counts=dict(counts)))
        print(f"receiver sparse SiLU forecast {number}/32", flush=True)
    assert counts == dict(full_forwards=192, CPU64_tangent_maps=4480, CPU64_quadratic_maps=4480, CPU64_cubic_maps=4480, CPU64_SiLU_selections=4480, CPU64_SiLU_FD_checks=8960, False_mlp_anchors=896, extra_receiving_down_projections=14336, extra_full_QK_softmax_PV_each=896, witness_extra_QK=72)
    write_json(run/"forecast-status.json", dict(status="complete", cases=entries, counts=dict(counts), ordinary_true=0))


def seal(run):
    source_check(run)
    assert not (run/"forecast-seal.json").exists() and not list((run/"observations").glob("*"))
    assert read(run/"pre-audit.json")["status"] == "passed"
    status = read(run/"forecast-status.json")
    write_json(run/"forecast-seal.json", dict(status="sealed", cases=status["cases"], ordinary_true=0, created_utc=stamp(),
        design_sha256=sha(run/"design.json"), status_sha256=sha(run/"forecast-status.json"), pre_audit_sha256=sha(run/"pre-audit.json")))


def measure(run, proof_path):
    design = source_check(run)
    assert not (run/"status.json").exists() and not list((run/"observations").glob("*"))
    proof, sealed = read(proof_path), read(run/"forecast-seal.json")
    assert proof["seal_sha256"] == sha(run/"forecast-seal.json") and proof["accepted_utc"] > sealed["created_utc"]
    assert proof["mode"] == ("local_replication" if design["replication"] else "public_remote")
    if not design["replication"]:
        assert proof["commit"] == proof["remote_head"]
        assert subprocess.check_output(["git", "ls-remote", "origin", proof["ref"]], cwd=ROOT, text=True).split()[0] == proof["commit"]
        receipt = proof["receipt_path"]
        assert subprocess.check_output(["git", "show", proof["commit"]+":"+receipt], cwd=ROOT) == (run/"forecast-seal.json").read_bytes()
    write_json(run/"measurement-proof.json", proof)
    model = load_model(True)
    assert runtime(model) == read(run/"runtime.json")
    counts, entries = Counter(), []
    for number, (case, ent) in enumerate(zip(read(run/"sources/cases258.json"), read(run/"forecast-status.json")["cases"], strict=True), 1):
        key = case["key"]
        assert key == ent["key"]
        fp = run/"forecasts"/(key+".npz")
        meta = read(fp.with_suffix(".json"))
        assert sha(fp) == ent["arrays_sha256"] and sha(fp.with_suffix(".json")) == ent["record_sha256"]
        started = stamp()
        assert started > proof["accepted_utc"]
        with np.load(fp) as data:
            payload = dict(data)
        payload["false_prefix_hashes"] = meta["false_prefix_hashes"]
        saved, witness, checks, used, _ = execute_case(model, case, "native", payload)
        counts.update(used)
        path = run/"observations"/fp.name
        np.savez_compressed(path, **saved)
        whash = None
        if witness:
            wp = run/"witnesses-native"/path.name
            np.savez_compressed(wp, **witness)
            whash = sha(wp)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path), witness_sha256=whash,
            checks=checks, native_started_utc=started, forecast_arrays_sha256=sha(fp), forecast_record_sha256=ent["record_sha256"]))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")), witness_sha256=whash))
        PROGRESS.clear()
        print(f"receiver sparse SiLU native/control {number}/32", flush=True)
    assert counts == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=24)
    write_json(run/"status.json", dict(status="complete", cases=entries, counts=dict(counts), ordinary_true=32,
        forecast_seal_sha256=sha(run/"forecast-seal.json"), proof_sha256=sha(run/"measurement-proof.json")))
    write_json(run/"seal.json", dict(status="sealed", cases=entries, status_sha256=sha(run/"status.json"), forecast_seal_sha256=sha(run/"forecast-seal.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "seal", "measure"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    try:
        run = args.out.resolve()
        if args.phase == "forecast":
            forecast(run, args.replication)
        elif args.phase == "seal":
            seal(run)
        else:
            assert args.proof
            measure(run, args.proof.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/("partial-"+args.phase+".npz"), **PROGRESS["data"])
                write_json(args.out/("partial-"+args.phase+".json"), {k:v for k, v in PROGRESS.items() if k != "data"})
            write_json(args.out/("failure-"+args.phase+".json"), dict(error=repr(exc)))
        raise
