"""Finite native source K/V communication clamps, with explicit query groups."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, load_model, observe, write_json
from .reference242b import digest

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "T", "Z", "K", "V", "B", "D", "L", "random0", "random1", "random2")
SOURCES = [ROOT/p for p in (
    "research/boolean_coupling/communication250.py", "research/boolean_coupling/audit250.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit242b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/reference242b.py", "research/boolean_coupling/settings.json",
    "research/boolean_coupling/cases242.json", "docs/experiments/250-native-source-communication.md")]
PROGRESS = {}


def read(path):
    return json.loads(path.read_text())


def direction(key, arm, layer, kind, delta):
    seed = int(hashlib.sha256(f"{key}:{arm}:{layer}:{kind}".encode()).hexdigest()[:16], 16)
    value = np.random.default_rng(seed).normal(size=delta.shape)
    return (value*np.linalg.norm(delta.astype(float))/np.linalg.norm(value)).astype(np.float32)


def main(previous, out):
    assert not out.exists(), "Never overwrite evidence"
    prior = read(previous/"design.json")
    assert prior["experiment"] == "242b" and read(previous/"audit.json")["status"] == "passed"
    _, identity = locate(ROOT)
    assert identity == prior["checkpoint"] and sha(qwen.__file__) == prior["qwen_source_sha256"]
    cases = read(ROOT/"research/boolean_coupling/cases242.json")
    firsts = {}
    for case in cases:
        firsts.setdefault(json.dumps(case["group"], sort_keys=True), case["key"])
    witness_cases = list(firsts.values())
    out.mkdir(parents=True)
    for name in ("cases", "witnesses", "sources"):
        (out/name).mkdir()
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"design.json", dict(experiment=250, retrospective=True, primary="continued_source_communication",
        checkpoint=identity, qwen_source_sha256=sha(qwen.__file__),
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        inputs={n:sha(previous/n) for n in ("design.json", "audit.json", "forecast-status.json", "status.json", "weights.npz")},
        arms=ARMS, witness_cases=witness_cases, witness_blocks=[10, 19, 28]))
    model = load_model(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)
    assert runtime == read(previous/"runtime.json")
    write_json(out/"runtime.json", runtime)
    fe = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    te = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    counts, entries, original = Counter(), [], qwen.eager_attention_forward
    for number, case in enumerate(cases, 1):
        key, start = case["key"], case["position"]
        fp, tp = (previous/n/(key+".npz") for n in ("forecasts", "observations"))
        assert sha(fp) == fe[key]["arrays_sha256"] and sha(tp) == te[key]["arrays_sha256"]
        fm = read(fp.with_suffix(".json"))
        assert sha(fp.with_suffix(".json")) == fe[key]["record_sha256"] and sha(tp.with_suffix(".json")) == te[key]["record_sha256"]
        saved, checks, witnesses, kv = {}, [], {}, {"F":{}, "T":{}}
        PROGRESS.update(key=key, data=saved, checks=checks)
        arm = "F"

        def attention(module, query, keys, values, mask, scaling, dropout=0., **kwargs):
            output, prob = original(module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs)
            index = module.layer_idx
            if index < 9:
                return output, prob
            assert dropout == 0 and not module.training
            current = dict(K=array(keys[:, :, start:start+1]), V=array(values[:, :, start:start+1]))
            if arm in ("F", "T"):
                kv[arm][index] = current
            elif arm != "H":
                for kind in ("K", "V"):
                    checked(checks, "native_True_source_"+kind, current[kind], kv["T"][index][kind], 0.)
            changed_keys, changed_values = keys, values
            if arm not in ("F", "T", "H"):
                proposed = {}
                for kind in ("K", "V"):
                    value = current[kind]
                    if arm in (kind, "B", "D", "L"):
                        value = kv["F"][index][kind]
                    elif arm.startswith("random"):
                        delta = kv["F"][index][kind].astype(float)-current[kind].astype(float)
                        value = current[kind]+direction(key, arm, index+1, kind, delta)
                        checked(checks, "random_"+kind+"_norm_match", np.linalg.norm((value-current[kind]).astype(float)), np.linalg.norm(delta), 1e-4)
                    proposed[kind] = value
                changed_keys, changed_values = keys.clone(), values.clone()
                changed_keys[:, :, start:start+1] = torch.tensor(proposed["K"], device="mps")
                changed_values[:, :, start:start+1] = torch.tensor(proposed["V"], device="mps")
                for kind, value in (("K", changed_keys), ("V", changed_values)):
                    checked(checks, "actual_"+kind+"_installation", array(value[:, :, start:start+1]), proposed[kind], 0.)
                rk = qwen.repeat_kv(changed_keys, module.num_key_value_groups)
                rv = qwen.repeat_kv(changed_values, module.num_key_value_groups)
                raw = (query@rk.transpose(2, 3))*scaling
                proposed_p = torch.softmax(raw+mask[:, :, :, :keys.shape[-2]], dim=-1, dtype=torch.float32).to(query.dtype)
                proposed_y = (proposed_p@rv).transpose(1, 2).contiguous()
                rows = slice(-1, None) if arm == "D" else slice(start+1, -1) if arm == "L" else slice(start+1, None)
                installed_y, installed_p = output.clone(), prob.clone()
                installed_y[:, rows] = proposed_y[:, rows]
                installed_p[:, :, rows] = proposed_p[:, :, rows]
                checked(checks, "prefix_source_query_outputs_unchanged", array(installed_y[:, :start+1]), array(output[:, :start+1]), 0.)
                checked(checks, "selected_query_installation", array(installed_y[:, rows]), array(proposed_y[:, rows]), 0.)
                if arm == "D":
                    checked(checks, "intermediate_query_outputs_unchanged", array(installed_y[:, start+1:-1]), array(output[:, start+1:-1]), 0.)
                if arm == "L":
                    checked(checks, "verdict_query_output_unchanged", array(installed_y[:, -1:]), array(output[:, -1:]), 0.)
                output, prob = installed_y, installed_p
                counts["extra_QK_softmax_PV_each"] += 1
            if key in witness_cases and index+1 in (10, 19, 28) and arm != "H":
                # D/L have two K/V banks; save only the query rows that use the changed bank.
                rows = slice(-1, None) if arm == "D" else slice(start+1, -1) if arm == "L" else slice(start+1, None)
                raw = (query@qwen.repeat_kv(changed_keys, module.num_key_value_groups).transpose(2, 3))*scaling
                prefix = f"{arm}_{index+1}_"
                values_to_save = dict(q=array(query[0, :, rows]), k=array(changed_keys[0]), v=array(changed_values[0]),
                    raw=array(raw[0, :, rows]), mask=array(mask[0, :, rows, :keys.shape[-2]]), p=array(prob[0, :, rows]),
                    y=array(output[0, rows]).transpose(1, 0, 2), groups=np.array(module.num_key_value_groups), scaling=np.array(scaling))
                witnesses.update({prefix+n:v for n, v in values_to_save.items()})
                counts["witness_extra_QK"] += 1
            return output, prob

        qwen.eager_attention_forward = attention
        try:
            with np.load(fp) as f, np.load(tp) as t, torch.no_grad():
                parents = {}
                for arm in ARMS:
                    ids = case["false_ids" if arm == "F" else "true_ids"]
                    states, context, scores = observe(model, torch.tensor([ids], device="mps"))
                    counts["full_forwards"] += 1
                    ref = f["false_trajectory"] if arm == "F" else t["true_trajectory"]
                    for i, state in enumerate(states):
                        assert digest(array(state[0, :start])) == fm["false_prefix_hashes"][i]
                        checked(checks, "native_source_state", array(state[0, start]), ref[i, 0], 0.)
                        if arm in ("F", "T", "Z") or i < 9:
                            checked(checks, "native_receiving_replay", array(state[0, start+1:]), ref[i, 1:], 0.)
                    if arm in ("F", "T", "Z"):
                        checked(checks, "native_score_replay", array(scores), f["false_scores"] if arm == "F" else t["true_scores"], 0.)
                    if arm in ("F", "T"):
                        parents[arm] = states[8]
                    saved[arm+"_trajectory"] = array(torch.stack(states[9:])[:, 0, start:])
                    saved[arm+"_scores"] = array(scores)
                    assert np.isfinite(saved[arm+"_trajectory"]).all() and np.isfinite(saved[arm+"_scores"]).all()
                arm = "H"
                state = parents["T"].clone()
                state[:, start:start+1] = parents["F"][:, start:start+1]
                tails = []
                for i in range(9, 28):
                    output = model.model.layers[i](state, attention_mask=context["mask"], position_embeddings=context["rope"])
                    state = output[0] if isinstance(output, tuple) else output
                    counts["H_suffix_blocks"] += 1
                    assert digest(array(state[0, :start])) == fm["false_prefix_hashes"][i]
                    tail = array(state[0, start:])
                    checked(checks, "H_source_False", tail[0], f["false_trajectory"][i, 0], 0.)
                    checked(checks, "joint_clamp_receiver_identity", tail[1:], saved["B_trajectory"][i-9, 1:], 0.)
                    tails.append(tail)
                scores = model.lm_head(model.model.norm(state)[:, -1:])[0, 0, [895, 830]]
                counts["H_finite_heads"] += 1
                checked(checks, "joint_clamp_score_identity", array(scores), saved["B_scores"], 0.)
                saved.update(H_trajectory=np.stack(tails), H_scores=array(scores))
        finally:
            qwen.eager_attention_forward = original
        assert shutil.disk_usage(out).free > 256*1024**2
        path = out/"cases"/(key+".npz")
        np.savez_compressed(path, **saved)
        witness = None
        if witnesses:
            wp = out/"witnesses"/path.name
            np.savez_compressed(wp, **witnesses)
            witness = sha(wp)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path), witness_sha256=witness,
            prior_forecast_sha256=sha(fp), prior_native_sha256=sha(tp), checks=checks))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")), witness_sha256=witness))
        PROGRESS.clear()
        write_json(out/"progress.json", dict(cases=number, counts=dict(counts)))
        print(f"native source communication {number}/32", flush=True)
    assert counts["full_forwards"] == 352 and counts["H_suffix_blocks"] == 608 and counts["H_finite_heads"] == 32
    assert counts["extra_QK_softmax_PV_each"] == 5472 and counts["witness_extra_QK"] == 132
    write_json(out/"status.json", dict(status="complete", cases=entries, counts=dict(counts)))
    write_json(out/"seal.json", dict(status="sealed", cases=entries, design_sha256=sha(out/"design.json"), status_sha256=sha(out/"status.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_242.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/"partial.npz", **PROGRESS["data"])
                write_json(args.out/"partial.json", {k:v for k, v in PROGRESS.items() if k != "data"})
            write_json(args.out/"failure.json", dict(error=repr(exc)))
        raise
