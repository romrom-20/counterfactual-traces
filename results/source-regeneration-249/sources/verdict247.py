"""Native earlier-context / verdict-state factorial at finite attention10."""

import argparse
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
ARMS = ("F", "P", "R", "T")
EDITS = ("earlier", "self", "coupling", "all", "weight", "value", "mixed", "random0", "random1", "random2")
SOURCES = [ROOT / p for p in (
    "research/boolean_coupling/verdict247.py", "research/boolean_coupling/audit247.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit242b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/reference242b.py", "research/boolean_coupling/primitive242b.py", "research/boolean_coupling/settings.json",
    "research/boolean_coupling/cases242.json", "docs/experiments/247-native-verdict-context.md")]
PROGRESS = {}


def read(path):
    import json
    return json.loads(path.read_text())


def random_direction(key, label, term):
    import hashlib
    rng = np.random.default_rng(int(hashlib.sha256((key+label).encode()).hexdigest()[:16], 16))
    value = rng.normal(size=term.shape)
    return value * np.linalg.norm(term) / np.linalg.norm(value)


def main(previous, out):
    assert not out.exists(), "Never overwrite evidence."
    design, prior_status, native_status = (read(previous/n) for n in ("design.json", "forecast-status.json", "status.json"))
    assert design["experiment"] == "242b" and read(previous/"audit.json")["status"] == "passed"
    _, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"] and sha(Path(qwen.__file__)) == design["qwen_source_sha256"]
    out.mkdir(parents=True)
    (out/"cases").mkdir()
    (out/"sources").mkdir()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"design.json", dict(experiment=247, retrospective=True, primary="native_coupling_only", checkpoint=checkpoint,
        qwen_source_sha256=sha(Path(qwen.__file__)), source_hashes=hashes,
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        inputs={n:sha(previous/n) for n in ("design.json", "audit.json", "forecast-status.json", "status.json", "weights.npz")}))
    model = load_model(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)
    assert runtime == read(previous/"runtime.json")
    write_json(out/"runtime.json", runtime)
    counts, entries = Counter(), []
    original = qwen.eager_attention_forward
    layer = model.model.layers[9]
    for number, (case, prior, native) in enumerate(zip(read(ROOT/"research/boolean_coupling/cases242.json"), prior_status["cases"], native_status["cases"], strict=True), 1):
        key, start = case["key"], case["position"]
        assert key == prior["key"] == native["key"]
        fp, tp = previous/"forecasts"/(key+".npz"), previous/"observations"/(key+".npz")
        assert sha(fp) == prior["arrays_sha256"] and sha(tp) == native["arrays_sha256"]
        fm = read(fp.with_suffix(".json"))
        assert sha(fp.with_suffix(".json")) == prior["record_sha256"] and sha(tp.with_suffix(".json")) == native["record_sha256"]
        saved, checks, parents = {}, [], {}
        PROGRESS.update(key=key, data=saved, checks=checks)
        phase, edit = "F", None

        def attention(module, query, key_states, value, attention_mask, scaling, dropout=0., **kwargs):
            result = original(module, query, key_states, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
            if module.layer_idx != 9:
                return result
            output, probability = result
            if edit is None:
                rk = qwen.repeat_kv(key_states, module.num_key_value_groups)
                raw = torch.matmul(query[:, :, -1:], rk.transpose(2, 3)) * scaling
                counts["diagnostic_QK_products"] += 1
                for name, val in dict(q=array(query[0, :, -1:]), k=array(key_states[0]), v=array(value[0]),
                    p=array(probability[0, :, -1:]), raw=array(raw[0]), mask=array(attention_mask[0, :, -1:, :key_states.shape[-2]]),
                    y=array(output[0, -1:]).transpose(1, 0, 2), groups=np.array(module.num_key_value_groups), scaling=np.array(scaling)).items():
                    saved[phase+"_"+name] = val
                return result
            checked(checks, "native_target_product_replay", array(output[0, -1:]).transpose(1, 0, 2), saved["T_y"], 0.)
            term = saved[edit] if not edit.startswith("random") else random_direction(key, edit, saved["coupling"])
            proposed = (saved["T_y"].astype(float)-term).astype(np.float32)
            changed = output.clone()
            changed[:, -1:] = torch.tensor(proposed.transpose(1, 0, 2)[None], device="mps")
            saved[edit+"_installed"] = array(changed[0, -1:]).transpose(1, 0, 2)
            checked(checks, "actual_primitive_installation", saved[edit+"_installed"], proposed, 0.)
            checked(checks, "all_earlier_attention_rows_unchanged", array(changed[:, :-1]), array(output[:, :-1]), 0.)
            counts["primitive_installations"] += 1
            return changed, probability

        def projected_output(module, args, output):
            label = phase if edit is None else edit
            saved[label+"_write"] = array(output[0][0, -1:])
        projection_handle = layer.self_attn.register_forward_hook(projected_output)
        qwen.eager_attention_forward = attention
        try:
            with np.load(fp) as f, np.load(tp) as t, torch.no_grad():
                for phase in ("F", "T"):
                    ids = case["false_ids" if phase == "F" else "true_ids"]
                    states, context, scores = observe(model, torch.tensor([ids], device="mps"))
                    counts["ordinary_false" if phase == "F" else "ordinary_true"] += 1
                    current_context = [array(context["mask"]), *(array(v) for v in context["rope"])]
                    if phase == "F":
                        false_context = current_context
                    else:
                        for a, b in zip(false_context, current_context, strict=True):
                            checked(checks, "identical_mask_and_rotary_context", a, b, 0.)
                    ref = f["false_trajectory"] if phase == "F" else t["true_trajectory"]
                    for i, state in enumerate(states):
                        checked(checks, "native_full_tail_replay", array(state[0, start:]), ref[i], 0.)
                        assert digest(array(state[0, :start])) == fm["false_prefix_hashes"][i]
                    checked(checks, "native_score_replay", array(scores), f["false_scores"] if phase == "F" else t["true_scores"], 0.)
                    parents[phase] = states[8]
                    saved[phase+"_input"] = array(states[8][0, start:])
                    saved[phase+"_final"], saved[phase+"_scores"] = array(states[-1][0, start:]), array(scores)
                for phase in ("P", "R"):
                    parent = parents["F"].clone()
                    if phase == "P":
                        parent[:, :-1] = parents["T"][:, :-1]
                    else:
                        parent[:, -1:] = parents["T"][:, -1:]
                    checked(checks, "crossed_prior_rows", array(parent[:, :-1]), array(parents["T" if phase == "P" else "F"][:, :-1]), 0.)
                    checked(checks, "crossed_receiver_row", array(parent[:, -1:]), array(parents["F" if phase == "P" else "T"][:, -1:]), 0.)
                    layer.self_attn(layer.input_layernorm(parent), attention_mask=context["mask"], position_embeddings=context["rope"])
                    counts["local_crossed_attention_calls"] += 1
                    saved[phase+"_input"] = array(parent[0, start:])
                checked(checks, "True_query_fixed_T_R", saved["T_q"], saved["R_q"], 0.)
                checked(checks, "False_query_fixed_P_F", saved["P_q"], saved["F_q"], 0.)
                y = {a:saved[a+"_y"].astype(float) for a in ARMS}
                p = {a:saved[a+"_p"].astype(float) for a in ARMS}
                v = {a:np.repeat(saved[a+"_v"].astype(float), int(saved[a+"_groups"]), axis=0) for a in ARMS}
                saved.update(earlier=y["T"]-y["R"], self=y["T"]-y["P"], coupling=y["T"]-y["P"]-y["R"]+y["F"], all=y["T"]-y["F"])
                saved["weight"] = (p["T"]-p["R"])@v["R"]
                saved["value"] = p["R"]@(v["T"]-v["R"])
                saved["mixed"] = (p["T"]-p["R"])@(v["T"]-v["R"])
                counts["component_value_products"] += 3
                saved["rounding"] = saved["earlier"]-saved["weight"]-saved["value"]-saved["mixed"]
                checked(checks, "explicit_value_product_rounding", saved["rounding"], np.zeros_like(saved["rounding"]), 1e-4)
                for edit in EDITS:
                    states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
                    counts["primitive_intervention_forwards"] += 1
                    for i, state in enumerate(states):
                        assert digest(array(state[0, :start])) == fm["false_prefix_hashes"][i]
                        checked(checks, "causal_earlier_rows_replay", array(state[0, start:-1]), t["true_trajectory"][i, :-1], 0.)
                        if i < 9:
                            checked(checks, "unchanged_pre_target_verdict", array(state[0, -1]), t["true_trajectory"][i, -1], 0.)
                    saved[edit+"_final"], saved[edit+"_scores"] = array(states[-1][0, start:]), array(scores)
                    assert np.isfinite(saved[edit+"_final"]).all() and np.isfinite(saved[edit+"_scores"]).all()
        finally:
            projection_handle.remove()
            qwen.eager_attention_forward = original
        assert shutil.disk_usage(out).free > 256*1024**2
        path = out/"cases"/(key+".npz")
        np.savez_compressed(path, **saved)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path),
            prior_forecast_sha256=sha(fp), prior_native_sha256=sha(tp), checks=checks))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        PROGRESS.clear()
        write_json(out/"progress.json", dict(cases=number, counts=dict(counts)))
        print(f"native verdict/context {number}/32", flush=True)
    assert counts["ordinary_false"] == counts["ordinary_true"] == 32 and counts["primitive_intervention_forwards"] == 320
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
                write_json(args.out/"partial.json", {k:v for k,v in PROGRESS.items() if k != "data"})
            write_json(args.out/"failure.json", dict(error=repr(exc)))
        raise
