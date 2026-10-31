"""Query-generator operand interventions; all other own-state operations are live."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .cache272 import digest
from .core225b import array, observe
from .messages298 import SOURCES as PREVIOUS
from .query296 import record

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/generator299.py",
    "research/boolean_coupling/run299.py",
    "research/boolean_coupling/audit299.py",
    "research/boolean_coupling/fixtures299.py",
    "research/boolean_coupling/preflight299.py",
    "research/boolean_coupling/cases299.json",
    "research/boolean_coupling/history299.json",
    "docs/experiments/299-query-generators.md",
)
ARMS = ("C", "M", "B")
PRIMARY = dict(
    norm=0.001, relative=0.05, cosine=0.99, margin=0.1, population=24, per_layout=6, control_max=8
)
EFFECT = dict(norm=0.001, relative=0.25, cosine=0.95, margin=0.1, material=0.02)
BOUNDS = dict(
    normalized=1e-5,
    projection_abs=0.01,
    projection_relative=1e-5,
    composed=1e-4,
    head=0.005,
    raw_diagnostic=1e-4,
)


def sites(model):
    return sorted({1, len(model.model.layers) // 2, len(model.model.layers) - 1})


def operand(corner, current_mid, current_write, false_mid, false_write):
    """Finite Float32 sum; C is carried state, M the MOST RECENT MLP write."""
    assert corner in ("N", "C", "M", "B")
    mid = current_mid if corner in ("N", "C") else false_mid
    write = current_write if corner in ("N", "M") else false_write
    return mid + write


def execute(model, case, phase, parent=None, parent_meta=None):
    start, key, device = case["position"], case["key"], model.device
    data, witness, checks, counts, source = {}, {}, [], Counter(), {}
    banks = {} if parent is None else parent
    live, handles = {}, []
    arm = "F" if phase == "forecast" else "T"
    selected = "N"
    fixed_witness = case["group"]["reverse"] == case["group"]["target"] == 0
    original = qwen.eager_attention_forward

    def entry(j):
        def hook(module, args, kwargs):
            live[j] = dict(x=args[0], rope=kwargs["position_embeddings"])

        return hook

    def mid_hook(j):
        def hook(module, args):
            live[j]["mid"] = args[0]

        return hook

    def write_hook(j):
        def hook(module, args, output):
            live[j]["write"] = output
            if arm == "F":
                for field in ("mid", "write"):
                    n = f"F_{field}{j + 1}"
                    data[n] = array(live[j][field][0, start + 1 :])
                    banks[n] = data[n]

        return hook

    def qhook(j, layer):
        def hook(module, args):
            z = args[0]
            live[j]["normal"] = z
            if j == 0:
                return None
            current = live[j - 1]
            record(
                checks,
                f"{arm}/{j}/current_sum_exact",
                array((current["mid"] + current["write"])[0, start + 1 :]),
                array(live[j]["x"][0, start + 1 :]),
            )
            fm = torch.tensor(banks[f"F_mid{j}"], dtype=z.dtype, device=device)
            fw = torch.tensor(banks[f"F_write{j}"], dtype=z.dtype, device=device)
            cm, cw = current["mid"][0, start + 1 :], current["write"][0, start + 1 :]
            corners = (
                ("C", "M", "B")
                if fixed_witness and j in sites(model) and not arm.startswith("Z")
                else (() if selected == "N" else (selected,))
            )
            live[j]["operands"] = dict(cm=cm, cw=cw, fm=fm, fw=fw)
            live[j]["corner_normal"] = {"N": z}
            live[j]["corner_input"] = {"N": live[j]["x"]}
            for corner in corners:
                x = live[j]["x"].clone()
                x[0, start + 1 :] = operand(corner, cm, cw, fm, fw)
                live[j]["corner_input"][corner] = x
                live[j]["corner_normal"][corner] = layer.input_layernorm(x)
                counts["extra_full_RMSNorm"] += 1
            if arm == "O":
                x = current["mid"] + current["write"]
                zz = layer.input_layernorm(x)
                counts["extra_full_RMSNorm"] += 1
                record(checks, f"{arm}/{j}/both_operand_replay", array(zz), array(z))
            elif selected != "N":
                zz = live[j]["corner_normal"][selected]
            else:
                return None
            changed = z.clone()
            changed[:, start + 1 :] = zz[:, start + 1 :]
            record(
                checks,
                f"{arm}/{j}/query_input_install",
                array(changed[:, start + 1 :]),
                array(zz[:, start + 1 :]),
            )
            record(
                checks,
                f"{arm}/{j}/protected_query_input",
                array(changed[:, : start + 1]),
                array(z[:, : start + 1]),
            )
            return (changed,)

        return hook

    def kvhook(j, name):
        def hook(module, args):
            record(
                checks,
                f"{arm}/{j}/{name}_normal_unchanged",
                array(args[0]),
                array(live[j]["normal"]),
            )

        return hook

    def attention(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        output, prob = original(
            module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
        )
        assert dropout == 0 and not module.training
        j = module.layer_idx
        if arm == "F":
            data[f"F_q{j + 1}"] = array(query[0, :, start + 1 :])
            banks[f"F_q{j + 1}"] = data[f"F_q{j + 1}"]
        if j == 0 or selected == "B":
            record(
                checks,
                f"{arm}/{j}/false_query_exact",
                array(query[0, :, start + 1 :]),
                banks[f"F_q{j + 1}"],
            )
        if not (fixed_witness and j in sites(model) and not arm.startswith("Z")):
            return output, prob
        layer, state = model.model.layers[j], live[j]
        kk = qwen.repeat_kv(keys, module.num_key_value_groups)
        vv = qwen.repeat_kv(values, module.num_key_value_groups)
        cos, sin = state["rope"]
        payload = {n: array(v) for n, v in state["operands"].items()}
        payload.update(
            k=array(keys[0]),
            v=array(values[0]),
            cos=array(cos[0, start + 1 :]),
            sin=array(sin[0, start + 1 :]),
            scaling=np.array(scaling),
            groups=np.array(module.num_key_value_groups),
            mask=array(mask[0, :, start + 1 :, : keys.shape[-2]]),
        )
        for corner in ("N", "C", "M", "B"):
            z = state["corner_normal"][corner]
            if corner == selected:
                q, p, y = query, prob, output.transpose(1, 2)
            else:
                q = functional.linear(z, layer.self_attn.q_proj.weight, layer.self_attn.q_proj.bias)
                q = q.view(1, z.shape[1], -1, module.head_dim).transpose(1, 2)
                q, _ = qwen.apply_rotary_pos_emb(q, keys, cos, sin)
                raw = (q @ kk.transpose(2, 3)) * scaling
                p = torch.softmax(
                    raw + mask[:, :, :, : keys.shape[-2]], -1, dtype=torch.float32
                ).to(q.dtype)
                y = p @ vv
                counts["witness_extra_full_query_projection"] += 1
                counts["witness_extra_full_QK"] += 1
                counts["witness_extra_full_softmax"] += 1
                counts["witness_extra_full_PV"] += 1
            payload.update(
                {
                    corner + "_input": array(state["corner_input"][corner][0, start + 1 :]),
                    corner + "_normal": array(z[0, start + 1 :]),
                    corner + "_q": array(q[0, :, start + 1 :]),
                    corner + "_prob": array(p[0, :, start + 1 :]),
                    corner + "_output": array(y[0, :, start + 1 :]),
                }
            )
        payload["selected_q"] = array(query[0, :, start + 1 :])
        payload["selected_output"] = array(output[0, start + 1 :]).transpose(1, 0, 2)
        payload["selected_prob"] = array(prob[0, :, start + 1 :])
        witness.update({f"{arm}_{j}_" + n: v for n, v in payload.items()})
        return output, prob

    for j, layer in enumerate(model.model.layers):
        handles += [
            layer.register_forward_pre_hook(entry(j), with_kwargs=True),
            layer.post_attention_layernorm.register_forward_pre_hook(mid_hook(j)),
            layer.mlp.register_forward_hook(write_hook(j)),
            layer.self_attn.q_proj.register_forward_pre_hook(qhook(j, layer)),
            layer.self_attn.k_proj.register_forward_pre_hook(kvhook(j, "key")),
            layer.self_attn.v_proj.register_forward_pre_hook(kvhook(j, "value")),
        ]
    qwen.eager_attention_forward = attention
    prefix = None if parent_meta is None else parent_meta["false_prefix_hashes"]
    try:
        for arm in ("F", *ARMS, "ZC", "ZM", "ZB") if phase == "forecast" else ("T", "O"):
            selected = "N" if arm in ("F", "T", "O") else arm[-1]
            ids = case["false_ids"] if arm == "F" or arm.startswith("Z") else case["true_ids"]
            states, _, scores = observe(model, torch.tensor([ids], device=device))
            counts["full_forwards"] += 1
            final, scores = array(states[-1][0, start:]), array(scores)
            trace = np.stack([array(s[0, start + 1 :]) for s in states])
            assert all(np.isfinite(x).all() for x in (final, scores, trace)), key
            if arm == "F":
                prefix = [digest(array(s[0, :start])) for s in states]
            for j, state in enumerate(states):
                record(
                    checks,
                    f"{arm}/{j}/prefix_exact",
                    float(digest(array(state[0, :start])) != prefix[j]),
                    0.0,
                )
            source[arm] = [digest(array(s[0, start])) for s in states]
            data[arm + "_final"], data[arm + "_scores"] = final, scores
            if not arm.startswith("Z"):
                data[arm + "_trace"] = trace
            if arm.startswith("Z"):
                for suffix, actual in (("final", final), ("scores", scores), ("trace", trace)):
                    record(checks, arm + "/zero_" + suffix, actual, data["F_" + suffix])
            if arm == "O":
                for suffix, actual in (("final", final), ("scores", scores), ("trace", trace)):
                    record(checks, "O/native_" + suffix, actual, data["T_" + suffix])
            if arm == "T":
                for a in ARMS:
                    record(
                        checks,
                        a + "/native_source_exact",
                        float(source["T"] != parent_meta["source_hashes"][a]),
                        0.0,
                    )
            live.clear()
            del states
    finally:
        qwen.eager_attention_forward = original
        for handle in handles:
            handle.remove()
    return (
        data,
        witness,
        checks,
        dict(counts),
        dict(false_prefix_hashes=prefix, source_hashes=source),
    )
