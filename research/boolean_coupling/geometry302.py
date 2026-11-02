"""First-source alias keys/values and defined geometric interventions.

This is an old-native-parent diagnostic, not a fresh False-only forecast.
"""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .alias301 import SOURCES as PREVIOUS
from .core225b import array, observe
from .query296 import record
from .subspace293 import statistics

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    *[
        "research/boolean_coupling/" + n
        for n in ("geometry302.py", "run302.py", "audit302.py", "preflight302.py")
    ],
    "docs/experiments/302-first-source-geometry.md",
)
ARMS = ("N", "SELF", "K", "V", "KV", "H", "S", "RH", "RK", "RV", "P", "G")
BOUNDS = dict(
    composed=1e-4,
    projection_abs=1e-3,
    projection_relative=1e-5,
    mlp=1e-4,
    state_relative=1e-5,
    head=1e-4,
    raw=1e-4,
    invariant_abs=1e-3,
    invariant_relative=1e-5,
    invariant_scores=1e-3,
    geometry=1e-9,
    random_relative=1e-6,
    contrast_relative=1e-3,
    rank_relative=1e-10,
)
CRITERIA = dict(
    norm=0.001,
    relative=0.25,
    cosine=0.95,
    material=0.02,
    margin_abs=0.1,
    margin_fraction=0.25,
    population=6,
    per_layout=1,
    per_mention_order=3,
    control_max=2,
)


def rotate(x, angle):
    """Qwen's split-half rotary convention; x[..., :d/2] pairs with last half."""
    half = x.shape[-1] // 2
    c, s = np.cos(angle), np.sin(angle)
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * c - b * s, b * c + a * s), axis=-1)


def components(case, f, inv):
    a, b = [f[x + "_F_0_recipient_k"].astype(float) for x in "AB"]
    av, bv = [f[x + "_F_0_recipient_v"].astype(float) for x in "AB"]
    idx = case["alias_changed"]
    dk, dv = b[:, idx] - a[:, idx], bv[:, idx] - av[:, idx]
    qf = f["A_F_0_query"].astype(float)[:, 0]
    groups = int(f["A_F_0_groups"])
    hidden, rank = np.empty_like(dk), []
    for g in range(dk.shape[0]):
        q = qf[g * groups : (g + 1) * groups]
        _, sv, vh = np.linalg.svd(q, full_matrices=False)
        r = int(np.sum(sv > BOUNDS["rank_relative"] * sv[0]))
        span = vh[:r]
        hidden[g] = dk[g] - (dk[g] @ span.T) @ span
        rank.append(r)
    seed = int.from_bytes(hashlib.sha256((case["key"] + "/302").encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    def matched(template, null=False):
        value = rng.normal(size=template.shape)
        if null:
            for g in range(value.shape[0]):
                q = qf[g * groups : (g + 1) * groups]
                value[g] -= value[g] @ np.linalg.pinv(q) @ q
        value *= np.linalg.norm(template, axis=-1, keepdims=True) / np.linalg.norm(
            value, axis=-1, keepdims=True
        )
        return value

    # Keep the alias key center fixed; rotate only its native A->B difference.
    angle = (case["position"] - np.array(idx))[None, :, None] * inv
    return dict(
        dk=dk,
        dv=dv,
        hidden=hidden,
        seen=dk - hidden,
        random_hidden=matched(hidden, True),
        random_key=matched(dk),
        random_value=matched(dv),
        phase_difference=rotate(dk, angle),
        ranks=np.array(rank),
        inv=inv,
    )


def inputs(case, alias, bit, arm, parents, comp):
    prefix = alias + ("_T_0_" if bit else "_F_0_")
    q = parents[prefix + "query"].astype(float).copy()
    k = parents[prefix + "recipient_k"].astype(float).copy()
    v = parents[prefix + "recipient_v"].astype(float).copy()
    idx, sign = case["alias_changed"], 1 if alias == "A" else -1
    if arm in ("K", "KV"):
        k[:, idx] += sign * comp["dk"]
    if arm in ("V", "KV"):
        v[:, idx] += sign * comp["dv"]
    for name, key in [
        ("H", "hidden"),
        ("S", "seen"),
        ("RH", "random_hidden"),
        ("RK", "random_key"),
    ]:
        if arm == name:
            k[:, idx] += sign * comp[key]
    if arm == "RV":
        v[:, idx] += sign * comp["random_value"]
    if arm == "P":
        k[:, idx] += sign * 0.5 * (comp["dk"] - comp["phase_difference"])
    if arm == "G":
        q, k = rotate(q, comp["inv"]), rotate(k, comp["inv"])
    return q.astype(np.float32), k.astype(np.float32), v.astype(np.float32)


def execute(model, case, parents, references, comp, arms, native=None):
    pos, original = case["position"], qwen.eager_attention_forward
    data, checks, counts, live = {}, [], Counter(), dict(native or {})
    label, alias, bit, arm = "", "A", False, "N"

    def attention(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        out, prob = original(
            module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
        )
        if module.layer_idx == 1:
            data[label + "_emit"] = array(values[0, :, pos])
        if module.layer_idx != 0:
            return out, prob
        prefix = alias + ("_T_0_" if bit else "_F_0_")
        for name, actual in [
            ("query", array(query[0, :, pos : pos + 1])),
            ("recipient_k", array(keys[0, :, : pos + 1])),
            ("recipient_v", array(values[0, :, : pos + 1])),
        ]:
            record(checks, label + "/parent_" + name, actual, parents[prefix + name])
        if arm == "N":
            qq, kk, vv = query, keys, values
        else:
            qs, ks, vs = inputs(case, alias, bit, arm, parents, comp)
            qq, kk, vv = query.clone(), keys.clone(), values.clone()
            qq[0, :, pos : pos + 1] = torch.tensor(qs, device=query.device)
            kk[0, :, : pos + 1] = torch.tensor(ks, device=keys.device)
            vv[0, :, : pos + 1] = torch.tensor(vs, device=values.device)
            changed, cp = original(
                module, qq, kk, vv, mask, scaling=scaling, dropout=dropout, **kwargs
            )
            counts["extra_full_QK_softmax_PV_each"] += 1
            installed, pp = out.clone(), prob.clone()
            installed[:, pos : pos + 1] = changed[:, pos : pos + 1]
            pp[:, :, pos : pos + 1] = cp[:, :, pos : pos + 1]
            record(
                checks,
                label + "/other_outputs_protected",
                array(torch.cat((installed[:, :pos], installed[:, pos + 1 :]), 1)),
                array(torch.cat((out[:, :pos], out[:, pos + 1 :]), 1)),
            )
            out, prob = installed, pp
        raw = (qq @ qwen.repeat_kv(kk, module.num_key_value_groups).transpose(2, 3)) * scaling
        counts["witness_full_QK"] += 1
        data.update(
            {
                label + "_" + n: value
                for n, value in dict(
                    query=array(qq[0, :, pos : pos + 1]),
                    key=array(kk[0, :, case["alias_changed"]]),
                    value=array(vv[0, :, case["alias_changed"]]),
                    raw=array(raw[0, :, pos : pos + 1, : pos + 1]),
                    prob=array(prob[0, :, pos : pos + 1, : pos + 1]),
                    head=array(out[0, pos : pos + 1]).transpose(1, 0, 2),
                ).items()
            }
        )
        return out, prob

    def capture(name, before=False):
        def hook(module, args, output=None):
            value = args[0] if before else output
            data[label + "_" + name] = array(value[0, pos])

        return hook

    layer = model.model.layers[0]
    handles = [
        layer.self_attn.o_proj.register_forward_hook(capture("write")),
        layer.post_attention_layernorm.register_forward_pre_hook(capture("mid", True)),
        layer.mlp.register_forward_hook(capture("mlp")),
    ]
    qwen.eager_attention_forward = attention
    try:
        for arm in arms:
            for alias in "AB":
                for bit in (False, True):
                    label = f"{arm}_{alias}_" + ("T" if bit else "F")
                    ids = case["aliases"][alias]["true_ids" if bit else "false_ids"]
                    states, _, scores = observe(model, torch.tensor([ids], device=model.device))
                    counts["full_forwards"] += 1
                    data[label + "_source"] = array(states[0][0, pos])
                    data[label + "_final"] = array(states[-1][0, pos + 1 :])
                    data[label + "_scores"] = array(scores)
                    native = alias + ("_T" if bit else "_F")
                    if arm == "N":
                        for j, state in enumerate(states):
                            record(
                                checks,
                                label + f"/native_trace{j}",
                                array(state[0, pos:]),
                                references[native + "_trace"][j],
                            )
                        record(
                            checks,
                            label + "/native_scores",
                            data[label + "_scores"],
                            references[native + "_scores"],
                        )
                    if arm == "SELF":
                        for name in ("source", "emit", "final", "scores"):
                            record(
                                checks,
                                label + "/self_" + name,
                                data[label + "_" + name],
                                live[f"N_{alias}_" + ("T" if bit else "F") + "_" + name],
                            )
                    if arm == "KV":
                        donor = ("B" if alias == "A" else "A") + ("_T" if bit else "_F")
                        record(
                            checks,
                            label + "/donor_source",
                            data[label + "_source"],
                            references[donor + "_trace"][0, 0],
                        )
                        record(
                            checks,
                            label + "/donor_emit",
                            data[label + "_emit"],
                            references[donor + "_emit1"],
                        )
                    if arm == "G":
                        for name in ("source", "final"):
                            expected = live[f"N_{alias}_" + ("T" if bit else "F") + "_" + name]
                            actual = data[label + "_" + name]
                            record(
                                checks,
                                label + "/global_" + name,
                                actual,
                                expected,
                                BOUNDS["invariant_abs"],
                            )
                            record(
                                checks,
                                label + "/global_" + name + "_relative",
                                np.linalg.norm(actual - expected)
                                / max(np.linalg.norm(expected), 1e-30),
                                0.0,
                                BOUNDS["invariant_relative"],
                            )
                        record(
                            checks,
                            label + "/global_scores",
                            data[label + "_scores"],
                            live[f"N_{alias}_" + ("T" if bit else "F") + "_scores"],
                            BOUNDS["invariant_scores"],
                        )
                    assert all(np.isfinite(v).all() for v in data.values())
                    del states
            if arm == "N":
                live.update(data)
            if not all(c["passes"] for c in checks):
                break
        return data, checks, dict(counts)
    finally:
        qwen.eager_attention_forward = original
        for h in handles:
            h.remove()


def delta(data, arm, alias, name):
    return data[f"{arm}_{alias}_T_{name}"].astype(float) - data[f"{arm}_{alias}_F_{name}"].astype(
        float
    )


def metrics(data, case):
    rows, vectors = [], []
    for arm in ("K", "V", "H", "S", "RH", "RK", "RV"):
        passed, local_pass = True, True
        entry = dict(
            key=case["key"],
            arm=arm,
            layout=case["group"]["block"],
            mention_order=case["group"]["mention_order"],
        )
        for alias in "AB":
            sign = 1 if alias == "A" else -1
            # Each receiver context has its OWN complete first-source KV effect.
            for scope in ("source", "emit", "receiving", "verdict"):
                name = "final" if scope in ("receiving", "verdict") else scope
                target = sign * (delta(data, "KV", alias, name) - delta(data, "N", alias, name))
                pred = sign * (delta(data, arm, alias, name) - delta(data, "N", alias, name))
                if scope == "verdict":
                    target, pred = target[-1], pred[-1]
                st = statistics(pred, target)
                ok = (
                    st["norm"] >= CRITERIA["norm"]
                    and st["relative"] <= CRITERIA["relative"]
                    and st["cosine"] >= CRITERIA["cosine"]
                )
                local_pass &= ok if scope in ("source", "emit") else True
                passed &= ok
                vectors.append(dict(key=case["key"], arm=arm, alias=alias, scope=scope, **st))
                entry[alias + "_" + scope + "_relative"] = st["relative"]

            def coupling(a):
                return float(np.diff(delta(data, a, alias, "scores"))[0])

            target = sign * (coupling("KV") - coupling("N"))
            pred = sign * (coupling(arm) - coupling("N"))
            bound = min(CRITERIA["margin_abs"], CRITERIA["margin_fraction"] * abs(target))
            material = abs(target) >= CRITERIA["material"]
            passed &= material and abs(pred - target) <= bound
            entry.update(
                {
                    alias + "_target_margin": target,
                    alias + "_predicted_margin": pred,
                    alias + "_margin_error": abs(pred - target),
                    alias + "_margin_bound": bound,
                    alias + "_material": material,
                }
            )
        entry.update(passes=bool(passed), local_passes=bool(local_pass))
        rows.append(entry)
    return rows, vectors
