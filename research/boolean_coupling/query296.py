"""Own-state query magnitude/direction interventions on the native finite model."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .cache272 import digest
from .core225b import array, observe
from .drive295 import SOURCES as PREVIOUS

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/query296.py",
    "research/boolean_coupling/run296.py",
    "research/boolean_coupling/audit296.py",
    "research/boolean_coupling/preflight296.py",
    "research/boolean_coupling/fixtures296.py",
    "research/boolean_coupling/cases296.json",
    "research/boolean_coupling/history296.json",
    "docs/experiments/296-query-geometry.md",
)
ARMS = ("A", "D", "B", "R")
PRIMARY = dict(
    norm=0.001, relative=0.05, cosine=0.99, margin=0.1, population=24, per_layout=6, control_max=8
)
BOUNDS = dict(map=1e-5, composed=1e-4, head=0.005, raw_diagnostic=1e-4)


def record(checks, name, a, b, bound=0.0):
    error = float(np.max(abs(np.asarray(a, float) - np.asarray(b, float))))
    checks.append(
        dict(
            name=name, error=error, bound=bound, passes=bool(np.isfinite(error) and error <= bound)
        )
    )


def random_axis(key, layer, shape):
    seed = int.from_bytes(hashlib.sha256(f"{key}/{layer}/296".encode()).digest()[:8], "big")
    return np.random.default_rng(seed).normal(size=shape)


def geometry(arm, current, false, key, layer):
    if arm == "N" or np.array_equal(current, false):
        return current.copy()
    if arm == "B":
        return false.copy()
    q, f = np.asarray(current, float), np.asarray(false, float)
    nq, nf = np.linalg.norm(q, axis=-1, keepdims=True), np.linalg.norm(f, axis=-1, keepdims=True)
    assert np.all(nq > 1e-12) and np.all(nf > 1e-12), "Degenerate query norm; preserve stop"
    if arm == "A":
        return (f * nq / nf).astype(current.dtype)
    if arm == "D":
        return (q * nf / nq).astype(current.dtype)
    assert arm == "R"
    unit = f / nf
    parallel = np.sum(q * unit, axis=-1, keepdims=True)
    tangent = q - parallel * unit
    rnd = random_axis(key, layer, q.shape)
    rnd -= np.sum(rnd * unit, axis=-1, keepdims=True) * unit
    nr = np.linalg.norm(rnd, axis=-1, keepdims=True)
    assert np.all(nr > 1e-12)
    return (parallel * unit + np.linalg.norm(tangent, axis=-1, keepdims=True) * rnd / nr).astype(
        current.dtype
    )


def execute(model, case, phase, parent=None, parent_meta=None):
    start, key = case["position"], case["key"]
    device = model.device
    data, witness, checks, counts, source = {}, {}, [], Counter(), {}
    banks = (
        {} if parent is None else {j: parent[f"F_q{j + 1}"] for j in range(len(model.model.layers))}
    )
    original = qwen.eager_attention_forward
    is_witness = case["group"]["reverse"] == case["group"]["target"] == 0
    sites = sorted({0, len(model.model.layers) // 2, len(model.model.layers) - 1})
    arm = "F" if phase == "forecast" else "T"

    def attention(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        output, prob = original(
            module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
        )
        assert dropout == 0 and not module.training
        j = module.layer_idx
        current = array(query[0, :, start + 1 :])
        used = query
        if arm == "F":
            banks[j] = current.copy()
            data[f"F_q{j + 1}"] = current.copy()
        elif arm != "T":
            mapped = geometry("N" if arm == "O" else arm[-1], current, banks[j], key, j)
            used = query.clone()
            used[0, :, start + 1 :] = torch.tensor(mapped, device=device, dtype=query.dtype)
            record(checks, f"{arm}/{j}/query_install", array(used[0, :, start + 1 :]), mapped)
            record(
                checks,
                f"{arm}/{j}/protected_queries",
                array(used[:, :, : start + 1]),
                array(query[:, :, : start + 1]),
            )
            raw = (
                used @ qwen.repeat_kv(keys, module.num_key_value_groups).transpose(2, 3)
            ) * scaling
            pp = torch.softmax(
                raw + mask[:, :, :, : keys.shape[-2]], dim=-1, dtype=torch.float32
            ).to(query.dtype)
            yy = (
                (pp @ qwen.repeat_kv(values, module.num_key_value_groups))
                .transpose(1, 2)
                .contiguous()
            )
            changed = output.clone()
            changed[:, start + 1 :] = yy[:, start + 1 :]
            cp = prob.clone()
            cp[:, :, start + 1 :] = pp[:, :, start + 1 :]
            record(
                checks,
                f"{arm}/{j}/output_install",
                array(changed[:, start + 1 :]),
                array(yy[:, start + 1 :]),
            )
            record(
                checks,
                f"{arm}/{j}/protected_output",
                array(changed[:, : start + 1]),
                array(output[:, : start + 1]),
            )
            output, prob = changed, cp
            counts["extra_full_QK_softmax_PV_each"] += 1
        if is_witness and j in sites and not arm.startswith("Z"):
            raw = (
                used @ qwen.repeat_kv(keys, module.num_key_value_groups).transpose(2, 3)
            ) * scaling
            payload = dict(
                q=array(used[0, :, start + 1 :]),
                current_q=current,
                false_q=banks[j],
                k=array(keys[0]),
                v=array(values[0]),
                mask=array(mask[0, :, start + 1 :, : keys.shape[-2]]),
                raw=array(raw[0, :, start + 1 :]),
                prob=array(prob[0, :, start + 1 :]),
                output=array(output[0, start + 1 :]).transpose(1, 0, 2),
                groups=np.array(module.num_key_value_groups),
                scaling=np.array(scaling),
            )
            witness.update({f"{arm}_{j}_" + n: v for n, v in payload.items()})
            counts["witness_extra_QK"] += 1
        return output, prob

    qwen.eager_attention_forward = attention
    prefix = None if parent_meta is None else parent_meta["false_prefix_hashes"]
    try:
        for arm in ("F", *ARMS, "ZA", "ZD", "ZR") if phase == "forecast" else ("T", "O"):
            ids = case["false_ids"] if arm == "F" or arm.startswith("Z") else case["true_ids"]
            states, _, scores = observe(model, torch.tensor([ids], device=device))
            counts["full_forwards"] += 1
            final = array(states[-1][0, start:])
            scores = array(scores)
            assert np.isfinite(final).all() and np.isfinite(scores).all()
            if arm == "F":
                prefix = [digest(array(s[0, :start])) for s in states]
            for j, s in enumerate(states):
                record(
                    checks,
                    f"{arm}/{j}/prefix_exact",
                    float(digest(array(s[0, :start])) != prefix[j]),
                    0.0,
                )
            source[arm] = [digest(array(s[0, start])) for s in states]
            data[arm + "_final"], data[arm + "_scores"] = final, scores
            if arm.startswith("Z"):
                record(checks, arm + "/zero_final", final, data["F_final"])
                record(checks, arm + "/zero_scores", scores, data["F_scores"])
            if arm == "O":
                record(checks, "O/native_final", final, data["T_final"])
                record(checks, "O/native_scores", scores, data["T_scores"])
            if arm == "T":
                for a in ARMS:
                    record(
                        checks,
                        a + "/native_source_exact",
                        float(source["T"] != parent_meta["source_hashes"][a]),
                        0.0,
                    )
            del states
    finally:
        qwen.eager_attention_forward = original
    return (
        data,
        witness,
        checks,
        dict(counts),
        dict(false_prefix_hashes=prefix, source_hashes=source),
    )
