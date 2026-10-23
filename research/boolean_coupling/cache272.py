"""Causal-prefix reuse baseline. No derivatives or approximate receiving dynamics."""

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import cache_utils, masking_utils
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .core225b import array

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "research/boolean_coupling/cache272.py", "research/boolean_coupling/run272.py",
    "research/boolean_coupling/audit272.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/settings.json", "research/boolean_coupling/cases269.json",
    "research/boolean_coupling/fields267.py", "research/boolean_coupling/transmission265.py",
    "docs/experiments/272-causal-prefix-budget-baseline.md")


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def installed_sources():
    return {m.__name__: sha(inspect.getfile(m)) for m in (qwen, cache_utils, masking_utils)}


def cache_seals(cache, stop):
    return [dict(layer=i + 1, key=digest(array(layer.keys[..., :stop, :])),
                 value=digest(array(layer.values[..., :stop, :])),
                 shape=list(layer.keys[..., :stop, :].shape))
            for i, layer in enumerate(cache.layers)]


def prefix(cache, stop, config):
    # Never transfer a cached state AT or AFTER the edited token to P.
    pairs = [(layer.keys[..., :stop, :].clone(), layer.values[..., :stop, :].clone())
             for layer in cache.layers]
    result = DynamicCache(pairs, config=config)
    assert result.get_seq_length() == stop
    return result


def forward(model, ids, total, past=None, cache=False, capture_start=None):
    states, profiles, attention, prefix_states, handles = [], [], [], [], []
    old_attention = qwen.eager_attention_forward
    record = capture_start is not None
    length = ids.shape[1]
    begin = past.get_seq_length() if past is not None else 0
    assert begin + length == total

    def block(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        states.append(array(value[0, capture_start:]))
        if begin == 0:
            prefix_states.append(digest(array(value[0, :capture_start])))

    def linear(name):
        def hook(module, args):
            shape = list(args[0].shape)
            rows = int(np.prod(shape[:-1]))
            profiles.append(dict(name=name, shape=shape, rows=rows,
                                 macs=rows * module.in_features * module.out_features))
        return hook

    def observed_attention(module, query, key, value, *args, **kwargs):
        attention.append(dict(layer=module.layer_idx + 1, query_shape=list(query.shape),
            key_shape=list(key.shape), value_shape=list(value.shape),
            dense_pairs=int(query.shape[0] * query.shape[1] * query.shape[2] * key.shape[2]),
            macs=int(2 * query.shape[0] * query.shape[1] * query.shape[2] * key.shape[2] * query.shape[3]),
            prefix_key=digest(array(key[..., :begin if begin else capture_start, :])),
            prefix_value=digest(array(value[..., :begin if begin else capture_start, :]))))
        return old_attention(module, query, key, value, *args, **kwargs)

    if record:
        handles.extend(layer.register_forward_hook(block) for layer in model.model.layers)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_pre_hook(linear(name)))
        qwen.eager_attention_forward = observed_attention
    try:
        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=torch.ones((1, total), device=ids.device, dtype=ids.dtype),
                past_key_values=past, use_cache=cache, logits_to_keep=1,
                cache_position=torch.arange(begin, total, device=ids.device),
                position_ids=torch.arange(begin, total, device=ids.device)[None])
    finally:
        for handle in handles:
            handle.remove()
        qwen.eager_attention_forward = old_attention
    scores = output.logits[0, -1, [895, 830]]
    meta = dict(input_length=length, cache_before=begin, cache_after=(output.past_key_values.get_seq_length()
                if cache else None), dense=profiles, attention=attention, prefix_states=prefix_states)
    return (np.stack(states) if record else None), scores, output.past_key_values, meta


def timed(model, ids, total, false_cache=None, stop=None):
    torch.mps.synchronize()
    began = time.perf_counter()
    past = None if false_cache is None else prefix(false_cache, stop, model.config)
    _, scores, _, _ = forward(model, ids, total, past=past, cache=past is not None)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - began
    return elapsed, array(scores)


def budget(profile):
    return dict(dense_macs=sum(r["macs"] for r in profile["dense"]),
                dense_calls=len(profile["dense"]),
                non_head_rows=sum(r["rows"] for r in profile["dense"] if r["name"] != "lm_head"),
                head_macs=sum(r["macs"] for r in profile["dense"] if r["name"] == "lm_head"),
                attention_macs=sum(r["macs"] for r in profile["attention"]),
                attention_dense_pairs=sum(r["dense_pairs"] for r in profile["attention"]))


def save_json(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + "\n")
