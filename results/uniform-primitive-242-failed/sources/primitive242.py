"""Uniform False-anchored RMSNorm/softmax/SiLU interventions in public Qwen."""

from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked
from .core225b import array
from .reference242 import Reference

ARMS = ("000", "001", "010", "011", "100", "101", "110", "100_raw")


class Unstable(RuntimeError):
    def __init__(self, label, reason, value, bound, raw):
        super().__init__(f"{label}: {reason}")
        self.record = dict(location=label, reason=reason,
            value=float(value) if np.isfinite(value) else None, bound=bound)
        self.raw = raw


def maxabs(value):
    return float(np.max(abs(value)))


class Taylor:
    def __init__(self, bank):
        self.bank, self.reference, self.prepared = bank, Reference(bank), {}
        self.statistics, self.arm = [], None
        for label, e in bank.items():
            x = e["x"].astype(float)
            if e["kind"] == "norm":
                rms = np.sqrt(np.mean(x * x, -1, keepdims=True) + float(e["eps"]))
                self.prepared[label] = (x, rms)
            elif e["kind"] == "silu":
                sigma = 1 / (1 + np.exp(-x))
                self.prepared[label] = (sigma * (1 + x * (1 - sigma)), sigma * (1 - sigma) * (2 + x * (1 - 2 * sigma)))
            else:
                y = e["y"].astype(float)
                self.prepared[label] = y / y.sum(-1, keepdims=True)

    def actions(self, label, d):
        entry, p = self.bank[label], self.prepared[label]
        if entry["kind"] == "norm":
            x, rms = p
            return entry["weight"].astype(float) * (d / rms - x * np.mean(x * d, -1, keepdims=True) / rms**3), np.zeros_like(d)
        if entry["kind"] == "silu":
            slope, curve = p
            return slope * d, curve * d * d
        u = d - np.sum(p * d, -1, keepdims=True)
        y = entry["y"].astype(float)
        return y * u, y * (u * u - np.sum(p * u * u, -1, keepdims=True))

    def propose(self, label, current, raw_direction=None, mode="taylor"):
        e = self.bank[label]
        delta = current.astype(float) - e["x"].astype(float) if raw_direction is None else raw_direction
        if not np.isfinite(delta).all():
            raise Unstable(label, "nonfinite_primitive_direction", np.inf, None, current)
        # Softmax directions are differences of UNMASKED logits, so exclude the causal
        # mask's very large constant from the displacement-budget denominator.
        base_input = e.get("raw", e["x"])
        bound = 100 * max(1., maxabs(base_input))
        if maxabs(delta) > bound:
            raise Unstable(label, "primitive_displacement_cap", maxabs(delta), bound, current)
        first, second = self.actions(label, delta)
        result = e["y"].astype(float) + first + .5 * second
        if e["kind"] == "softmax" and mode == "tilt":
            p = self.prepared[label]
            u = delta - np.sum(p * delta, -1, keepdims=True)
            weights = 1 + u + .5 * u * u
            result = e["y"].astype(float) * weights / np.sum(p * weights, -1, keepdims=True)
        output_bound = min(1024., 100 * max(1., maxabs(e["y"])))
        if not np.isfinite(result).all() or maxabs(result) > output_bound:
            raise Unstable(label, "primitive_output_cap", maxabs(result), output_bound, result)
        value = result.astype(np.float32)
        self.reference.installed(label, delta, value, lambda v: self.actions(label, v), mode=mode)
        if e["kind"] == "softmax":
            self.statistics.append(dict(arm=self.arm, label=label, mode=mode,
                minimum=float(value.min()), negative_entries=int(np.count_nonzero(value < 0)),
                public_mass_error=maxabs(value.astype(float).sum(-1) - e["y"].astype(float).sum(-1))))
        if e["kind"] == "softmax" and mode == "tilt":
            assert np.min(value) >= 0
            checked(self.reference.checks, "tilt_public_mass", value.astype(float).sum(-1), e["y"].astype(float).sum(-1), 1e-5)
        return value


@contextmanager
def capture(model, start, bank):
    handles = []

    def hook(label, kind):
        def save(module, args, output):
            e = dict(kind=kind, x=array(args[0][0, start:]), y=array(output[0, start:]))
            if kind == "norm":
                e.update(weight=array(module.weight), eps=np.array(module.variance_epsilon))
            bank[label] = e
        return save

    for i, layer in enumerate(model.model.layers):
        for name, module in (("a", layer.input_layernorm), ("m", layer.post_attention_layernorm)):
            handles.append(module.register_forward_hook(hook(f"{name}{i}_norm", "norm")))
        handles.append(layer.mlp.act_fn.register_forward_hook(hook(f"g{i}", "silu")))
    original = qwen.eager_attention_forward

    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        result = original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
        repeated = qwen.repeat_kv(key, module.num_key_value_groups)
        raw = torch.matmul(query, repeated.transpose(2, 3)) * scaling
        masked = raw if attention_mask is None else raw + attention_mask[:, :, :, :repeated.shape[-2]]
        bank[f"s{module.layer_idx}"] = dict(kind="softmax", x=array(masked[0, :, start:]),
            raw=array(raw[0, :, start:]), y=array(result[1][0, :, start:]))
        return result

    qwen.eager_attention_forward = attention
    try:
        yield
    finally:
        qwen.eager_attention_forward = original
        for h in handles:
            h.remove()


@contextmanager
def intervene(model, start, taylor, arm, counts):
    handles = []

    def replacement(label):
        def change(module, args, output):
            counts["approximated_primitive_attempts"] += 1
            value = output.clone()
            proposed = taylor.propose(label, array(args[0][0, start:]))
            value[:, start:] = torch.tensor(proposed[None], device="mps")
            assert torch.equal(value[:, :start], output[:, :start])
            counts["approximated_primitive_calls"] += 1
            return value
        return change

    for i, layer in enumerate(model.model.layers):
        if arm[0] == "0":
            for name, module in (("a", layer.input_layernorm), ("m", layer.post_attention_layernorm)):
                handles.append(module.register_forward_hook(replacement(f"{name}{i}_norm")))
        if arm[2] == "0":
            handles.append(layer.mlp.act_fn.register_forward_hook(replacement(f"g{i}")))
    original = qwen.eager_attention_forward

    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        if arm[1] == "1":
            return original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
        key_states = qwen.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen.repeat_kv(value, module.num_key_value_groups)
        raw = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        masked = raw if attention_mask is None else raw + attention_mask[:, :, :, :key_states.shape[-2]]
        probability = F.softmax(masked, dim=-1, dtype=torch.float32).to(query.dtype)
        label = f"s{module.layer_idx}"
        counts["approximated_primitive_attempts"] += 1
        current = array(raw[0, :, start:])
        delta = current.astype(float) - taylor.bank[label]["raw"].astype(float)
        proposed = taylor.propose(label, array(masked[0, :, start:]), raw_direction=delta,
            mode="taylor" if arm.endswith("_raw") else "tilt")
        probability[:, :, start:] = torch.tensor(proposed[None], device="mps")
        counts["approximated_primitive_calls"] += 1
        output = torch.matmul(probability, value_states).transpose(1, 2).contiguous()
        return output, probability

    qwen.eager_attention_forward = attention
    try:
        yield
    finally:
        qwen.eager_attention_forward = original
        for h in handles:
            h.remove()
