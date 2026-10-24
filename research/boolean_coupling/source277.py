"""Same local S rule as276, with causal first-attention operands and no native MLP target."""

import hashlib

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .core225b import array, observe
from .firstmlp276 import SOURCES as SOURCES276

ARMS = ("C", "A", "L", "S", "G0", "G1", "G2", "Z", "FZ")
SOURCES = (*SOURCES276, "research/boolean_coupling/source277.py",
           "research/boolean_coupling/run277.py", "research/boolean_coupling/audit277.py",
           "research/boolean_coupling/fixtures277.py", "research/boolean_coupling/cases277.json",
           "research/boolean_coupling/historical-input-hashes-277.json",
           "docs/experiments/277-sealed-first-source-mlp.md")


class First:
    """Record first-source operands; optional P/V corners execute before any MLP."""

    def __init__(self, model, case, false=None):
        self.model, self.case, self.false = model, case, false
        self.data, self.handles = {}, []
        self.original = qwen.eager_attention_forward

    def __enter__(self):
        start, layer = self.case["position"], self.model.model.layers[0]
        for module, name, before in (
            (layer.input_layernorm, "before", True), (layer.input_layernorm, "norm", False),
            (layer.self_attn.o_proj, "attwrite", False), (layer.post_attention_layernorm, "mid", True),
            (layer.post_attention_layernorm, "z", False), (layer.mlp.gate_proj, "gate", False),
            (layer.mlp.up_proj, "up", False), (layer.mlp.act_fn, "phi", False),
            (layer.mlp.down_proj, "product", True), (layer.mlp.down_proj, "write", False)):
            def hook(module, args, output=None, name=name, before=before):
                self.data[name] = array((args[0] if before else output)[0, start])
            register = module.register_forward_pre_hook if before else module.register_forward_hook
            self.handles.append(register(hook))

        def attention(module, query, key, value, mask, scaling, dropout=0., **kwargs):
            output, p = self.original(module, query, key, value, mask, scaling=scaling, dropout=dropout, **kwargs)
            if module.layer_idx != 0:
                return output, p
            assert dropout == 0 and not module.training
            self.data.update(q=array(query[0, :, start]), k=array(key[0, :, start]),
                v=array(value[0, :, start]), full_k=array(key[0]), full_v=array(value[0]),
                p=array(p[0, :, start]), mask=array(mask[0, 0, start]),
                scaling=np.array(scaling), y=array(output[0, start]))
            if self.false is not None:
                assert np.array_equal(self.data["full_k"][:, :start], self.false["full_k"][:, :start])
                assert np.array_equal(self.data["full_v"][:, :start], self.false["full_v"][:, :start])
                weights = p.clone()
                weights[0, :, start] = torch.tensor(self.false["p"], device="mps")
                for corner in ("W", "B"):
                    values = value.clone()
                    if corner == "B":
                        values[0, :, start] = torch.tensor(self.false["v"], device="mps")
                    y = (weights @ qwen.repeat_kv(values, 6)).transpose(1, 2).contiguous()
                    write = layer.self_attn.o_proj(y.reshape(1, y.shape[1], -1))
                    self.data[corner + "_y"] = array(y[0, start])
                    self.data[corner + "_write"] = array(write[0, start])
            return output, p

        qwen.eager_attention_forward = attention
        return self

    def __exit__(self, *_):
        qwen.eager_attention_forward = self.original
        for handle in self.handles:
            handle.remove()


def prepare(model, case, context, false):
    """Execute first attention only from known embeddings; never execute True MLP1."""
    with torch.no_grad():
        full = model.model.embed_tokens(torch.tensor([case["false_ids"]], device="mps"))
        full[0, case["position"]] = model.model.embed_tokens.weight[830]
        layer = model.model.layers[0]
        with First(model, case, false) as capture:
            layer.self_attn(layer.input_layernorm(full), attention_mask=context["mask"],
                            position_embeddings=context["rope"])
        data = capture.data
    assert not set(data) & {"mid", "z", "gate", "up", "phi", "product", "write"}
    data["mid"] = (data["before"] + data["attwrite"]).astype(np.float32)
    assert np.array_equal(data["B_y"], false["y"])
    assert np.array_equal(data["B_write"], false["attwrite"])
    e = data["before"].astype(float) - false["before"].astype(float)
    value = data["W_write"].astype(float) - data["B_write"].astype(float)
    carrier = (false["mid"].astype(float) + e + value).astype(np.float32)
    r = data["mid"].astype(float) - carrier.astype(float)
    points = dict(F=false["mid"], C=carrier, R=(false["mid"].astype(float) + r).astype(np.float32), T=data["mid"])
    return data, points, dict(embedding=e, self_value=value,
                            carrier=carrier.astype(float) - false["mid"].astype(float), read=r)


def proposals(nodes, derivatives, key):
    mf, mc, mr = (nodes[n]["write"].astype(float) for n in ("F", "C", "R"))
    low, shifted = (mc + derivatives[n]["write"] for n in ("F", "C"))
    gain = shifted - low
    writes = dict(C=mc, A=mc + mr - mf, L=low, S=shifted, Z=mf, FZ=mf)
    random = {}
    for number in range(3):
        arm = "G" + str(number)
        seed = int(hashlib.sha256((key + f":276gain:{number}").encode()).hexdigest()[:16], 16)
        draw = np.random.default_rng(seed).normal(size=1536)
        draw *= np.linalg.norm(gain) / np.linalg.norm(draw)
        random[arm], writes[arm] = draw, low + draw
    return writes, gain, random


class Forecast:
    """Only source MLP1 is approximated; remaining computation stays finite."""

    def __init__(self, model, case, points, nodes, writes):
        self.model, self.case = model, case
        self.points, self.nodes, self.writes = points, nodes, writes
        self.arm, self.record, self.handles = None, None, []

    def __enter__(self):
        start, layer = self.case["position"], self.model.model.layers[0]

        def before(module, args):
            self.record["native_mid"] = array(args[0][0, start])
            native = "F" if self.arm == "FZ" else "T"
            branch = "F" if self.arm in ("Z", "FZ") else "C"
            assert np.array_equal(self.record["native_mid"], self.points[native])
            changed = args[0].clone()
            changed[0, start] = torch.tensor(self.points[branch], device="mps")
            assert torch.equal(changed[:, :start], args[0][:, :start])
            assert torch.equal(changed[:, start + 1:], args[0][:, start + 1:])
            self.record["used_point"] = array(changed[0, start])
            return (changed,)

        def write(module, args, output):
            branch = "F" if self.arm in ("Z", "FZ") else "C"
            self.record["raw_write"] = array(output[0, start])
            assert np.array_equal(self.record["raw_write"], self.nodes[branch]["write"])
            changed = output.clone()
            candidate = self.writes[self.arm].astype(np.float32)
            changed[0, start] = torch.tensor(candidate, device="mps")
            assert torch.equal(changed[:, :start], output[:, :start])
            assert torch.equal(changed[:, start + 1:], output[:, start + 1:])
            self.record["installed_write"] = array(changed[0, start])
            assert np.array_equal(self.record["installed_write"], candidate)
            return changed

        self.handles = [layer.post_attention_layernorm.register_forward_pre_hook(before),
                        layer.mlp.register_forward_hook(write)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def forward(self, arm):
        assert arm in ARMS
        self.arm, self.record = arm, {}
        ids = self.case["false_ids" if arm == "FZ" else "true_ids"]
        states, _, scores = observe(self.model, torch.tensor([ids], device="mps"))
        return states, array(scores), self.record
