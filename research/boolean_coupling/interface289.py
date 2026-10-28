"""Finite attention-interface interventions inside 288's receiving branch."""

import itertools
from contextlib import contextmanager

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .composition280 import digest
from .core225b import array
from .routing288 import SOURCES as PREVIOUS
from .routing288 import Routing

ARMS = tuple("".join(x) for x in itertools.product("01", repeat=3))
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/interface289.py",
    "research/boolean_coupling/probes289.py",
    "research/boolean_coupling/run289.py",
    "research/boolean_coupling/audit289.py",
    "research/boolean_coupling/preflight289.py",
    "docs/experiments/289-attention-interface.md",
)
BOUNDS = dict(
    primal=1e-4,
    boundary_gradient=1e-4,
    primitive=1e-4,
    cached_finite_coordinate=0.002,
    cached_finite_score=0.001,
    CPU64_head=0.005,
    AD_FD_absolute=0.001,
    AD_FD_relative=0.02,
    halving_absolute=0.002,
    halving_relative=0.03,
)
PRIMARY = dict(
    arm="111",
    target="remaining gE-g101 and finite EE-E101",
    norm_min=0.001,
    relative_max=0.25,
    cosine_min=0.95,
    finite_absolute=0.005,
    finite_relative=0.25,
    material_min=0.02,
    population_min=24,
    per_layout_min=6,
)


class Interface(Routing):
    def __init__(self, model, case, write):
        self.native_attention = qwen.eager_attention_forward
        self.collect_anchors, self.attention, self.norm_outputs = True, [], []
        super().__init__(model, case, write)
        self.collect_anchors = False
        assert len(self.attention) == len(self.norm_outputs) == 2
        self.counts.update(
            interface_forwards=0,
            interface_blocks=0,
            interface_heads=0,
            interface_backwards=0,
            blockwise_blocks=0,
            blockwise_VJPs=0,
            blockwise_head_VJPs=0,
            encoding_norm_calls=0,
            reference_forwards=0,
            reference_backwards=0,
            primitive_norm_VJPs=0,
            primitive_attention_VJPs=0,
        )
        self.record_paths = False

    def native(self, base):
        if not self.collect_anchors:
            return super().native(base)
        captured, norms, handles = {}, {}, []

        def collect(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            result = self.native_attention(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
            )
            k, v = (qwen.repeat_kv(x, module.num_key_value_groups) for x in (key, value))
            raw = query @ k.transpose(2, 3) * scaling
            masked = raw if attention_mask is None else raw + attention_mask[:, :, :, : k.shape[-2]]
            captured[module.layer_idx - 1] = {
                n: x.detach().clone()
                for n, x in dict(
                    q=query,
                    k=k,
                    v=v,
                    raw=raw,
                    masked=masked,
                    p=result[1],
                    product=result[0].transpose(1, 2),
                ).items()
            }
            captured[module.layer_idx - 1]["scaling"] = float(scaling)
            return result

        for j, layer in enumerate(self.model.model.layers[1:]):

            def save(module, args, output, j=j):
                norms[j] = output[:, :1].detach().clone()

            handles.append(layer.input_layernorm.register_forward_hook(save))
        previous = qwen.eager_attention_forward
        qwen.eager_attention_forward = collect
        try:
            result = super().native(base)
        finally:
            qwen.eager_attention_forward = previous
            for h in handles:
                h.remove()
        assert set(captured) == set(norms) == set(range(len(result[0]) - 1))
        self.attention.append(captured)
        self.norm_outputs.append(norms)
        return result

    @contextmanager
    def receiving(self, j, arm):
        encoding, routing, mixing = map(int, arm)
        layer = self.model.model.layers[j + 1]
        f, e = self.attention[0][j], self.attention[1][j]

        def norm(module, args, output):
            if not encoding:
                return output
            source_offset = args[0][:, :1] - self.tapes[0][j][:, :1]
            incoming = torch.cat(
                (self.tapes[1][j][:, :1] + source_offset, self.tapes[1][j][:, 1:]), dim=1
            )
            # Full cached-tail shape and actual native norm; bypass this hook only.
            normalized = type(module).forward(module, incoming)[:, :1]
            replacement = self.norm_outputs[0][j] + (normalized - self.norm_outputs[1][j])
            self.counts["encoding_norm_calls"] += 1
            return torch.cat((replacement, output[:, 1:]), dim=1)

        def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            assert module.layer_idx == j + 1 and dropout == 0.0 and not module.training
            if not routing and not mixing:
                return self.native_attention(
                    module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
                )
            k, v = (qwen.repeat_kv(x, module.num_key_value_groups) for x in (key, value))
            if routing:
                raw = f["raw"] + (
                    ((e["q"] + (query - f["q"])) @ (e["k"] + (k - f["k"])).transpose(2, 3))
                    * scaling
                    - e["raw"]
                )
                scores = e["masked"] + (raw - f["raw"])
                p = f["p"] + (
                    torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                    - e["p"]
                )
            else:
                raw = query @ k.transpose(2, 3) * scaling
                scores = (
                    raw if attention_mask is None else raw + attention_mask[:, :, :, : k.shape[-2]]
                )
                p = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            if mixing:
                product = f["product"] + (
                    (e["p"] + (p - f["p"])) @ (e["v"] + (v - f["v"])) - e["product"]
                )
            else:
                product = p @ v
            return product.transpose(1, 2).contiguous(), p

        previous = qwen.eager_attention_forward
        handle = layer.input_layernorm.register_forward_hook(norm)
        qwen.eager_attention_forward = attention
        try:
            yield
        finally:
            qwen.eager_attention_forward = previous
            handle.remove()

    def step(self, j, value, ca, cr, arm):
        source = self.block(j + 1, self.tapes[1][j] + value, ca)
        with self.receiving(j, arm):
            receiving = self.block(j + 1, self.tapes[0][j] + value, cr)
        return torch.cat(
            (
                source[:, :1] - self.tapes[1][j + 1][:, :1],
                receiving[:, 1:] - self.tapes[0][j + 1][:, 1:],
            ),
            dim=1,
        )

    def _modified(self, arm, point, trace=False):
        ca, cr = self.fresh(), self.fresh()
        value = torch.cat((point[None, None], torch.zeros_like(self.false_tail[:, 1:])), dim=1)
        if trace:
            self.trace = {}
        paths = []
        for j in range(len(self.tapes[0]) - 1):
            if trace:

                def record(g, j=j):
                    self.trace[j] = array(g[0, 0])

                value.register_hook(record)
            value = self.step(j, value, ca, cr, arm)
            if self.record_paths:
                paths.append(digest(array(value[0, 0])))
            self.counts["interface_blocks"] += 2
        self.last_paths = paths
        head_state = self.tapes[1][-1] + value
        scores = self.head(head_state)
        self.counts["interface_forwards"] += 1
        self.counts["interface_heads"] += 1
        return value, head_state, scores

    def finite(self, arm, displacement):
        point = torch.tensor(displacement, device=self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            offset, state, scores = self._modified(arm, point)
        return dict(
            offset=array(offset[0]),
            head_state=array(state[0]),
            raw_scores=array(scores),
            lifted_scores=array(scores).astype(float),
        )

    def gradient(self, arm):
        result = super().gradient(arm)
        self.counts["tape_backwards"] -= 1
        self.counts["interface_backwards"] += 1
        return result

    def blockwise(self, arm):
        ca, cr = self.fresh(), self.fresh()
        with torch.enable_grad():
            point = self.tapes[1][-1].clone().requires_grad_(True)
            scores = self.head(point)
            adjoint = torch.autograd.grad(scores[1] - scores[0], point)[0].detach()
        self.counts["blockwise_head_VJPs"] += 1
        trace = []
        for j in reversed(range(len(self.tapes[0]) - 1)):
            with torch.enable_grad():
                point = torch.zeros_like(self.tapes[0][j]).requires_grad_(True)
                output = self.step(j, point, ca, cr, arm)
                adjoint = torch.autograd.grad(output, point, grad_outputs=adjoint)[0].detach()
            self.counts["blockwise_blocks"] += 2
            self.counts["blockwise_VJPs"] += 1
            trace.append(array(adjoint[0, 0]))
        return dict(gradient=array(adjoint[0, 0]), source_trace=np.stack(trace[::-1]))

    def reference_gradient(self):
        with torch.enable_grad():
            point = torch.zeros_like(self.false_tail[0, 0]).requires_grad_(True)
            offset, state, scores = Routing._modified(self, "111", point)
            grad = torch.autograd.grad(scores[1] - scores[0], point)[0]
        self.counts["reference_forwards"] += 1
        self.counts["reference_backwards"] += 1
        return array(grad), array(scores), array(offset[0]), array(state[0])
