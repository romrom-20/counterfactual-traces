"""Source-only attention probability/self-value factorial with finite MLPs."""

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .core225b import array, observe
from .source273 import ROOT
from .source273 import SOURCES as SOURCES273

FACTORS = ("N", "W", "V", "B")
VALUE_ARMS = ("U", "W", "V", "B", "Z", "FZ", "T")
SOURCES = (*SOURCES273, "research/boolean_coupling/reanchor274.py",
           "research/boolean_coupling/reading275.py", "research/boolean_coupling/run275.py",
           "research/boolean_coupling/audit275.py", "docs/experiments/275-source-attention-reading.md")


class SourceReading:
    """Only the source QUERY row's attention result is changed, never receiver reads."""

    def __init__(self, model):
        self.model = model
        self.original = qwen.eager_attention_forward
        self.case = self.arm = self.false = self.records = None
        self.handles = []
        self.counts = dict(full_forwards=0, source_intervention_PV=0,
                           native_corner_PV=0, native_corner_output_projections=0)

    def __enter__(self):
        qwen.eager_attention_forward = self.attention
        for i, layer in enumerate(self.model.model.layers):
            for module, name, before in ((layer.input_layernorm, "before", True),
                                         (layer.input_layernorm, "norm", False),
                                         (layer.self_attn.o_proj, "attwrite", False),
                                         (layer.post_attention_layernorm, "mid", True),
                                         (layer.mlp, "mlpwrite", False)):
                def hook(module, args, output=None, i=i, name=name, before=before):
                    value = args[0] if before else output
                    self.records[i][name] = array(value[0, self.case["position"]])

                register = module.register_forward_pre_hook if before else module.register_forward_hook
                self.handles.append(register(hook))
        return self

    def __exit__(self, *_):
        qwen.eager_attention_forward = self.original
        for h in self.handles:
            h.remove()

    def attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        assert dropout == 0 and not module.training
        output, p = self.original(module, query, key, value, mask, scaling=scaling,
                                  dropout=dropout, **kwargs)
        i, start = module.layer_idx, self.case["position"]
        rec = self.records[i]
        assert mask is not None and torch.isfinite(query).all() and torch.isfinite(value).all()
        rec.update(q=array(query[0, :, start]), k=array(key[0, :, start]),
                   v=array(value[0, :, start]), p=array(p[0, :, start]),
                   mask=array(mask[0, 0, start, :key.shape[-2]]), scaling=np.array(scaling))
        if self.arm == "F":
            rec.update(full_k=array(key[0]), full_v=array(value[0]),
                       used_p=rec["p"], used_self_v=rec["v"], y=array(output[0, start]))
            return output, p
        f = self.false[i]
        assert np.array_equal(array(key[0, :, :start]), f["full_k"][:, :start])
        assert np.array_equal(array(value[0, :, :start]), f["full_v"][:, :start])
        assert np.array_equal(rec["mask"], f["mask"])
        assert np.count_nonzero(rec["p"][:, start + 1:]) == 0

        def proposal(arm):
            weights, values = p.clone(), value.clone()
            if arm in ("W", "B"):
                weights[0, :, start] = torch.tensor(f["p"], device="mps")
            if arm in ("V", "B"):
                values[0, :, start] = torch.tensor(f["v"], device="mps")
            y = (weights @ qwen.repeat_kv(values, module.num_key_value_groups)).transpose(1, 2).contiguous()
            return y, weights, values

        if self.arm == "N":
            for arm in ("W", "V", "B"):
                y, _, _ = proposal(arm)
                write = self.model.model.layers[i].self_attn.o_proj(y.reshape(1, y.shape[1], -1))
                rec["corner_" + arm + "_y"] = array(y[0, start])
                rec["corner_" + arm + "_write"] = array(write[0, start])
                self.counts["native_corner_PV"] += 1
                self.counts["native_corner_output_projections"] += 1
            assert np.array_equal(rec["corner_B_y"], f["y"])
            assert np.array_equal(rec["corner_B_write"], f["attwrite"])
            rec.update(used_p=rec["p"], used_self_v=rec["v"], y=array(output[0, start]))
            return output, p
        y, weights, values = proposal("B" if self.arm == "FB" else self.arm)
        result = output.clone()
        result[:, start] = y[:, start]
        assert torch.equal(result[:, :start], output[:, :start])
        assert torch.equal(result[:, start + 1:], output[:, start + 1:])
        rec.update(used_p=array(weights[0, :, start]), used_self_v=array(values[0, :, start]),
                   y=array(result[0, start]), only_source_query_changed=True)
        self.counts["source_intervention_PV"] += 1
        return result, p

    def forward(self, case, arm):
        self.case, self.arm, self.records = case, arm, [dict() for _ in range(28)]
        ids = case["false_ids" if arm in ("F", "FB") else "true_ids"]
        states, context, scores = observe(self.model, torch.tensor([ids], device="mps"))
        self.counts["full_forwards"] += 1
        if arm == "F":
            self.false = self.records
        return states, context, array(scores), self.records


def packed(records, prefix):
    names = set.intersection(*(set(r) for r in records)) - {"only_source_query_changed"}
    return {prefix + "_" + n: np.stack([r[n] for r in records]) for n in names}


__all__ = ["ROOT", "FACTORS", "VALUE_ARMS", "SOURCES", "SourceReading", "packed"]
