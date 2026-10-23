"""Native source-value interchange, with a source-only causal generator."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .cache272 import digest
from .core225b import array, observe

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("U", "X", "D", "A", "Z", "R", "FZ", "T")
SOURCES = (
    "research/boolean_coupling/source273.py",
    "research/boolean_coupling/run273.py",
    "research/boolean_coupling/audit273.py",
    "docs/experiments/273-native-source-value-interchange.md",
    "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/cache272.py",
    "research/boolean_coupling/audit272.py",
    "research/boolean_coupling/run272.py",
    "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/fields267.py",
    "research/boolean_coupling/transmission265.py",
    "research/boolean_coupling/settings.json",
    "research/boolean_coupling/cases269.json",
)


def paired(cases):
    groups = {}
    for case in cases:
        group = {k: v for k, v in case["group"].items() if k != "schema"}
        key = json.dumps(group, sort_keys=True)
        groups.setdefault(key, {})[case["group"]["schema"]] = case
    assert len(groups) == 16
    assert all(set(g) == {"generic", "descriptive"} for g in groups.values())
    return [(g["generic"], g["descriptive"]) for g in groups.values()]


def deltas(own, donor, key):
    """Each vector has two KV heads. Match norms separately within each head."""
    x = own["G_V"].astype(float) - own["F_V"].astype(float)
    y = donor["G_V"].astype(float) - donor["F_V"].astype(float)
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    assert np.all(nx > 1e-8) and np.all(ny > 1e-8)
    seed = int(hashlib.sha256((key + ":273-random").encode()).hexdigest()[:16], 16)
    r = np.random.default_rng(seed).normal(size=x.shape)
    r = r / np.linalg.norm(r, axis=-1, keepdims=True) * nx
    return dict(U=x, X=y, D=y * nx / ny, A=x * ny / nx, Z=np.zeros_like(x), R=r,
                FZ=np.zeros_like(x))


class Engine:
    """Only receiving attention value reads change; source computation stays native."""

    def __init__(self, model):
        self.model = model
        self.original = qwen.eager_attention_forward
        self.case = None
        self.arm = None
        self.bank = None
        self.record = None
        self.fields = None
        self.local = None
        self.witness = None
        self.counts = dict(full_forwards=0, source_blocks=0, intervention_PV=0,
                           diagnostic_PV=0, diagnostic_output_projections=0)
        self.handles = []

    def __enter__(self):
        qwen.eager_attention_forward = self.attention
        for index, layer in enumerate(self.model.model.layers):
            def norm(module, args, output, index=index):
                self.record[index]["norm"] = array(output[0, self.case["position"]])

            self.handles.append(layer.input_layernorm.register_forward_hook(norm))
        return self

    def __exit__(self, *_):
        qwen.eager_attention_forward = self.original
        for handle in self.handles:
            handle.remove()

    def begin(self, case, arm):
        self.case, self.arm = case, arm
        self.record = [dict() for _ in range(28)]

    def attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        assert dropout == 0 and not module.training
        output, prob = self.original(module, query, key, value, mask,
                                     scaling=scaling, dropout=dropout, **kwargs)
        i, start = module.layer_idx, self.case["position"]
        self.record[i].update(K=array(key[0, :, start]), V=array(value[0, :, start]))
        if self.arm in ("F", "G"):
            return output, prob
        target = "F" if self.arm == "FZ" else "G"
        for name in ("K", "V"):
            assert np.array_equal(self.record[i][name], self.bank[target + "_" + name][i])
        if self.arm == "T":
            self.diagnostic(module, query, key, value, mask, scaling, prob, output)
            return output, prob
        changed = value.clone()
        candidate = (self.bank["F_V"][i].astype(float) + self.fields[self.arm][i]).astype(np.float32)
        changed[0, :, start] = torch.tensor(candidate, device=value.device)
        assert np.array_equal(array(changed[0, :, start]), candidate)
        proposal = (prob @ qwen.repeat_kv(changed, module.num_key_value_groups)).transpose(1, 2).contiguous()
        result = output.clone()
        result[:, start + 1:] = proposal[:, start + 1:]
        assert torch.equal(result[:, :start + 1], output[:, :start + 1])
        self.record[i].update(installed_V=candidate, prefix_source_unchanged=True,
                              own_K_sha256=digest(array(key[0, :, start])))
        self.counts["intervention_PV"] += 1
        return result, prob

    def diagnostic(self, module, query, key, value, mask, scaling, prob, output):
        """Same native input: source-value removal and donor substitution, before transport."""
        i, start = module.layer_idx, self.case["position"]
        layer = self.model.model.layers[i]
        nodes = {}
        for arm in ("T", "Z", "X"):
            if arm == "T":
                y = output
            else:
                changed = value.clone()
                candidate = (self.bank["F_V"][i].astype(float) + self.fields[arm][i]).astype(np.float32)
                changed[0, :, start] = torch.tensor(candidate, device=value.device)
                y = (prob @ qwen.repeat_kv(changed, module.num_key_value_groups)).transpose(1, 2).contiguous()
                self.counts["diagnostic_PV"] += 1
            write = layer.self_attn.o_proj(y.reshape(1, y.shape[1], -1))
            self.counts["diagnostic_output_projections"] += 1
            nodes[arm] = (array(y[0, start + 1:]), array(write[0, start + 1:]))
            self.local[arm + "_y"].append(nodes[arm][0])
            self.local[arm + "_write"].append(nodes[arm][1])
        if i in (0, 9, 27):
            assert mask is not None
            prefix = f"layer{i + 1}_"
            self.witness.update({prefix + n: v for n, v in dict(
                q=array(query[0, :, start + 1:]), k=array(key[0]), v=array(value[0]),
                mask=array(mask[0, :, start + 1:, :key.shape[-2]]),
                p=array(prob[0, :, start + 1:]), scaling=np.array(scaling),
                groups=np.array(module.num_key_value_groups)).items()})

    def forward(self, case, arm):
        self.begin(case, arm)
        ids = case["false_ids" if arm in ("F", "FZ") else "true_ids"]
        states, context, scores = observe(self.model, torch.tensor([ids], device="mps"))
        self.counts["full_forwards"] += 1
        return states, context, array(scores), self.record

    def source(self, case, false_states, context):
        """Discard every generated receiver; next block receives its observed False rows."""
        self.begin(case, "G")
        start = case["position"]
        with torch.no_grad():
            initial = self.model.model.embed_tokens(torch.tensor([case["false_ids"]], device="mps"))
            source = self.model.model.embed_tokens.weight[830].detach().clone()
            trajectory = []
            for i, layer in enumerate(self.model.model.layers):
                state = (initial if i == 0 else false_states[i - 1]).clone()
                state[0, start] = source
                out = layer(state, attention_mask=context["mask"], position_embeddings=context["rope"])
                out = out[0] if isinstance(out, tuple) else out
                source = out[0, start].detach().clone()
                trajectory.append(array(source))
                self.counts["source_blocks"] += 1
        return np.stack(trajectory), self.record


def bank_arrays(false, generated, source):
    return {**{prefix + "_" + node: np.stack([r[node] for r in records])
               for prefix, records in (("F", false), ("G", generated))
               for node in ("K", "V", "norm")}, "G_source": source}
