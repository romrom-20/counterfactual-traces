"""False-state polynomial source activation; finite norm/product, no full True SiLU."""

import hashlib

import numpy as np
import torch

from .core225b import array, observe
from .primitive278 import SOURCES as SOURCES278
from .source277 import SOURCES as SOURCES277
from .source277 import First

ARMS = ("L", "Q", "C", "S", "R0", "R1", "R2", "Z", "FZ")
SOURCES = (*dict.fromkeys((*SOURCES278, *SOURCES277)),
    "research/boolean_coupling/source279.py", "research/boolean_coupling/run279.py",
    "research/boolean_coupling/audit279.py", "research/boolean_coupling/fixtures279.py",
    "research/boolean_coupling/cases279.json", "research/boolean_coupling/historical-input-hashes-279.json",
    "docs/experiments/279-sealed-source-activation.md")


def derivatives(g):
    g = g.astype(float)
    sig = 1 / (1 + np.exp(-g))
    first = sig * (1 - sig)
    second = first * (1 - 2 * sig)
    third = first * ((1 - 2 * sig)**2 - 2 * first)
    return dict(d1=sig + g * first, d2=2 * first + g * second, d3=3 * second + g * third)


def embed(row, length, start):
    full = torch.zeros((1, length, len(row)), device="mps", dtype=torch.float32)
    full[0, start] = torch.tensor(row.astype(np.float32), device="mps")
    return full


def prepare(model, case, context, false):
    start = case["position"]
    layer = model.model.layers[0]
    with torch.no_grad():
        state = model.model.embed_tokens(torch.tensor([case["false_ids"]], device="mps"))
        state[0, start] = model.model.embed_tokens.weight[830]
        with First(model, case) as captured:
            layer.self_attn(layer.input_layernorm(state), attention_mask=context["mask"], position_embeddings=context["rope"])
        data = captured.data
        assert not set(data) & {"z", "gate", "up", "phi", "product", "write", "mid"}
        data["mid"] = (data["before"] + data["attwrite"]).astype(np.float32)
        for name in ("full_k", "full_v"):
            assert np.array_equal(data[name][:, :start], false[name][:, :start])
        point = embed(data["mid"], len(case["false_ids"]), start)
        z = layer.post_attention_layernorm(point)
        data.update(z=array(z[0, start]), gate=array(layer.mlp.gate_proj(z)[0, start]), up=array(layer.mlp.up_proj(z)[0, start]))
    return data


def candidates(layer, case, false, prepared, column_norm):
    start, length = case["position"], len(case["false_ids"])
    dg = prepared["gate"].astype(float) - false["gate"].astype(float)
    d = derivatives(false["gate"])
    phi = {}
    phi["L"] = false["phi"].astype(float) + d["d1"] * dg
    phi["Q"] = phi["L"] + .5 * d["d2"] * dg**2
    phi["C"] = phi["Q"] + d["d3"] * dg**3 / 6
    count = int(np.ceil(.01 * len(dg)))
    score = abs(dg)**4 * abs(prepared["up"].astype(float)) * column_norm
    rank = np.lexsort((np.arange(len(dg)), -score))[:count]
    masks = {"S": np.sort(rank)}
    for number in range(3):
        seed = int(hashlib.sha256((case["key"] + f":279-mask:{number}").encode()).hexdigest()[:16], 16)
        masks["R" + str(number)] = np.sort(np.random.default_rng(seed).choice(len(dg), count, replace=False))
    exact, mixed_inputs = {}, {}
    with torch.no_grad():
        for arm, indices in masks.items():
            mixed = false["gate"].copy()
            mixed[indices] = prepared["gate"][indices]
            full = embed(mixed, length, start)
            # Current gates are evaluated ONLY on the sealed selected coordinates.
            selected = array(layer.mlp.act_fn(full)[0, start])[indices]
            mixed_inputs[arm], exact[arm] = mixed, selected
            current = phi["C"].copy()
            current[indices] = selected.astype(float)
            phi[arm] = current
        writes, products, installed_phi = {}, {}, {}
        for arm, value in phi.items():
            ph = embed(value.astype(np.float32), length, start)
            up = embed(prepared["up"], length, start)
            product = ph * up
            writes[arm] = array(layer.mlp.down_proj(product)[0, start])
            products[arm], installed_phi[arm] = array(product[0, start]), array(ph[0, start])
    writes.update(Z=false["write"].copy(), FZ=false["write"].copy())
    return dict(derivatives=d, dg=dg, score=score, masks=masks, mixed_inputs=mixed_inputs,
                selected_phi=exact, phi=installed_phi, products=products, writes=writes)


class Forecast:
    """Full continuation with source MLP1's actual primal kept False."""

    def __init__(self, model, case, false, prepared, proposals):
        self.model, self.case, self.false = model, case, false
        self.prepared, self.proposals = prepared, proposals
        self.arm, self.record, self.handles = None, None, []

    def __enter__(self):
        layer, start = self.model.model.layers[0], self.case["position"]

        def before(module, args):
            self.record["native_mid"] = array(args[0][0, start])
            expected = self.false["mid"] if self.arm == "FZ" else self.prepared["mid"]
            assert np.array_equal(self.record["native_mid"], expected)
            changed = args[0].clone()
            changed[0, start] = torch.tensor(self.false["mid"], device="mps")
            assert torch.equal(changed[:, :start], args[0][:, :start])
            assert torch.equal(changed[:, start + 1:], args[0][:, start + 1:])
            self.record["used_point"] = array(changed[0, start])
            return (changed,)

        def write(module, args, output):
            self.record["False_primal_write"] = array(output[0, start])
            assert np.array_equal(self.record["False_primal_write"], self.false["write"])
            row = self.proposals["writes"][self.arm]
            changed = output.clone()
            changed[0, start] = torch.tensor(row, device="mps")
            assert torch.equal(changed[:, :start], output[:, :start])
            assert torch.equal(changed[:, start + 1:], output[:, start + 1:])
            self.record["installed_write"] = array(changed[0, start])
            assert np.array_equal(self.record["installed_write"], row)
            return changed

        self.handles = [layer.post_attention_layernorm.register_forward_pre_hook(before), layer.mlp.register_forward_hook(write)]
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
