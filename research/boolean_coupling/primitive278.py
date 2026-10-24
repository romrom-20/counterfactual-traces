"""Single-source-MLP finite/False-affine RMS, SiLU and product factorial."""

import hashlib
from itertools import product

import numpy as np
import torch

from .core225b import array, observe
from .firstmlp276 import SOURCES as SOURCES276

BITS = tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))
ARMS = ("F", "Z", "FZ", *(p + bits for p in ("N", "C") for bits in BITS), "G0", "G1", "G2", "T")
SOURCES = (*SOURCES276, "research/boolean_coupling/primitive278.py",
           "research/boolean_coupling/run278.py", "research/boolean_coupling/audit278.py",
           "docs/experiments/278-source-mlp-primitives.md")


def rms_direction(x, d, gamma, eps):
    x, d, gamma = (np.asarray(a, dtype=float) for a in (x, d, gamma))
    sigma = np.sqrt(np.mean(x * x) + eps)
    return gamma * (d / sigma - x * np.mean(x * d) / sigma**3)


def silu_direction(g, d):
    g, d = np.asarray(g, dtype=float), np.asarray(d, dtype=float)
    sig = 1 / (1 + np.exp(-g))
    return sig * (1 + g * (1 - sig)) * d


def product_direction(phi, up, dphi, dup):
    return np.asarray(up, dtype=float) * dphi + np.asarray(phi, dtype=float) * dup


def norm_value(raw, point, anchor, gamma, eps, finite):
    if finite:
        return raw.copy()
    d = np.asarray(point, dtype=float) - anchor["x"].astype(float)
    return (anchor["z"].astype(float) + rms_direction(anchor["x"], d, gamma, eps)).astype(np.float32)


def phi_value(raw, gate, anchor, finite):
    if finite:
        return raw.copy()
    d = gate.astype(float) - anchor["gate"].astype(float)
    return (anchor["phi"].astype(float) + silu_direction(anchor["gate"], d)).astype(np.float32)


def product_value(raw, phi, up, anchor, finite):
    if finite:
        return raw.copy()
    dphi, dup = phi.astype(float) - anchor["phi"].astype(float), up.astype(float) - anchor["up"].astype(float)
    return (anchor["product"].astype(float) + product_direction(anchor["phi"], anchor["up"], dphi, dup)).astype(np.float32)


def full_row(row, length, start):
    out = torch.zeros((1, length, len(row)), device="mps", dtype=torch.float32)
    out[0, start] = torch.tensor(row.astype(np.float32), device="mps")
    return out


def local(layer, point, anchor, bits, length, start):
    """Full-shaped public primitives, source-row replacements between them."""
    n, s, p = map(int, bits)
    gamma, eps = array(layer.post_attention_layernorm.weight), layer.post_attention_layernorm.variance_epsilon
    x = full_row(point, length, start)
    raw_z = layer.post_attention_layernorm(x)
    z = raw_z.clone()
    z[0, start] = torch.tensor(norm_value(array(raw_z[0, start]), point, anchor, gamma, eps, n), device="mps")
    gate, up = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
    raw_phi = layer.mlp.act_fn(gate)
    phi = raw_phi.clone()
    phi[0, start] = torch.tensor(phi_value(array(raw_phi[0, start]), array(gate[0, start]), anchor, s), device="mps")
    raw_product = phi * up
    product = raw_product.clone()
    product[0, start] = torch.tensor(product_value(array(raw_product[0, start]), array(phi[0, start]), array(up[0, start]), anchor, p), device="mps")
    write = layer.mlp.down_proj(product)
    return {name: array(value[0, start]) for name, value in dict(x=x, raw_z=raw_z, z=z, gate=gate, up=up,
            raw_phi=raw_phi, phi=phi, raw_product=raw_product, product=product, write=write).items()}


def random_writes(nodes, key):
    baseline, target = (nodes["N" + b]["write"].astype(float) for b in ("000", "111"))
    difference = target - baseline
    output, draws = {}, {}
    for number in range(3):
        name = "G" + str(number)
        seed = int(hashlib.sha256((key + f":278-write:{number}").encode()).hexdigest()[:16], 16)
        draw = np.random.default_rng(seed).normal(size=difference.shape)
        draw *= np.linalg.norm(difference) / np.linalg.norm(draw)
        output[name], draws[name] = baseline + draw, draw
    return output, draws


def family(arm):
    if arm in ("F", "FZ", "Z"):
        return "F111" if arm != "FZ" else "F000"
    if arm == "T":
        return "N111"
    return "N000" if arm.startswith("G") else arm


class Engine:
    """Actual primitive interventions only at source MLP1; outer residual stays native."""

    def __init__(self, model, case, points, anchor, nodes, random):
        self.model, self.case, self.points = model, case, points
        self.anchor, self.nodes, self.random = anchor, nodes, random
        self.arm, self.record, self.handles = None, None, []

    def replace(self, output, row):
        start = self.case["position"]
        changed = output.clone()
        changed[0, start] = torch.tensor(row.astype(np.float32), device="mps")
        assert torch.equal(changed[:, :start], output[:, :start])
        assert torch.equal(changed[:, start + 1:], output[:, start + 1:])
        return changed

    def __enter__(self):
        layer, start = self.model.model.layers[0], self.case["position"]
        gamma, eps = array(layer.post_attention_layernorm.weight), layer.post_attention_layernorm.variance_epsilon

        def before(module, args):
            self.record["native_mid"] = array(args[0][0, start])
            native = "F" if self.arm in ("F", "FZ") else "N"
            assert np.array_equal(self.record["native_mid"], self.points[native])
            point = self.points[family(self.arm)[0]]
            self.record["used_point"] = point
            return (self.replace(args[0], point),)

        def norm(module, args, output):
            self.record["raw_z"] = array(output[0, start])
            bits = family(self.arm)[1:]
            row = norm_value(self.record["raw_z"], self.record["used_point"], self.anchor, gamma, eps, int(bits[0]))
            self.record["z"] = row
            return self.replace(output, row)

        def gate(module, args, output):
            self.record["gate"] = array(output[0, start])

        def up(module, args, output):
            self.record["up"] = array(output[0, start])

        def phi(module, args, output):
            self.record["raw_phi"] = array(output[0, start])
            row = phi_value(self.record["raw_phi"], self.record["gate"], self.anchor, int(family(self.arm)[2]))
            self.record["phi"] = row
            return self.replace(output, row)

        def product(module, args):
            self.record["raw_product"] = array(args[0][0, start])
            row = product_value(self.record["raw_product"], self.record["phi"], self.record["up"], self.anchor, int(family(self.arm)[3]))
            self.record["product"] = row
            return (self.replace(args[0], row),)

        def write(module, args, output):
            self.record["raw_write"] = array(output[0, start])
            expected = self.nodes[family(self.arm)]
            for name in ("x", "raw_z", "z", "gate", "up", "raw_phi", "phi", "raw_product", "product", "write"):
                actual = self.record["used_point"] if name == "x" else self.record["raw_write" if name == "write" else name]
                assert np.array_equal(actual, expected[name]), (self.arm, name)
            row = self.random[self.arm].astype(np.float32) if self.arm.startswith("G") else self.record["raw_write"]
            self.record["installed_write"] = row
            return self.replace(output, row)

        self.handles = [layer.post_attention_layernorm.register_forward_pre_hook(before),
            layer.post_attention_layernorm.register_forward_hook(norm), layer.mlp.gate_proj.register_forward_hook(gate),
            layer.mlp.up_proj.register_forward_hook(up), layer.mlp.act_fn.register_forward_hook(phi),
            layer.mlp.down_proj.register_forward_pre_hook(product), layer.mlp.register_forward_hook(write)]
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()

    def forward(self, arm):
        self.arm, self.record = arm, {}
        ids = self.case["false_ids" if arm in ("F", "FZ") else "true_ids"]
        states, _, scores = observe(self.model, torch.tensor([ids], device="mps"))
        return states, array(scores), self.record
