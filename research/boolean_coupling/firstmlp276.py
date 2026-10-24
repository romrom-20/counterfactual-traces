"""First-source MLP: finite input factorial and explicitly qualified gain forecasts."""

import hashlib

import numpy as np
import torch

from .core225b import array, observe
from .reading275 import SOURCES as SOURCES275
from .source273 import ROOT

ARMS = ("F", "U", "C", "R", "A", "I", "L", "S", "G0", "G1", "G2", "Z", "FZ", "T")
SOURCES = (*SOURCES275, "research/boolean_coupling/firstmlp276.py",
           "research/boolean_coupling/run276.py", "research/boolean_coupling/audit276.py",
           "docs/experiments/276-first-source-mlp-gain.md")


def coordinates(old):
    x = old["F_mid"][0].astype(float)
    t = old["N_mid"][0].astype(float)
    e = old["N_before"][0].astype(float) - old["F_before"][0].astype(float)
    value = old["N_corner_W_write"][0].astype(float) - old["N_corner_B_write"][0].astype(float)
    # Fix the actual Float32 carrier point first; put its casting error in r.
    carrier = (x + e + value).astype(np.float32)
    c = carrier.astype(float) - x
    r = t - carrier.astype(float)
    points = dict(F=x.astype(np.float32), C=carrier,
                  R=(x + r).astype(np.float32), T=(carrier.astype(float) + r).astype(np.float32))
    assert np.array_equal(points["T"], old["N_mid"][0])
    return points, dict(embedding=e, self_value=value, carrier=c, read=r)


def finite(layer, point, length, start):
    full = torch.zeros((1, length, 1536), device="mps")
    full[0, start] = torch.tensor(point, device="mps")
    z = layer.post_attention_layernorm(full)
    g, u = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
    phi = layer.mlp.act_fn(g)
    product = phi * u
    write = layer.mlp.down_proj(product)
    return {name: array(value[0, start]) for name, value in
            dict(x=full, z=z, gate=g, up=u, phi=phi, product=product, write=write).items()}


def tangent(w, x, direction):
    """Derivative of the smooth CPU64 MLP, not a derivative of Float32 execution."""
    x, direction = x.astype(float), direction.astype(float)
    sigma = np.sqrt(np.mean(x * x) + float(w["eps"]))
    z = w["norm"] * x / sigma
    dz = w["norm"] * (direction / sigma - x * np.mean(x * direction) / sigma**3)
    g, u = (w[name + "_weight"] @ z for name in ("gate", "up"))
    dg, du = (w[name + "_weight"] @ dz for name in ("gate", "up"))
    sigmoid = 1 / (1 + np.exp(-g))
    phi = g * sigmoid
    dphi = sigmoid * (1 + g * (1 - sigmoid)) * dg
    dp = dphi * u + phi * du
    return dict(x=x, direction=direction, z=z, dz=dz, gate=g, up=u, phi=phi,
                dgate=dg, dup=du, dphi=dphi, dproduct=dp, write=w["down_weight"] @ dp)


def proposals(nodes, derivatives, key):
    f, c, q, t = (nodes[n]["write"].astype(float) for n in ("F", "C", "R", "T"))
    low = c + derivatives["F"]["write"]
    shifted = c + derivatives["C"]["write"]
    gain = shifted - low
    out = dict(U=t, C=c, R=q, A=c + q - f, I=t - c - q + 2 * f,
               L=low, S=shifted, Z=f, FZ=f)
    random = {}
    for number in range(3):
        seed = int(hashlib.sha256((key + f":276gain:{number}").encode()).hexdigest()[:16], 16)
        vector = np.random.default_rng(seed).normal(size=1536)
        vector *= np.linalg.norm(gain) / np.linalg.norm(vector)
        random["G" + str(number)] = vector
        out["G" + str(number)] = low + vector
    return out, gain, random


def primal(arm):
    if arm in ("F", "FZ", "Z"):
        return "F"
    if arm in ("U", "T", "I"):
        return "T"
    return "R" if arm == "R" else "C"


class SiteEngine:
    """Change only MLP1's source row; its outer residual stays at the native mid."""

    def __init__(self, model, case, points, nodes, writes):
        self.model, self.case = model, case
        self.points, self.nodes, self.writes = points, nodes, writes
        self.arm, self.record, self.handles = None, None, []

    def __enter__(self):
        layer = self.model.model.layers[0]

        def before(module, args):
            start, arm = self.case["position"], self.arm
            self.record["native_mid"] = array(args[0][0, start])
            expected = self.points["F" if arm in ("F", "FZ") else "T"]
            assert np.array_equal(self.record["native_mid"], expected)
            out = args[0].clone()
            out[0, start] = torch.tensor(self.points[primal(arm)], device="mps")
            assert torch.equal(out[:, :start], args[0][:, :start])
            assert torch.equal(out[:, start + 1:], args[0][:, start + 1:])
            self.record["used_point"] = array(out[0, start])
            return (out,)

        def write(module, args, output):
            start = self.case["position"]
            self.record["raw_write"] = array(output[0, start])
            assert np.array_equal(self.record["raw_write"], self.nodes[primal(self.arm)]["write"])
            if self.arm in ("F", "T"):
                self.record["installed_write"] = self.record["raw_write"]
                return output
            out = output.clone()
            out[0, start] = torch.tensor(self.writes[self.arm].astype(np.float32), device="mps")
            assert torch.equal(out[:, :start], output[:, :start])
            assert torch.equal(out[:, start + 1:], output[:, start + 1:])
            self.record["installed_write"] = array(out[0, start])
            assert np.array_equal(self.record["installed_write"], self.writes[self.arm].astype(np.float32))
            return out

        self.handles = [layer.post_attention_layernorm.register_forward_pre_hook(before),
                        layer.mlp.register_forward_hook(write)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def forward(self, arm):
        self.arm, self.record = arm, {}
        ids = self.case["false_ids" if arm in ("F", "FZ") else "true_ids"]
        states, _, scores = observe(self.model, torch.tensor([ids], device="mps"))
        return states, array(scores), self.record


__all__ = ["ROOT", "ARMS", "SOURCES", "coordinates", "finite", "tangent", "proposals", "SiteEngine"]
