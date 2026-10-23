"""Native source-value reanchoring: separate False background from residual change."""

import hashlib

import numpy as np
import torch

from .core225b import array
from .source273 import ROOT, Engine
from .source273 import SOURCES as SOURCES273

FAMILIES = ("U", "P", "C", "D", "E", "R")
ARMS = ("F", *FAMILIES, "J", "X", "Z", "FZ", "T")
SOURCES = (*SOURCES273, "research/boolean_coupling/reanchor274.py",
           "research/boolean_coupling/run274.py", "research/boolean_coupling/audit274.py",
           "docs/experiments/274-source-residual-reanchoring.md")


def inputs(bank, embeddings):
    false = np.concatenate((embeddings[0][None], bank["F_states"][:-1, 0]), axis=0)
    true = np.concatenate((embeddings[1][None], bank["G_source"][:-1]), axis=0)
    return false.astype(float), true.astype(float) - false.astype(float)


def geometry(own, donor, reference, embeddings, key):
    x, h = inputs(own, embeddings)
    y, k = inputs(donor, embeddings)
    _, fixed = inputs(reference, embeddings)
    rng = np.random.default_rng(int(hashlib.sha256((key + ":reanchor274").encode()).hexdigest()[:16], 16))
    random = rng.normal(size=k.shape)
    random = random / np.linalg.norm(random, axis=-1, keepdims=True) * np.linalg.norm(k, axis=-1, keepdims=True)
    constant = np.broadcast_to((embeddings[1].astype(float) - embeddings[0].astype(float))[None], h.shape).copy()
    return dict(U=(x, h), P=(x, k), C=(x, fixed), D=(y, h), E=(x, constant), R=(x, random))


def public_map(layer, value, length, start):
    state = torch.zeros((1, length, 1536), device="mps", dtype=torch.float32)
    state[0, start] = torch.tensor(value.astype(np.float32), device="mps")
    z = layer.input_layernorm(state)
    v = layer.self_attn.v_proj(z)
    return array(z[0, start]), array(v[0, start]).reshape(2, 128)


def fields(model, case, parameters):
    nodes, deltas = {}, {}
    length, start = len(case["false_ids"]), case["position"]
    for arm, (center, step) in parameters.items():
        saved = dict(center=center, step=step, z0=[], z1=[], v0=[], v1=[])
        for i, layer in enumerate(model.model.layers):
            for suffix, value in (("0", center[i]), ("1", center[i] + step[i])):
                z, v = public_map(layer, value, length, start)
                saved["z" + suffix].append(z)
                saved["v" + suffix].append(v)
        for name in ("z0", "z1", "v0", "v1"):
            saved[name] = np.stack(saved[name])
        deltas[arm] = saved["v1"].astype(float) - saved["v0"].astype(float)
        nodes.update({"geom_" + arm + "_" + name: value for name, value in saved.items()})
    return deltas, nodes


class ValueEngine(Engine):
    def diagnostic(self, *args):
        # 274 tests emitter generation and actual continuation; 273's fixed-input
        # attention diagnostic is preserved, not redundantly recomputed here.
        return None


def load_bank(path):
    names = ("F_states", "T_states", "F_scores", "T_scores", "X_states", "X_scores",
             "Z_states", "Z_scores", "F_norm", "G_norm", "F_K", "G_K", "F_V", "G_V", "G_source")
    with np.load(path) as data:
        return {name: data[name].copy() for name in names}


def check_own(nodes, own):
    for tag, expected in (("z0", "F_norm"), ("z1", "G_norm"), ("v0", "F_V"), ("v1", "G_V")):
        assert np.array_equal(nodes["geom_U_" + tag], own[expected]), tag


__all__ = ["ROOT", "ARMS", "FAMILIES", "SOURCES", "ValueEngine", "geometry", "fields", "load_bank", "check_own"]
