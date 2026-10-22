"""Finite primitive ablations of the L-anchored adaptive MLP backbone."""

import hashlib
import shutil
from pathlib import Path

import numpy as np

from .amplitude268 import SOURCES as PREVIOUS_SOURCES
from .amplitude268 import policy as amplitude_policy
from .audit229 import sha
from .fields267 import ROOT
from .silu258 import sigmoid
from .transmission265 import project

ARMS = ("F", "Z", "L", "H", "B", "J", "G", "X", "R", "T")
SOURCES = (*PREVIOUS_SOURCES, "research/boolean_coupling/backbone270.py",
           "research/boolean_coupling/rollout270.py", "research/boolean_coupling/audit270.py",
           "docs/experiments/270-backbone-curvature-feedback.md")


def bind(out):
    (out / "sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    return {name: sha(ROOT / name) for name in SOURCES}


def cache_nodes(star, nodes):
    for name in ("gate", "up", "phi", "product"):
        star["backbone_" + name] = nodes["base"][name].copy()


def components(node, star):
    g0, u0, f0, p0 = (star["backbone_" + n].astype(float)
                      for n in ("gate", "up", "phi", "product"))
    g, u, f, product = (node[n].astype(float) for n in ("gate", "up", "phi", "product"))
    dg, du, df = g - g0, u - u0, f - f0
    sigma = sigmoid(g0)
    slope = sigma * (1 + g0 * (1 - sigma))
    linear = p0 + slope * dg * u0 + f0 * du
    gate = (df - slope * dg) * u0
    cross = df * du
    smooth_gate = (g * sigmoid(g) - g0 * sigma - slope * dg) * u0
    return dict(linear=linear, gate=gate, cross=cross,
                arithmetic=product - linear - gate - cross,
                gate_smooth=smooth_gate, gate_arithmetic=gate - smooth_gate)


def product_for(name, product, parts):
    # Direct finite removals retain the same measured arithmetic remainder.
    if name == "B":
        return product.copy()
    value = product.astype(float)
    if name in ("J", "X"):
        value = value - parts["gate"]
    if name in ("J", "G"):
        value = value - parts["cross"]
    return value.astype(np.float32)


def random_write(value, key, layer):
    result = []
    for row, v in enumerate(value.astype(float)):
        seed = hashlib.sha256(f"{key}:{layer}:{row}:backbone270".encode()).hexdigest()
        draw = np.random.default_rng(int(seed[:16], 16)).normal(size=v.shape)
        result.append(draw / np.linalg.norm(draw) * np.linalg.norm(v))
    return np.array(result, dtype=np.float32)


def policy(arm, parts, original_z, star, matrices, length, start, key, layer):
    original = arm if arm in ("F", "Z", "L", "H", "T") else "A10"
    value, nodes = amplitude_policy(original, parts, original_z, star, matrices, length, start)
    if arm in ("F", "Z", "L", "H", "T"):
        return value, nodes, {}
    terms = components(nodes["base"], star)
    product = product_for("J" if arm == "R" else arm, nodes["base"]["product"], terms)
    changed = (nodes["base"]["write"] if arm == "B"
               else project(product, matrices["down"], length, start))
    mechanism = dict(**terms, installed_product=product, projected_write=changed)
    if arm == "R":
        removed = nodes["base"]["write"].astype(float) - changed.astype(float)
        random = random_write(removed, key, layer)
        mechanism.update(removed=removed, random=random)
        changed = nodes["base"]["write"].astype(float) - random.astype(float)
    if arm != "B":
        value = changed.astype(float)
        for tag in ("p", "d"):
            value = value + (nodes[tag]["write"].astype(float) - star["write"])
        value = value.astype(np.float32)
    return value, nodes, mechanism
