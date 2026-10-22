"""Finite MLP correction fields with crossed generator and operating-point clamps."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .core225b import write_json
from .input263 import plans
from .transmission265 import digest, full, project

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("11", "10", "01", "00", "r")
ARMS = ("F", "Z", "L", "H", *(p+s for s in FAMILIES for p in ("P", "D", "A")), "T")
SOURCES = (
    "research/boolean_coupling/fields267.py",
    "research/boolean_coupling/rollout267.py",
    "research/boolean_coupling/audit267.py",
    "docs/experiments/267-correction-fields-and-operating-points.md",
    "research/boolean_coupling/input263.py",
    "research/boolean_coupling/input261b.py",
    "research/boolean_coupling/audit261b.py",
    "research/boolean_coupling/curvature255.py",
    "research/boolean_coupling/transmission265.py",
    "research/boolean_coupling/silu258.py",
    "research/boolean_coupling/mlp258.py",
    "research/boolean_coupling/local265.py",
    "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/settings.json",
)


def read(path):
    return json.loads(path.read_text())


def emit(out, path, value):
    assert path.resolve().is_relative_to(out.resolve()) and not path.exists(), path
    write_json(path, value)


def bind(out):
    (out/"sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/Path(name).name)
    return {name: sha(ROOT/name) for name in SOURCES}


def finite_mlp(z, matrices, length, start):
    z = z.astype(np.float32)
    gate, up = (project(z, matrices[n], length, start) for n in ("gate", "up"))
    with torch.no_grad():
        phi = torch.nn.functional.silu(full(gate, length, start))[0, start+1:].cpu().numpy()
    product = phi*up
    write = project(product, matrices["down"], length, start)
    return dict(z=z, gate=gate, up=up, phi=phi, product=product, write=write)


def node_seal(nodes):
    return {name: {field: digest(value) for field, value in node.items()}
            for name, node in nodes.items()}


def random_field(value, key, layer, tag):
    rows = []
    for row, vector in enumerate(value.astype(float)):
        seed = hashlib.sha256(f"{key}:{layer}:{row}:{tag}:fields267".encode()).hexdigest()
        draw = np.random.default_rng(int(seed[:16], 16)).normal(size=vector.shape)
        rows.append(draw/np.linalg.norm(draw)*np.linalg.norm(vector))
    return np.array(rows, dtype=np.float32)


def freeze_fields(parts, matrices, length, start, key, layer):
    z = parts["linear"].copy()
    base = finite_mlp(z, matrices, length, start)
    result = dict(z=z, p=parts["base"].copy(), d=parts["direction"].copy(),
                  write=base["write"].astype(float))
    nodes = dict(base=base)
    for tag, field in (("p", "base"), ("d", "direction")):
        nodes[tag] = finite_mlp(z+parts[field], matrices, length, start)
        result["c"+tag] = nodes[tag]["write"].astype(float)-result["write"]
        result["r"+tag] = random_field(result["c"+tag], key, layer, tag)
    return result, nodes


def policy(arm, parts, original_z, star, matrices, length, start):
    """Keep the L backbone adaptive; cross ONLY the two correction-field arguments."""
    native = arm in ("F", "T", "H")
    z = original_z if native else parts["linear"]
    base = finite_mlp(z, matrices, length, start)
    nodes = dict(base=base)
    if native or arm == "L":
        return base["write"], nodes
    selection, family = ("A", "11") if arm == "Z" else (arm[0], arm[1:])
    corrections = {}
    if family in ("00", "r"):
        for tag in ("p", "d"):
            corrections[tag] = star[("c" if family == "00" else "r")+tag].astype(float)
    else:
        reader = parts["linear"] if family[1] == "1" else star["z"]
        if family[1] == "1":
            baseline = base["write"].astype(float)
        else:
            baseline = star["write"]
        for tag, field in (("p", "base"), ("d", "direction")):
            delta = parts[field] if family[0] == "1" else star[tag]
            nodes[tag] = finite_mlp(reader+delta, matrices, length, start)
            corrections[tag] = nodes[tag]["write"].astype(float)-baseline
    value = base["write"].astype(float)
    for tag in (("p", "d") if selection == "A" else (selection.lower(),)):
        value = value+corrections[tag]
    return value.astype(np.float32), nodes


def normalization(base, point, anchor, gamma, eps):
    return plans(base.astype(float), point.astype(float), anchor.astype(float),
                 gamma.astype(float), eps)


def public_norm(value, gamma, eps, length, start):
    module = qwen.Qwen2RMSNorm(len(gamma), eps).to(device="mps", dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(torch.tensor(gamma.astype(np.float32), device="mps"))
        return module(full(value, length, start))[0, start+1:].cpu().numpy()


class PublicHead:
    """Match final RMSNorm full-sequence shape and the one-token full-vocabulary head."""

    def __init__(self, handle, config):
        self.gamma = handle.get_tensor("model.norm.weight").float().numpy()
        self.eps = config["rms_norm_eps"]
        name = "lm_head.weight" if "lm_head.weight" in handle.keys() else "model.embed_tokens.weight"
        self.weight = handle.get_tensor(name).to(device="mps", dtype=torch.float32)

    def __call__(self, verdict, length):
        z = public_norm(verdict[None], self.gamma, self.eps, length, length-2)
        with torch.no_grad():
            logits = torch.nn.functional.linear(
                torch.tensor(z[None], device="mps"), self.weight)
            return logits[0, -1, [895, 830]].cpu().numpy()


def preflight():
    """Synthetic algebra and real full-shape primitive checks, no study inputs."""
    rng = np.random.default_rng(267)
    x = rng.normal(size=(2, 8)).astype(np.float32)
    z = x.copy()
    gamma = np.ones(8)
    zero = normalization(x, x, z, gamma, 1e-6)
    assert np.array_equal(zero["linear"], z)
    assert not np.any(zero["base"]) and not np.any(zero["direction"])
    matrices = {name: torch.tensor(rng.normal(size=shape).astype(np.float32), device="mps")
                for name, shape in (("gate", (12, 8)), ("up", (12, 8)), ("down", (8, 12)))}
    parts = normalization(x, x+.1*rng.normal(size=x.shape), z, gamma, 1e-6)
    star, _ = freeze_fields(parts, matrices, 7, 4, "synthetic", 1)
    zero_star, _ = freeze_fields(zero, matrices, 7, 4, "synthetic", 1)
    baseline, _ = policy("L", zero, z, None, matrices, 7, 4)
    installed, _ = policy("Z", zero, z, zero_star, matrices, 7, 4)
    assert np.array_equal(installed, baseline)
    for arm in ("P11", "D11"):
        installed, nodes = policy(arm, parts, z, star, matrices, 7, 4)
        assert np.array_equal(installed, nodes[arm[0].lower()]["write"])
    for tag in ("p", "d"):
        error = abs(np.linalg.norm(star["r"+tag].astype(float), axis=-1)
                    -np.linalg.norm(star["c"+tag], axis=-1))
        assert np.max(error) <= 1e-5
    print("267 synthetic zero, finite fields and exact direct-corner identities passed", flush=True)


if __name__ == "__main__":
    preflight()
