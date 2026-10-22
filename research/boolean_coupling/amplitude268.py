"""Cross signed correction amplitudes and directions at the frozen267 L MLP point."""

import hashlib
import shutil

import numpy as np

from .audit229 import sha
from .fields267 import ROOT, finite_mlp
from .fields267 import SOURCES as PREVIOUS_SOURCES
from .fields267 import freeze_fields as old_freeze
from .fields267 import normalization as old_normalization
from .input261b import Instability

FAMILIES = ("11", "10", "01", "00", "r")
ARMS = ("F", "Z", "L", "H", *(p+s for s in FAMILIES for p in ("P", "D", "A")), "T")
SOURCES = (*PREVIOUS_SOURCES,
    "research/boolean_coupling/amplitude268.py",
    "research/boolean_coupling/rollout268.py",
    "research/boolean_coupling/audit268.py",
    "docs/experiments/268-signed-amplitudes-and-directions.md")


def bind(out):
    (out/"sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/name.rsplit("/", 1)[-1])
    return {name: sha(ROOT/name) for name in SOURCES}


def normalization(base, point, anchor, gamma, eps):
    parts = old_normalization(base, point, anchor, gamma, eps)
    x, h, g = base.astype(float), point.astype(float)-base.astype(float), gamma.astype(float)
    scale = np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)
    weighted = g*h/scale
    size = np.linalg.norm(weighted, axis=-1, keepdims=True)
    unit = weighted/np.where(size > 0, size, 1.)
    a = np.mean(x*h, axis=-1, keepdims=True)/scale**2
    b = np.mean(h*h, axis=-1, keepdims=True)/scale**2
    rho = (1+2*a+b)**(-.5)
    parts.update(alpha=rho-1+a, amplitude=(rho-1)*size, unit=unit,
                 weighted_norm=size, false_pattern=g*x/scale, overlap=a, energy=b)
    return parts


def freeze_fields(parts, matrices, length, start, key, layer):
    star, nodes = old_freeze(parts, matrices, length, start, key, layer)
    for name in ("unit", "amplitude", "alpha", "weighted_norm"):
        star[name] = parts[name].copy()
    random = []
    for row in range(len(parts["unit"])):
        seed = hashlib.sha256(f"{key}:{layer}:{row}:fixed-unit268".encode()).hexdigest()
        draw = np.random.default_rng(int(seed[:16], 16)).normal(size=parts["unit"].shape[-1])
        random.append(draw/np.linalg.norm(draw))
    star["random_unit"] = np.array(random)
    return star, nodes


def deltas(family, parts, star):
    if family == "11":
        # Preserve the exact26710 arithmetic, rather than refactor and round twice.
        return dict(p=parts["base"], d=parts["direction"])
    if family == "00":
        return dict(p=star["p"], d=star["d"])
    if family == "01":
        if np.any((parts["weighted_norm"] == 0) & (abs(star["amplitude"]) > 1e-12)):
            raise Instability("undefined live direction with nonzero frozen amplitude")
        return dict(p=star["p"], d=star["amplitude"]*parts["unit"])
    assert family in ("10", "r")
    if family == "10" and np.any((star["weighted_norm"] == 0) & (abs(parts["amplitude"]) > 1e-12)):
        raise Instability("undefined L direction with nonzero live amplitude")
    unit = star["unit" if family == "10" else "random_unit"]
    return dict(p=parts["base"], d=parts["amplitude"]*unit)


def policy(arm, parts, original_z, star, matrices, length, start):
    native = arm in ("F", "T", "H")
    base = finite_mlp(original_z if native else parts["linear"], matrices, length, start)
    nodes = dict(base=base)
    if native or arm in ("L", "Z"):
        if arm == "Z":
            assert not np.any(parts["base"]) and not np.any(parts["direction"])
        return base["write"], nodes
    selection, family = arm[0], arm[1:]
    corrections = {}
    if family == "00":
        corrections = {tag: star["c"+tag] for tag in ("p", "d")}
    else:
        values = deltas(family, parts, star)
        for tag in ("p", "d"):
            nodes[tag] = finite_mlp(star["z"]+values[tag], matrices, length, start)
            corrections[tag] = nodes[tag]["write"].astype(float)-star["write"]
    value = base["write"].astype(float)
    for tag in (("p", "d") if selection == "A" else (selection.lower(),)):
        value = value+corrections[tag]
    return value.astype(np.float32), nodes


def geometry(parts, checks, checked):
    checked(checks, "signed_p_factorization", parts["alpha"]*parts["false_pattern"], parts["base"], 1e-10)
    checked(checks, "signed_d_factorization", parts["amplitude"]*parts["unit"], parts["direction"], 1e-10)
    expected = (parts["weighted_norm"][:, 0] > 0).astype(float)
    checked(checks, "unit_direction_norm", np.linalg.norm(parts["unit"], axis=-1), expected, 1e-12)


def preflight():
    import torch

    from .audit229 import checked
    from .fields267 import policy as old_policy

    rng = np.random.default_rng(268)
    x = rng.normal(size=(2, 8)).astype(np.float32)
    point = (x+.2*rng.normal(size=x.shape)).astype(np.float32)
    parts = normalization(x, point, x, np.ones(8), 1e-6)
    checks = []
    geometry(parts, checks, checked)
    matrices = {name: torch.tensor(rng.normal(size=shape).astype(np.float32), device="mps")
                for name, shape in (("gate", (12, 8)), ("up", (12, 8)), ("down", (8, 12)))}
    star, _ = freeze_fields(parts, matrices, 7, 4, "synthetic", 1)
    for prefix in ("P", "D", "A"):
        actual, _ = policy(prefix+"11", parts, x, star, matrices, 7, 4)
        expected, _ = old_policy(prefix+"10", parts, x, star, matrices, 7, 4)
        assert np.array_equal(actual, expected)
    for family in FAMILIES:
        delta = deltas(family, parts, star)
        amplitude = star["amplitude"] if family in ("01", "00") else parts["amplitude"]
        checked(checks, "crossed_signed_amplitude_norm", np.linalg.norm(delta["d"], axis=-1), abs(amplitude[:, 0]), 1e-10)
    zero = normalization(x, x, x, np.ones(8), 1e-6)
    actual, _ = policy("Z", zero, x, None, matrices, 7, 4)
    expected, _ = policy("F", zero, x, None, matrices, 7, 4)
    assert np.array_equal(actual, expected)
    print("268 synthetic geometry, exact26710 equivalence, amplitude control and zero checks passed.", flush=True)


if __name__ == "__main__":
    preflight()
