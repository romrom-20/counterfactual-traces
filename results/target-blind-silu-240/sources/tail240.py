"""Target-blind bound mask and direct SiLU activation replacement for240."""

import hashlib

import numpy as np
import torch

from .core225b import array

ARMS = ("full", "quadratic", "mask", "random0", "random1", "random2")


def coefficients(g):
    s = 1 / (1 + np.exp(-g))
    return s * (1 + g * (1 - s)), s * (1 - s) * (2 + g * (1 - 2 * s))


def select(g0, g1, u1, down_norm, key):
    near = np.where(g0 * g1 <= 0, 0, np.where(abs(g0) < abs(g1), g0, g1))
    p = 1 / (1 + np.exp(-near))
    bound = p * (1 - p) * (3 + np.maximum(abs(g0), abs(g1))) * abs(g1 - g0)**3 / 6
    score = bound * abs(u1) * down_norm[None]
    k = int(np.ceil(.01 * score.size))
    order = np.lexsort((np.arange(score.size), -score.ravel()))
    mask = np.zeros(score.shape, dtype=bool)
    mask.flat[order[:k]] = True
    masks = {"mask": mask}
    for j in range(3):
        rng = np.random.default_rng(int(hashlib.sha256((key + "240" + str(j)).encode()).hexdigest()[:16], 16))
        control = np.zeros_like(mask)
        for row in range(len(mask)):
            control[row, rng.choice(mask.shape[1], int(mask[row].sum()), replace=False)] = True
        masks["random" + str(j)] = control
    return bound, score, masks


def execute(layer, state, start, prepared, arm):
    """Alter the public SiLU activation; multiplication and down projection stay finite."""
    saved = {}

    def activation(module, args, output):
        gate = array(args[0][0, start:]).astype(float)
        dg = gate - prepared["g0"]
        slope, curve = coefficients(prepared["g0"])
        quadratic = prepared["phi0"] + slope * dg + .5 * curve * dg * dg
        value = output.clone()
        if arm != "full":
            installed = quadratic.astype(np.float32)
            if arm != "quadratic":
                installed = np.where(prepared[arm], array(output[0, start:]), installed)
            value[:, start:] = torch.tensor(installed[None], device="mps")
        saved.update(gate=gate.astype(np.float32), phi=array(value[0, start:]))
        assert torch.equal(value[:, :start], output[:, :start])
        return value

    def up(module, args, output):
        saved["up"] = array(output[0, start:])

    handles = [layer.mlp.act_fn.register_forward_hook(activation), layer.mlp.up_proj.register_forward_hook(up)]
    try:
        result = state + layer.mlp(layer.post_attention_layernorm(state))
    finally:
        for h in handles:
            h.remove()
    return result, saved
