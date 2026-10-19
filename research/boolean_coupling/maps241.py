"""Checkpoint-based receiving maps and independent batched analytic tangents."""

import numpy as np
import torch
from torch.nn import functional as F

from .audit229 import build
from .audit235 import finite
from .checkpoint238 import mlp_derivative
from .curvature239 import torch_function


def maps(w, context, full, start, kind):
    if kind == "mlp":
        return torch_function(w), lambda d: mlp_derivative(w, full, d), lambda y: finite(w, {}, y, "mlp")
    original = build(w, context, full)
    x = full[start:]
    width, groups = int(w["width"]), int(w["groups"])
    cosine, sine = context["rope_cos"][0, start:], context["rope_sin"][0, start:]

    def heads(v):
        return v.reshape(*v.shape[:-1], -1, width).swapaxes(-3, -2)

    def rotate(v):
        a, b = np.split(v, 2, -1)
        return v * cosine + np.concatenate((-b, a), -1) * sine

    def tangent(d):
        rms = original["rms"][start:]
        dz = w["attn_norm"] * (d / rms - x * np.mean(x * d, -1, keepdims=True) / rms**3)
        dq, dk, dv = (heads(dz @ w[n + "_weight"].T) for n in ("q", "k", "v"))
        dq, dk = rotate(dq), rotate(dk)
        dk, dv = (np.repeat(v, groups, axis=-3) for v in (dk, dv))
        shape = (*d.shape[:-2], original["k"].shape[0], len(full), width)
        kfull, vfull = np.zeros(shape), np.zeros(shape)
        kfull[..., start:, :], vfull[..., start:, :] = dk, dv
        dl = (dq @ original["k"].swapaxes(-1, -2) + original["q"][:, start:] @ kfull.swapaxes(-1, -2)) * w["scaling"]
        p = original["p"][:, start:]
        dp = p * (dl - np.sum(p * dl, -1, keepdims=True))
        dy = dp @ original["v"] + p @ vfull
        return d + dy.swapaxes(-3, -2).reshape(d.shape) @ w["o_weight"].T

    def numpy_function(y):
        z = y / np.sqrt(np.mean(y * y, -1, keepdims=True) + w["attn_eps"]) * w["attn_norm"]
        q, k, v = (heads(z @ w[n + "_weight"].T + w.get(n + "_bias", 0)) for n in ("q", "k", "v"))
        q, k = rotate(q), rotate(k)
        k, v = (np.repeat(a, groups, axis=-3) for a in (k, v))
        shape = (*y.shape[:-2], original["k"].shape[0], start, width)
        k = np.concatenate((np.broadcast_to(original["k"][:, :start], shape), k), -2)
        v = np.concatenate((np.broadcast_to(original["v"][:, :start], shape), v), -2)
        logits = q @ k.swapaxes(-1, -2) * w["scaling"] + context["mask"][0, :, start:, :]
        exp = np.exp(logits - logits.max(-1, keepdims=True))
        p = exp / exp.sum(-1, keepdims=True)
        output = (p @ v).swapaxes(-3, -2).reshape(y.shape)
        return y + output @ w["o_weight"].T + w.get("o_bias", 0)

    p = {k: torch.from_numpy(np.asarray(v)) for k, v in w.items()}
    cos, sin = torch.from_numpy(cosine), torch.from_numpy(sine)
    prefix_k = torch.from_numpy(original["k"][:, :start].copy())
    prefix_v = torch.from_numpy(original["v"][:, :start].copy())
    mask = torch.from_numpy(context["mask"][0, :, start:, :].copy())

    def th(v):
        return v.reshape(v.shape[0], -1, width).transpose(0, 1)

    def tr(v):
        a, b = v.chunk(2, -1)
        return v * cos + torch.cat((-b, a), -1) * sin

    def function(y):
        z = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + p["attn_eps"]) * p["attn_norm"]
        q, k, v = (th(F.linear(z, p[n + "_weight"], p.get(n + "_bias"))) for n in ("q", "k", "v"))
        q, k = tr(q), tr(k)
        k, v = (a.repeat_interleave(groups, dim=0) for a in (k, v))
        k, v = torch.cat((prefix_k, k), 1), torch.cat((prefix_v, v), 1)
        probability = torch.softmax(q @ k.transpose(-1, -2) * p["scaling"] + mask, -1)
        value = (probability @ v).transpose(0, 1).reshape_as(y)
        return y + F.linear(value, p["o_weight"], p.get("o_bias"))
    return function, tangent, numpy_function
