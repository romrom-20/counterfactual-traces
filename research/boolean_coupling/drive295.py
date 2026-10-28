"""Directed endpoint/path closure test at supplied P gate/up probes."""

from pathlib import Path

import numpy as np

from .change294 import SOURCES as PREVIOUS
from .joint292 import rational
from .subspace293 import statistics

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/drive295.py",
    "research/boolean_coupling/audit295.py",
    "research/boolean_coupling/run295.py",
    "research/boolean_coupling/preflight295.py",
    "docs/experiments/295-finite-drive.md",
)
ARMS = (
    "ENDPOINT_DRIVE",
    "MIDPOINT_DRIVE",
    "SIMPSON_DRIVE",
    "INTEGRAL16",
    "SHUFFLED_DRIVE",
    "ORACLE_DIRECTION",
    "CHANGE_RANGE",
)
PRIMARY = dict(norm=0.001, relative=0.25, cosine=0.95, population=36, per_scope=18)
BOUNDS = dict(
    absolute=1e-8, relative=1e-10, bridge_absolute=1e-4, bridge_relative=1e-5, bridge_norm=0.01
)
CLIP_GATE = 8 * float(np.roots([1, -6, 15, -15])[np.isreal(np.roots([1, -6, 15, -15]))][0].real)


def ingredients(p):
    f = {n: p["False_" + n].astype(float) for n in ("g", "u", "phi")}
    f["d1"] = p["False_slope"].astype(float)
    return f, p["g"].astype(float) - f["g"], p["u"].astype(float) - f["u"]


def response(g, f):
    g = np.asarray(g, dtype=float)
    r, dr = rational(g, 2)
    r0, dr0 = rational(f["g"], 2)
    return f["phi"] + (r - r0) + (f["d1"] - dr0) * (g - f["g"]), dr + f["d1"] - dr0


def kernel(t, f, dg, du):
    g = f["g"] + t * dg
    u = f["u"] + t * du
    phi, slope = response(g, f)
    return (u * slope - f["u"] * f["d1"]) * dg + (phi - f["phi"]) * du


def integrate(f, dg, du, nodes):
    # Split every coordinate at BOTH clipping crossings; zero-length spans stay zero.
    safe = np.where(dg == 0, 1.0, dg)
    cuts = [-CLIP_GATE, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 16.0, CLIP_GATE]
    crosses = np.sort(np.stack([np.clip((cut - f["g"]) / safe, 0, 1) for cut in cuts]), axis=0)
    bounds = np.concatenate([np.zeros_like(crosses[:1]), crosses, np.ones_like(crosses[:1])])
    x, w = np.polynomial.legendre.leggauss(nodes)
    result = np.zeros_like(dg)
    for low, high in zip(bounds[:-1], bounds[1:], strict=True):
        for xx, ww in zip(x, w, strict=True):
            t = (high + low) / 2 + xx * (high - low) / 2
            result += ww * (high - low) / 2 * kernel(t, f, dg, du)
    return result


def evaluate(wd, p, old_projection):
    f, dg, du = ingredients(p)
    phi, slope = response(p["g"], f)
    k1, km = kernel(1.0, f, dg, du), kernel(0.5, f, dg, du)
    smooth = f["u"] * (phi - f["phi"] - f["d1"] * dg) + (phi - f["phi"]) * du
    shuffled = (
        (p["u"].astype(float) * slope - f["u"] * f["d1"]) * np.roll(dg, 1, axis=-1)
        + (phi - f["phi"]) * np.roll(du, 1, axis=-1)
    ) / 2
    products = dict(
        ENDPOINT_DRIVE=k1 / 2,
        MIDPOINT_DRIVE=km,
        SIMPSON_DRIVE=(k1 + 4 * km) / 6,
        INTEGRAL16=integrate(f, dg, du, 16),
        SHUFFLED_DRIVE=shuffled,
    )
    q32 = integrate(f, dg, du, 32)
    out = dict(
        integral16_product=products["INTEGRAL16"],
        integral32_product=q32,
        target=p["nonlinear"].astype(float),
        smooth_product=smooth,
        smooth_n=smooth @ wd.T,
        endpoint_product=k1 / 2,
        midpoint_product=km,
        integral32=q32 @ wd.T,
        gate_change=dg,
        up_change=du,
    )
    out.update({name: prod @ wd.T for name, prod in products.items()})
    desired = np.linalg.norm(out["ENDPOINT_DRIVE"], axis=-1)
    norm = np.linalg.norm(out["SHUFFLED_DRIVE"], axis=-1)
    assert np.all((norm > 0) | (desired == 0)), "Degenerate shuffled control; stop"
    out["SHUFFLED_DRIVE"] *= (desired / np.maximum(norm, 1e-300))[:, None]
    e = out["ENDPOINT_DRIVE"]
    n = out["target"]
    out["ORACLE_DIRECTION"] = (
        e * (np.sum(e * n, axis=-1) / np.maximum(np.sum(e * e, axis=-1), 1e-300))[:, None]
    )
    out["CHANGE_RANGE"] = old_projection.astype(float)
    return out


def records(key, layer, data):
    return [
        dict(
            key=key,
            layer=layer + 1,
            scope=scope,
            arm=arm,
            **statistics(data[arm][row], data["target"][row]),
        )
        for row, scope in enumerate(("source", "verdict"))
        for arm in ARMS
    ]


def decisions(metrics):
    output = {}
    for arm in ARMS:
        selected = [r for r in metrics if r["arm"] == arm]
        scopes = {
            s: sum(
                r["norm"] >= 0.001 and r["relative"] <= 0.25 and r["cosine"] >= 0.95
                for r in selected
                if r["scope"] == s
            )
            for s in ("source", "verdict")
        }
        n = sum(scopes.values())
        output[arm] = dict(
            passes=n,
            total=len(selected),
            scopes=scopes,
            population_pass=n >= 36 and min(scopes.values()) >= 18,
        )
    return output
