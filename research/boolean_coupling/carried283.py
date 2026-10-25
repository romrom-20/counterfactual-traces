"""One staged derivative of the missing-product map at an own-input carried state."""

import hashlib
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .composition280 import digest
from .origin282 import SOURCES as ORIGINAL
from .qualify282b import SOURCES as QUALIFIED_B
from .qualify282c import SOURCES as QUALIFIED_C
from .source279 import derivatives

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ("K", "P", "H", "J", "G0", "G1", "G2", "FZ")
CONTROLS = ("C1", "U", "N", "T")
ARMS = ("F", "B", *CANDIDATES, *CONTROLS)
EVAL = ("B", "K", "P", "H", "J", "G0", "G1", "G2", *CONTROLS)
PRODUCT_ARMS = ("K", "P", "H", "J", "G0", "G1", "G2", "U", "N")
SOURCES = (
    *dict.fromkeys((*ORIGINAL, *QUALIFIED_B, *QUALIFIED_C)),
    "research/boolean_coupling/carried283.py",
    "research/boolean_coupling/run283.py",
    "research/boolean_coupling/audit283.py",
    "research/boolean_coupling/preflight283.py",
    "docs/experiments/283-carried-state-response.md",
)


def response(anchor, corner, point_u, w):
    """Declared Float64 tangent with observed Float32 C gate/up/activation anchors."""
    x = corner["mid"].astype(float)
    direction = point_u.astype(float) - x
    rms = np.sqrt(np.mean(x * x) + w["eps"])
    direct = w["norm"] * direction / rms
    scale = -w["norm"] * x * np.mean(x * direction) / rms**3
    dz = direct + scale
    dg, du = (dz @ w[n + "_weight"].T for n in ("gate", "up"))
    g, up = corner["gate"].astype(float), corner["up"].astype(float)
    e = corner["phi"].astype(float) - corner["cubic"].astype(float)
    df = derivatives(anchor["gate"])
    step = g - anchor["gate"].astype(float)
    prime = derivatives(g)["d1"]
    cubic_prime = df["d1"] + df["d2"] * step + 0.5 * df["d3"] * step**2
    gate_delta = (prime - cubic_prime) * dg * up
    up_delta = e * du
    result = dict(
        direction=direction,
        rms_direct=direct,
        rms_scale=scale,
        rms_jvp=dz,
        dg=dg,
        du=du,
        e=e,
        silu_prime=prime,
        cubic_prime=cubic_prime,
        gate_delta=gate_delta,
        up_delta=up_delta,
        increment=gate_delta + up_delta,
    )
    assert all(np.isfinite(a).all() for a in result.values())
    return result


def products(carry, calculated):
    inc = calculated["increment"]
    d = dict(
        K=carry, P=carry + inc, H=carry + calculated["gate_delta"], J=carry + calculated["up_delta"]
    )
    d.update({a: carry for a in ("G0", "G1", "G2")})
    return d


def random_writes(case, increment):
    result = {}
    for i in range(3):
        seed = int(
            hashlib.sha256((case["key"] + f":283-increment:{i}").encode()).hexdigest()[:16], 16
        )
        draw = np.random.default_rng(seed).normal(size=increment.shape)
        result["G" + str(i)] = (draw * np.linalg.norm(increment)) / np.linalg.norm(draw)
    return result


class Observations:
    """Resolved lossless282 aliases, verified including their281d dependencies."""

    def __init__(self, reference, upstream, entry):
        key = entry["key"]
        path = reference / "cases" / (key + ".npz")
        assert sha(path) == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        old = upstream / "cases" / (key + ".npz")
        assert sha(old) == meta["reference_arrays_sha256"]
        with np.load(path) as saved, np.load(old) as prior:
            values = {n: saved[n] for n in saved.files}
            for n, alias in meta["array_aliases"].items():
                assert n not in values
                values[n] = (
                    prior[alias["reference"]] if "reference" in alias else values[alias["local"]]
                )
        assert set(values) == set(meta["logical_arrays"])
        for n, value in values.items():
            specification = meta["logical_arrays"][n]
            assert (
                str(value.dtype) == specification["dtype"]
                and list(value.shape) == specification["shape"]
            )
            assert digest(value) == specification["sha256"]
        self.values, self.files = values, list(values)

    def __getitem__(self, name):
        return self.values[name]


def restore(path, metadata, reference):
    with np.load(path) as saved:
        values = {n: saved[n] for n in saved.files}
    for name, alias in metadata["array_aliases"].items():
        assert name not in values
        values[name] = (
            reference[alias["reference"]] if "reference" in alias else values[alias["local"]]
        )
    assert set(values) == set(metadata["logical_arrays"])
    for name, value in values.items():
        spec = metadata["logical_arrays"][name]
        assert (
            str(value.dtype) == spec["dtype"]
            and list(value.shape) == spec["shape"]
            and digest(value) == spec["sha256"]
        )
    return values
