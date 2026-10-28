"""False-value/slope anchored rational gate updates at all affected MLPs."""

from pathlib import Path

import numpy as np
import torch

from .cache272 import digest
from .coeff291 import SOURCES as PREVIOUS
from .core225b import array
from .source279 import derivatives

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("P", "D", "C", "B", "A")
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/joint292.py",
    "research/boolean_coupling/run292.py",
    "research/boolean_coupling/audit292.py",
    "research/boolean_coupling/preflight292.py",
    "research/boolean_coupling/fixtures292.py",
    "research/boolean_coupling/cases292.json",
    "research/boolean_coupling/history292.json",
    "docs/experiments/292-joint-gate-forecast.md",
)
PRIMARY = dict(
    native_relative=0.05,
    native_cosine=0.99,
    margin=0.1,
    trajectory_relative=0.1,
    trajectory_cosine=0.995,
    norm_min=0.001,
    effect_relative=0.25,
    effect_cosine=0.95,
    effect_absolute=0.005,
    effect_margin_relative=0.25,
    material=0.02,
    removal_vector=0.1,
    removal_margin=0.2,
    population=24,
    per_layout=6,
)
BOUNDS = dict(replay=0.0, witness=1e-4, head=0.005, derivative_AD=1e-8, derivative_FD=2e-6)
PUBLIC_SEAL = "results/joint-gate-292/pre-native/forecast-seal.json"


def rational(g, doubles):
    """SiLU-shaped rational R and its derivative, with bounded tanh core."""
    g = np.asarray(g, dtype=np.float64)
    x = g / (2 * 2**doubles)
    n, d = x * (15 + x * x), 15 + 6 * x * x
    raw = n / d
    t = np.clip(raw, -1.0, 1.0)
    dt = ((15 + 3 * x * x) * d - n * 12 * x) / d**2 / (2 * 2**doubles)
    dt = np.where(abs(raw) < 1, dt, 0.0)
    for _ in range(doubles):
        dt = 2 * (1 - t * t) / (1 + t * t) ** 2 * dt
        t = 2 * t / (1 + t * t)
    return 0.5 * g * (1 + t), 0.5 * (1 + t) + 0.5 * g * dt


def proposal(arm, g, false):
    g = np.asarray(g, dtype=np.float64)
    f = false["g"].astype(float)
    phi = false["phi"].astype(float)
    dg = g - f
    ds = {k: false[k] for k in ("d1", "d2", "d3")} if "d1" in false else derivatives(f)
    if arm in ("P", "D", "A"):
        m = 0 if arm == "D" else 2
        r, dr = rational(g, m)
        r0, dr0 = (
            (false["R" + str(m)], false["Rprime" + str(m)])
            if "R" + str(m) in false
            else rational(f, m)
        )
        return phi + (r - r0) + (ds["d1"] - dr0) * dg, dr + ds["d1"] - dr0
    raw = phi + ds["d1"] * dg + ds["d2"] * dg**2 / 2 + ds["d3"] * dg**3 / 6
    slope = ds["d1"] + ds["d2"] * dg + ds["d3"] * dg**2 / 2
    if arm == "B":
        low = np.maximum(np.minimum(g, 0), -0.278465)
        high = np.maximum(g, 0)
        return np.clip(raw, low, high), slope  # B derivative is not an attributed quantity.
    assert arm == "C"
    return raw, slope


class Engine:
    def __init__(self, model, case, allow_native=False):
        self.model, self.case, self.allow_native = model, case, allow_native
        self.start = case["position"]
        self.anchors = None
        self.prefixes = None
        self.layers = len(model.model.layers)
        self.sites = sorted({0, self.layers // 2, self.layers - 1})

    def run(self, arm, zero=False):
        assert arm in (*ARMS, "F", "T") and (arm != "T" or self.allow_native)
        assert arm in ("F", "T") or self.anchors is not None
        current = [{} for _ in range(self.layers)]
        handles = []
        predicted = arm in ARMS
        start = self.start
        checks = []

        def check(name, a, b, bound=0.0):
            error = float(np.max(abs(np.asarray(a, float) - np.asarray(b, float))))
            checks.append(
                dict(
                    name=name,
                    error=error,
                    bound=bound,
                    passes=bool(np.isfinite(error) and error <= bound),
                )
            )

        for j, layer in enumerate(self.model.model.layers):

            def gate(module, args, j=j):
                current[j]["g"] = array(args[0][0, start:])
                if not predicted:
                    return None
                changed = args[0].clone()
                changed[0, start:] = torch.tensor(
                    self.anchors[j]["g"], device=changed.device, dtype=changed.dtype
                )
                check(f"{j}_activation_input", array(changed[0, start:]), self.anchors[j]["g"])
                return (changed,)

            def activation(module, args, output, j=j):
                if predicted:
                    check(f"{j}_False_phi_primal", array(output[0, start:]), self.anchors[j]["phi"])
                    phi, sl = proposal(arm, current[j]["g"], self.anchors[j])
                    current[j]["slope"] = sl
                    value = output.clone()
                    value[0, start:] = torch.tensor(
                        phi.astype(np.float32), device=value.device, dtype=value.dtype
                    )
                    current[j]["phi"] = array(value[0, start:])
                    check(f"{j}_phi_install", current[j]["phi"], phi.astype(np.float32))
                    return value
                current[j]["phi"] = array(output[0, start:])
                return None

            def up(module, args, output, j=j):
                current[j]["u"] = array(output[0, start:])

            def product(module, args, j=j):
                observed = array(args[0][0, start:])
                if predicted:
                    expected = current[j]["phi"] * current[j]["u"]
                    check(f"{j}_joint_product", observed, expected, 1e-4)
                if arm == "A":
                    f = self.anchors[j]
                    p = (
                        f["product"].astype(float)
                        + f["u"].astype(float) * (current[j]["phi"].astype(float) - f["phi"])
                        + f["phi"].astype(float) * (current[j]["u"].astype(float) - f["u"])
                    )
                    changed = args[0].clone()
                    changed[0, start:] = torch.tensor(
                        p.astype(np.float32), device=changed.device, dtype=changed.dtype
                    )
                    current[j]["product"] = array(changed[0, start:])
                    check(f"{j}_affine_install", current[j]["product"], p.astype(np.float32))
                    return (changed,)
                current[j]["product"] = observed
                return None

            def write(module, args, output, j=j):
                current[j]["write"] = array(output[0, start:])

            def state(module, args, output, j=j):
                value = output[0] if isinstance(output, tuple) else output
                current[j]["state"] = array(value[0, start:])
                if arm == "F":
                    self.prefixes[j] = value[0, :start].detach().clone()
                if self.prefixes is not None:
                    check(
                        f"{j}_prefix_exact",
                        float(not torch.equal(value[0, :start], self.prefixes[j])),
                        0.0,
                    )
                current[j]["prefix_sha256"] = digest(array(value[0, :start]))

            handles += [
                layer.mlp.act_fn.register_forward_pre_hook(gate),
                layer.mlp.act_fn.register_forward_hook(activation),
                layer.mlp.up_proj.register_forward_hook(up),
                layer.mlp.down_proj.register_forward_pre_hook(product),
                layer.mlp.register_forward_hook(write),
                layer.register_forward_hook(state),
            ]
        if arm == "F":
            self.prefixes = [None] * self.layers
        ids = self.case["false_ids"] if arm == "F" or zero else self.case["true_ids"]
        try:
            with torch.no_grad():
                output = self.model(
                    input_ids=torch.tensor([ids], device=self.model.device),
                    attention_mask=torch.ones(
                        (1, len(ids)), device=self.model.device, dtype=torch.long
                    ),
                    use_cache=False,
                    logits_to_keep=1,
                )
        finally:
            for handle in handles:
                handle.remove()
        data = dict(final=current[-1]["state"], scores=array(output.logits[0, -1, [895, 830]]))
        if arm in ("F", "P", "T") and not zero:
            data.update(
                states=np.stack([r["state"] for r in current]),
                writes=np.stack([r["write"][[0, -1]] for r in current]),
            )
        if arm in ("F", "P") and not zero:
            for name in ("g", "u", "phi", "product"):
                data["witness_" + name] = np.stack([current[j][name][[0, -1]] for j in self.sites])
            data["witness_slope"] = np.stack(
                [
                    (derivatives(current[j]["g"])["d1"] if arm == "F" else current[j]["slope"])[
                        [0, -1]
                    ]
                    for j in self.sites
                ]
            )
        elif predicted and not zero:
            for name in ("g", "u", "phi", "product"):
                data["control_" + name] = current[-1][name][-1]
        if zero:
            assert predicted
            for j, r in enumerate(current):
                for n in ("state", "write", "g", "u", "phi", "product"):
                    check(f"{j}_zero_{n}", r[n], self.anchors[j][n])
            check("zero_scores", data["scores"], self.false_scores)
        if arm == "F":
            for values in current:
                values.update(derivatives(values["g"]))
                for m in (0, 2):
                    values["R" + str(m)], values["Rprime" + str(m)] = rational(values["g"], m)
            self.anchors = current
            self.false_scores = data["scores"].copy()
        meta = dict(
            arm=arm,
            zero=zero,
            checks=checks,
            prefixes=[r["prefix_sha256"] for r in current],
            parent_seals=[
                {n: digest(r[n]) for n in ("g", "u", "phi", "product", "write", "state")}
                for r in current
            ],
            activation_calls=self.layers,
            current_affected_finite_SiLU_coordinates=0 if predicted else None,
            ordinary_true=int(arm == "T"),
            editing_hooks_applied=predicted,
            full_forwards=1,
        )
        assert all(np.isfinite(v).all() for v in data.values())
        return data, meta
