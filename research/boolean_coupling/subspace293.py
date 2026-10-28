"""False/adaptive randomized MLP response ranges; no whole-model autograd."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as TF

from .core225b import array
from .joint292 import SOURCES as PREVIOUS
from .joint292 import Engine, proposal

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/subspace293.py",
    "research/boolean_coupling/run293.py",
    "research/boolean_coupling/audit293.py",
    "research/boolean_coupling/preflight293.py",
    "docs/experiments/293-response-subspace.md",
)
ARMS = ("P", "L", "F", "E", "R")
BOUNDS = dict(
    orthogonal=1e-10,
    closure=1e-10,
    replay=0.0,
    primitive_abs=1e-4,
    primitive_rel=1e-5,
    FD_abs=1e-6,
    FD_rel=1e-5,
    head=0.005,
    range=1e-3,
)
PRIMARY = dict(
    rank=32,
    power_steps=2,
    native_relative=0.05,
    native_cosine=0.99,
    margin=0.1,
    pair_relative=0.1,
    pair_cosine=0.995,
    norm=0.001,
    population=6,
    per_layout=1,
    pairs=3,
)


def check(rows, name, x, y, bound=0.0):
    err = float(np.max(abs(np.asarray(x, float) - np.asarray(y, float))))
    rows.append(
        dict(
            name=name, error=err, bound=float(bound), passes=bool(np.isfinite(err) and err <= bound)
        )
    )


def orth(x, checks, label):
    qs = []
    for i, z in enumerate(np.asarray(x, float)):
        q, r = np.linalg.qr(z.T, mode="reduced")
        diag = abs(np.diag(r))
        assert min(diag) > 1e-8 * max(diag), ("Rank loss", label, i)
        check(checks, label + "_orth_" + str(i), q.T @ q, np.eye(q.shape[1]), BOUNDS["orthogonal"])
        qs.append(q.T)
    return np.stack(qs)


def omega(key, j, rows, hidden):
    k = min(32, hidden // 2)
    return np.stack(
        [
            np.random.default_rng(
                int.from_bytes(hashlib.sha256(f"{key}/{j}/{row}/293".encode()).digest()[:8], "big")
            )
            .normal(0, hidden**-0.5, (k, hidden))
            .astype(np.float32)
            for row in range(rows)
        ]
    )


def basis(layer, a, b, o, checks, label):
    device = layer.gate_proj.weight.device
    aa = torch.tensor(a.astype(np.float32), device=device)[:, None, :]
    bb = torch.tensor(b.astype(np.float32), device=device)[:, None, :]

    def japply(x, transpose=False):
        t = torch.tensor(x.astype(np.float32), device=device)
        with torch.no_grad():
            if transpose:
                d = TF.linear(t, layer.down_proj.weight.T)
                v = TF.linear(d * aa, layer.gate_proj.weight.T) + TF.linear(
                    d * bb, layer.up_proj.weight.T
                )
            else:
                v = TF.linear(
                    aa * TF.linear(t, layer.gate_proj.weight)
                    + bb * TF.linear(t, layer.up_proj.weight),
                    layer.down_proj.weight,
                )
        return array(v)

    y = japply(o)
    q0 = orth(y, checks, label + "_q0")
    z0 = japply(q0, True)
    q = q0
    for step in range(2):
        z = z0 if step == 0 else japply(q, True)
        q = orth(
            japply(orth(z, checks, label + "_input" + str(step))),
            checks,
            label + "_out" + str(step),
        )
    return q, dict(omega=o, a=a.astype(np.float32), b=b.astype(np.float32), y0=y, q0=q0, z0=z0, Q=q)


def project(q, n):
    return np.einsum("rkh,rk->rh", q, np.einsum("rkh,rh->rk", q, n))


def statistics(p, t):
    p, t = np.asarray(p, float).ravel(), np.asarray(t, float).ravel()
    pp, tt, pt, ee = float(p @ p), float(t @ t), float(p @ t), float((p - t) @ (p - t))
    return dict(
        prediction_squared_norm=pp,
        target_squared_norm=tt,
        prediction_target_dot=pt,
        error_squared_norm=ee,
        norm=tt**0.5,
        relative=ee**0.5 / max(tt**0.5, 1e-30),
        cosine=pt / max((pp * tt) ** 0.5, 1e-30),
        signed_target_fraction=pt / max(tt, 1e-30),
    )


class Restricted:
    def __init__(self, model, case):
        self.model, self.case = model, case
        self.engine = Engine(model, case)
        self.false, self.false_meta = self.engine.run("F")
        self.fixed = {}
        self.fixed_witness = {}
        self.sites = self.engine.sites

    def run(self, arm, zero=False):
        assert arm in ARMS
        checks = []
        captured = {}
        probes = {}
        local = []
        handles = []
        start = self.case["position"]
        tail = len(self.case["false_ids"]) - start
        h = self.model.config.hidden_size
        k = min(32, h // 2)
        counts = dict(
            full_forwards=1, extra_full_down=0, J_calls=0, JT_calls=0, QR_batches=0, QR_matrices=0
        )
        for j, block in enumerate(self.model.model.layers):
            layer = block.mlp
            f = self.engine.anchors[j]

            def gate(module, args, j=j):
                captured[j] = {"g": array(args[0][0, start:])}

            def up(module, args, output, j=j):
                captured[j]["u"] = array(output[0, start:])

            def write(module, args, output, j=j, layer=layer, f=f):
                c = captured[j]
                g, u = c["g"], c["u"]
                ph, sl = proposal("P", g, f)
                ph = ph.astype(np.float32)
                dp = f["u"].astype(float) * f["d1"] * (g.astype(float) - f["g"]) + f["phi"].astype(
                    float
                ) * (u.astype(float) - f["u"])
                full = torch.zeros(
                    (1, len(self.case["false_ids"]), layer.down_proj.in_features),
                    device=output.device,
                    dtype=output.dtype,
                )
                full[0, start:] = torch.tensor(dp.astype(np.float32), device=output.device)
                with torch.no_grad():
                    lin = array(TF.linear(full, layer.down_proj.weight)[0, start:])
                counts["extra_full_down"] += 1
                wp = array(output[0, start:])
                n = wp.astype(float) - f["write"].astype(float) - lin.astype(float)
                check(
                    checks,
                    f"{j}_closure",
                    f["write"].astype(float) + lin + n,
                    wp,
                    BOUNDS["closure"],
                )
                o = omega(self.case["key"], j, tail, h)
                if j not in self.fixed:
                    qf, wf = basis(
                        layer,
                        f["u"].astype(float) * f["d1"],
                        f["phi"].astype(float),
                        o,
                        checks,
                        f"{j}_False",
                    )
                    self.fixed[j], self.fixed_witness[j] = qf, wf
                    counts["J_calls"] += 3
                    counts["JT_calls"] += 2
                    counts["QR_batches"] += 5
                    counts["QR_matrices"] += 5 * tail
                qf = self.fixed[j]
                qe = None
                we = None
                if arm in ("P", "E", "R") and not zero:
                    qe, we = basis(
                        layer, u.astype(float) * sl, ph.astype(float), o, checks, f"{j}_current"
                    )
                    counts["J_calls"] += 3
                    counts["JT_calls"] += 2
                    counts["QR_batches"] += 5
                    counts["QR_matrices"] += 5 * tail
                if zero:
                    check(checks, f"{j}_zero_remainder", n, np.zeros_like(n))
                    chosen = np.zeros_like(n)
                elif arm == "L":
                    chosen = np.zeros_like(n)
                elif arm == "F":
                    chosen = project(qf, n)
                elif arm == "E":
                    chosen = project(qe, n)
                elif arm == "R":
                    qr = orth(o, checks, f"{j}_random")
                    counts["QR_batches"] += 1
                    counts["QR_matrices"] += tail
                    rnd = project(qr, n)
                    wanted = np.linalg.norm(project(qe, n), axis=1)
                    rn = np.linalg.norm(rnd, axis=1)
                    chosen = np.stack(
                        [
                            np.zeros(h)
                            if wanted[i] == 0
                            else (rnd[i] * wanted[i] / rn[i] if rn[i] > 0 else qr[i, 0] * wanted[i])
                            for i in range(tail)
                        ]
                    )
                    check(checks, f"{j}_random_norm", np.linalg.norm(chosen, axis=1), wanted, 1e-10)
                else:
                    chosen = n
                if arm == "P" and not zero:
                    for name, q in [("F", qf), ("E", qe)]:
                        selected = project(q, n)
                        for scope, row in [
                            ("source", 0),
                            ("receiving", slice(1, None)),
                            ("verdict", -1),
                        ]:
                            local.append(
                                dict(
                                    layer=j + 1,
                                    basis=name,
                                    scope=scope,
                                    eligible=bool(np.linalg.norm(n[row]) >= 0.001),
                                    **statistics(selected[row], n[row]),
                                )
                            )
                installed = (f["write"].astype(float) + lin.astype(float) + chosen).astype(
                    np.float32
                )
                if arm == "P":
                    installed = wp
                value = output.clone()
                value[0, start:] = torch.tensor(installed, device=output.device)
                check(checks, f"{j}_install", array(value[0, start:]), installed)
                check(checks, f"{j}_prefix", array(value[0, :start]), array(output[0, :start]))
                if j in self.sites and not zero:
                    ix = [0, tail - 1]
                    prefix = f"probe{j}_"
                    for name, v in dict(
                        g=g,
                        u=u,
                        phi=ph,
                        slope=sl,
                        linear=lin,
                        write=wp,
                        nonlinear=n,
                        installed=installed,
                    ).items():
                        probes[prefix + name] = v[ix]
                    if we is not None:
                        for name, v in we.items():
                            probes[prefix + "current_" + name] = v[ix]
                    if arm == "P":
                        probes[prefix + "False_write"] = f["write"][ix]
                        for name, v in self.fixed_witness[j].items():
                            probes[prefix + "False_" + name] = v[ix]
                return value

            handles.extend(
                [
                    layer.act_fn.register_forward_pre_hook(gate),
                    layer.up_proj.register_forward_hook(up),
                    layer.register_forward_hook(write),
                ]
            )
        try:
            data, meta = self.engine.run("P", zero=zero)
        finally:
            for handle in handles:
                handle.remove()
        checks.extend(meta["checks"])
        result = dict(final=data["final"], scores=data["scores"])
        if not zero:
            result["first_source"] = data["states"][0, 0]
            result.update(probes)
        return result, dict(
            arm=arm,
            zero=zero,
            checks=checks,
            counts=counts,
            local=local,
            no_current_native_SiLU=True,
            engine=meta,
            rank=k,
        )
