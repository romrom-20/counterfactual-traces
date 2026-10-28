"""Independent TorchDouble product calculus and rounded-target bridge checks."""

import numpy as np
import torch

from .drive295 import ARMS, BOUNDS, ingredients


def check(rows, name, x, y, absolute=None, relative=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    error = float(np.max(abs(x - y)))
    bound = (BOUNDS["absolute"] if absolute is None else absolute) + (
        BOUNDS["relative"] if relative is None else relative
    ) * float(np.max(abs(y)))
    rows.append(
        dict(
            name=name, error=error, bound=bound, passes=bool(np.isfinite(error) and error <= bound)
        )
    )


def psi(g, f):
    # Independent value implementation; AD provides slope rather than copied derivative.
    def value(x):
        z = x / 8
        t = torch.clamp((z**3 + 15 * z) / (6 * z**2 + 15), -1.0, 1.0)
        for _ in range(2):
            t = 2 * t / (1 + t * t)
        return x * (1 + t) / 2

    gf = torch.tensor(f["g"], dtype=torch.float64, requires_grad=True)
    rf = value(gf)
    df = torch.autograd.grad(rf.sum(), gf)[0].detach()
    correction = torch.tensor(f["d1"]) - df
    val = torch.tensor(f["phi"]) + value(g) - rf.detach() + correction * (g - gf.detach())
    slope = torch.autograd.grad(val.sum(), g, retain_graph=True)[0]
    return val, slope


def qualify(wd, p, data, old_projection):
    f, dg, du = ingredients(p)
    rows = []
    wg = torch.tensor(wd, dtype=torch.float64)
    ft = {n: torch.tensor(v) for n, v in f.items()}
    products = {}
    for label, t in [("endpoint", 1.0), ("midpoint", 0.5), ("zero", 0.0)]:
        g = torch.tensor(f["g"] + t * dg, requires_grad=True)
        phi, slope = psi(g, f)
        u = torch.tensor(f["u"] + t * du)
        k = (u * slope - ft["u"] * ft["d1"]) * torch.tensor(dg) + (phi - ft["phi"]) * torch.tensor(
            du
        )
        products[label] = k.detach()
        if t == 1:
            ph, s = phi.detach().numpy(), slope.detach().numpy()
            check(rows, "saved_activation_replay", ph.astype(np.float32), p["phi"], 0.0, 0.0)
            check(rows, "saved_slope_replay", s, p["slope"])
            smooth = ft["u"] * (phi.detach() - ft["phi"] - ft["d1"] * torch.tensor(dg)) + (
                phi.detach() - ft["phi"]
            ) * torch.tensor(du)
            check(rows, "independent_smooth_product", data["smooth_product"], smooth.numpy())
            check(rows, "independent_smooth_down", data["smooth_n"], (smooth @ wg.T).numpy())
    check(rows, "zero_kernel", products["zero"].numpy(), np.zeros_like(dg))
    for name, k in [
        ("ENDPOINT_DRIVE", products["endpoint"] / 2),
        ("MIDPOINT_DRIVE", products["midpoint"]),
        ("SIMPSON_DRIVE", (products["endpoint"] + 4 * products["midpoint"]) / 6),
    ]:
        check(rows, "independent_" + name, data[name], (k @ wg.T).numpy())
    check(rows, "endpoint_product", data["endpoint_product"], products["endpoint"].numpy() / 2)
    check(rows, "midpoint_product", data["midpoint_product"], products["midpoint"].numpy())
    check(rows, "unchanged_gate_drive", data["gate_change"], dg, 0.0, 0.0)
    check(rows, "unchanged_up_drive", data["up_change"], du, 0.0, 0.0)
    check(rows, "unchanged_target", data["target"], p["nonlinear"], 0.0, 0.0)
    check(rows, "unchanged294projection", data["CHANGE_RANGE"], old_projection, 0.0, 0.0)
    check(rows, "integral_product_identity", data["integral16_product"], data["smooth_product"])
    check(rows, "integral_product_doubling", data["integral16_product"], data["integral32_product"])
    check(rows, "integral_identity", data["INTEGRAL16"], data["smooth_n"])
    check(rows, "integral_node_doubling", data["INTEGRAL16"], data["integral32"])
    shifted = (
        torch.tensor(p["u"].astype(float)) * torch.tensor(s) - ft["u"] * ft["d1"]
    ) * torch.tensor(np.roll(dg, 1, axis=-1)) + (torch.tensor(ph) - ft["phi"]) * torch.tensor(
        np.roll(du, 1, axis=-1)
    )
    shifted = shifted / 2 @ wg.T
    endpoint = products["endpoint"] / 2 @ wg.T
    norm = torch.linalg.vector_norm(endpoint, dim=-1)
    shifted = (
        shifted
        * (norm / torch.clamp(torch.linalg.vector_norm(shifted, dim=-1), min=1e-300))[:, None]
    )
    check(rows, "independent_shuffled", data["SHUFFLED_DRIVE"], shifted.numpy())
    check(rows, "shuffled_norm", np.linalg.norm(data["SHUFFLED_DRIVE"], axis=-1), norm.numpy())
    n = torch.tensor(data["target"])
    oracle = (
        endpoint
        * ((endpoint * n).sum(-1) / torch.clamp((endpoint * endpoint).sum(-1), min=1e-300))[:, None]
    )
    check(rows, "independent_oracle", data["ORACLE_DIRECTION"], oracle.numpy())
    for row in range(len(dg)):
        scale = max(float(np.max(abs(p[name][row]))) for name in ("write", "False_write", "linear"))
        check(
            rows,
            "rounded_bridge_" + str(row),
            data["smooth_n"][row],
            data["target"][row],
            BOUNDS["bridge_absolute"] + BOUNDS["bridge_relative"] * scale,
            0.0,
        )
        check(
            rows,
            "rounded_bridge_norm_" + str(row),
            np.linalg.norm(data["smooth_n"][row] - data["target"][row]),
            0.0,
            BOUNDS["bridge_norm"] * max(np.linalg.norm(data["target"][row]), 0.001),
            0.0,
        )
        for arm in ARMS:
            check(
                rows,
                arm + "_finite_" + str(row),
                np.all(np.isfinite(data[arm][row])),
                True,
                0.0,
                0.0,
            )
    return rows
