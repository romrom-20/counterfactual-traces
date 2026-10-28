"""TorchDouble independent operator arithmetic, power-range and vector checks."""

import numpy as np
import torch

from .change294 import ARMS, BOUNDS, check


def qualify(w, parent, data):
    rows = []
    af, bf, ae, be = (
        parent[n].astype(float) for n in ("False_a", "False_b", "current_a", "current_b")
    )
    a, b = ae - af, be - bf
    o = parent["current_omega"].astype(float)
    check(rows, "change_coefficient_a", data["change_a"], a, 0.0, 0.0)
    check(rows, "change_coefficient_b", data["change_b"], b, 0.0, 0.0)
    check(rows, "unchanged_target", data["target"], parent["nonlinear"], 0.0, 0.0)
    check(rows, "unchanged_omega", data["omega"], o, 0.0, 0.0)
    wg, wu, wd = (
        torch.tensor(w[n], dtype=torch.float64) for n in ("gate_weight", "up_weight", "down_weight")
    )

    def op(x, aa, bb, transpose=False):
        x = torch.tensor(x, dtype=torch.float64)
        aa = torch.tensor(aa, dtype=torch.float64)[:, None, :]
        bb = torch.tensor(bb, dtype=torch.float64)[:, None, :]
        if transpose:
            z = x @ wd
            return ((z * aa) @ wg + (z * bb) @ wu).numpy()
        return (((x @ wg.T) * aa + (x @ wu.T) * bb) @ wd.T).numpy()

    def orth(x):
        return (
            torch.linalg.qr(torch.tensor(x, dtype=torch.float64).transpose(-1, -2), mode="reduced")[
                0
            ]
            .transpose(-1, -2)
            .numpy()
        )

    check(
        rows,
        "D_is_endpoint_difference",
        op(o, a, b),
        op(o, ae, be) - op(o, af, bf),
        BOUNDS["operator"],
        0.0,
    )
    for arm, aa, bb in [
        ("FALSE_RANGE", af, bf),
        ("ENDPOINT_RANGE", ae, be),
        ("CHANGE_RANGE", a, b),
    ]:
        y0 = op(o, aa, bb)
        q0 = orth(y0)
        q = q0
        z0 = op(q0, aa, bb, True)
        if arm == "CHANGE_RANGE":
            check(rows, "independent_D_y0", data["change_y0"], y0, BOUNDS["operator"], 0.0)
            check(
                rows,
                "independent_D_z0",
                data["change_z0"],
                op(data["change_q0"], aa, bb, True),
                BOUNDS["operator"],
                0.0,
            )
            check(
                rows,
                "DT_is_endpoint_difference",
                op(data["change_q0"], a, b, True),
                op(data["change_q0"], ae, be, True) - op(data["change_q0"], af, bf, True),
                BOUNDS["operator"],
                0.0,
            )
        for step in range(2):
            q = orth(op(orth(z0 if step == 0 else op(q, aa, bb, True)), aa, bb))
        saved = data[arm + "_Q"]
        for row in range(len(saved)):
            distance = np.linalg.norm(saved[row] - (saved[row] @ q[row].T) @ q[row])
            check(rows, arm + "_independent_range_" + str(row), distance, 0.0, BOUNDS["range"], 0.0)
    random = orth(o)
    for row in range(len(o)):
        saved = data["RANDOM_RANGE_Q"][row]
        check(
            rows,
            "random_range_" + str(row),
            np.linalg.norm(saved - (saved @ random[row].T) @ random[row]),
            0.0,
            BOUNDS["range"],
            0.0,
        )
    for arm in ARMS:
        q = data[arm + "_Q"]
        target = data["target"]
        pred = data[arm + "_projected"]
        res = data[arm + "_orthogonal"]
        for row in range(len(q)):
            expected = (torch.tensor(q[row]) @ torch.tensor(target[row])) @ torch.tensor(q[row])
            check(rows, arm + "_independent_projection_" + str(row), pred[row], expected.numpy())
            check(rows, arm + "_orthogonality_" + str(row), q[row] @ q[row].T, np.eye(len(q[row])))
            check(rows, arm + "_residual_" + str(row), pred[row] + res[row], target[row])
            check(
                rows,
                arm + "_pythagorean_" + str(row),
                target[row] @ target[row],
                pred[row] @ pred[row] + res[row] @ res[row],
            )
            check(
                rows,
                arm + "_signed_projection_" + str(row),
                pred[row] @ target[row],
                pred[row] @ pred[row],
            )
    return rows
