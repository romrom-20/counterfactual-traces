"""Matched response and response-change ranges at saved293 P probe inputs."""

from pathlib import Path

import numpy as np

from .subspace293 import SOURCES as PREVIOUS
from .subspace293 import statistics

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/change294.py",
    "research/boolean_coupling/run294.py",
    "research/boolean_coupling/audit294.py",
    "research/boolean_coupling/preflight294.py",
    "docs/experiments/294-response-change.md",
)
ARMS = ("FALSE_RANGE", "ENDPOINT_RANGE", "CHANGE_RANGE", "RANDOM_RANGE")
PRIMARY = dict(
    rank=32,
    power_steps=2,
    norm=0.001,
    improvement_factor=0.5,
    relative=0.25,
    cosine=0.95,
    population=36,
    per_scope=18,
)
BOUNDS = dict(operator=1e-8, range=1e-6, original_range=1e-3, absolute=1e-10, relative=1e-11)


def check(rows, name, x, y, absolute=None, relative=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    err = float(np.max(abs(x - y)))
    scale = float(np.max(abs(y)))
    bound = (BOUNDS["absolute"] if absolute is None else absolute) + (
        BOUNDS["relative"] if relative is None else relative
    ) * scale
    rows.append(
        dict(name=name, error=err, bound=bound, passes=bool(np.isfinite(err) and err <= bound))
    )


def orth(x):
    result = []
    for a in x:
        q, r = np.linalg.qr(a.T, mode="reduced")
        diag = abs(np.diag(r))
        assert min(diag) > 1e-8 * max(diag), "Rank loss; preserve stop"
        result.append(q.T)
    return np.stack(result)


def apply(w, a, b, x, transpose=False):
    if transpose:
        v = x @ w["down_weight"]
        return (v * a[:, None, :]) @ w["gate_weight"] + (v * b[:, None, :]) @ w["up_weight"]
    return ((x @ w["gate_weight"].T) * a[:, None, :] + (x @ w["up_weight"].T) * b[:, None, :]) @ w[
        "down_weight"
    ].T


def basis(w, a, b, o):
    y0 = apply(w, a, b, o)
    q0 = orth(y0)
    z0 = apply(w, a, b, q0, True)
    q = q0
    for step in range(2):
        q = orth(apply(w, a, b, orth(z0 if step == 0 else apply(w, a, b, q, True))))
    return q, dict(y0=y0, q0=q0, z0=z0)


def project(q, n):
    return np.einsum("rkh,rk->rh", q, np.einsum("rkh,rh->rk", q, n))


def evaluate(w, parent):
    af, bf, ae, be = (
        parent[n].astype(float) for n in ("False_a", "False_b", "current_a", "current_b")
    )
    o = parent["current_omega"].astype(float)
    n = parent["nonlinear"].astype(float)
    assert np.array_equal(parent["False_omega"], parent["current_omega"])
    checks = []
    out = dict(target=n, omega=o, change_a=ae - af, change_b=be - bf)
    for arm, a, b in [
        ("FALSE_RANGE", af, bf),
        ("ENDPOINT_RANGE", ae, be),
        ("CHANGE_RANGE", ae - af, be - bf),
    ]:
        q, trace = basis(w, a, b, o)
        out[arm + "_Q"] = q
        if arm == "CHANGE_RANGE":
            out.update({"change_" + k: v for k, v in trace.items()})
        if arm != "CHANGE_RANGE":
            saved = parent["False_Q" if arm == "FALSE_RANGE" else "current_Q"]
            for row in range(len(n)):
                distance = np.linalg.norm(q[row] - ((q[row] @ saved[row].T) @ saved[row]))
                check(
                    checks,
                    arm + "_original_span_" + str(row),
                    distance,
                    0.0,
                    BOUNDS["original_range"],
                    0.0,
                )
    out["RANDOM_RANGE_Q"] = orth(o)
    for arm in ARMS:
        q = out[arm + "_Q"]
        pred = project(q, n)
        out[arm + "_projected"] = pred
        out[arm + "_orthogonal"] = n - pred
        for row in range(len(n)):
            check(checks, arm + "_orthonormal_" + str(row), q[row] @ q[row].T, np.eye(len(q[row])))
            check(
                checks,
                arm + "_closure_" + str(row),
                pred[row] + out[arm + "_orthogonal"][row],
                n[row],
            )
    return out, checks


def records(key, layer, data):
    result = []
    for row, scope in enumerate(("source", "verdict")):
        stats = {
            arm: statistics(data[arm + "_projected"][row], data["target"][row]) for arm in ARMS
        }
        for arm, v in stats.items():
            result.append(
                dict(
                    key=key, layer=layer + 1, scope=scope, arm=arm, eligible=v["norm"] >= 0.001, **v
                )
            )
    return result


def decisions(rows):
    index = {(r["key"], r["layer"], r["scope"], r["arm"]): r for r in rows}
    out = []
    for key, layer, scope, _ in index:
        if _ != "CHANGE_RANGE":
            continue
        d = index[key, layer, scope, "CHANGE_RANGE"]
        f = index[key, layer, scope, "FALSE_RANGE"]
        e = index[key, layer, scope, "ENDPOINT_RANGE"]
        best = min(f["relative"], e["relative"])
        eligible = d["norm"] >= 0.001
        out.append(
            dict(
                key=key,
                layer=layer,
                scope=scope,
                norm=d["norm"],
                best_endpoint_error=best,
                change_error=d["relative"],
                change_cosine=d["cosine"],
                improvement=bool(eligible and d["relative"] <= 0.5 * best),
                vector_account=bool(eligible and d["relative"] <= 0.25 and d["cosine"] >= 0.95),
            )
        )

    def population(field):
        scopes = {s: sum(r[field] for r in out if r["scope"] == s) for s in ("source", "verdict")}
        n = sum(scopes.values())
        return dict(
            passes=n,
            total=len(out),
            scopes=scopes,
            population_pass=n >= 36 and min(scopes.values()) >= 18,
        )

    return out, dict(
        H1_improvement=population("improvement"), H2_vector_account=population("vector_account")
    )
