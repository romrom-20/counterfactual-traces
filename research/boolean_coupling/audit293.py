"""Separate CPU64 response reconstruction, smooth FD, record and decision audit."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .audit229 import sha
from .audit272 import read, table
from .audit292 import torch_rational
from .cache272 import save_json
from .core225b import array, load_model
from .run286 import cpu_head
from .run293 import binding
from .source279 import derivatives
from .subspace293 import ARMS, BOUNDS, statistics


def margin(v):
    return float(v[1]) - float(v[0])


def vector(p, t, rel=0.05, cos=0.99):
    v = statistics(p, t)
    return dict(**v, passes=v["norm"] >= 0.001 and v["relative"] <= rel and v["cosine"] >= cos)


def main(run, model=None):
    assert not (run / "audit.json").exists()
    d = binding(run, model)
    st = read(run / "status.json")
    assert st["status"] == "complete" and st["ordinary_checkpoint_true"] == 0
    assert st["raw_bytes"] == sum(e["bytes"] for e in st["cases"]) <= 256 * 1024**2
    assert sha(run / "head.npz") == read(run / "head.json")["arrays_sha256"]
    with np.load(run / "head.npz") as z:
        head = {n: z[n] for n in z.files}
    model = load_model(True) if model is None else model
    checks = []
    native = []
    pairs = []
    locals_ = []
    bank = {}

    def check(name, x, y, bound=0.0, key="", arm=""):
        err = float(np.max(abs(np.asarray(x, float) - np.asarray(y, float))))
        checks.append(
            dict(
                key=key,
                arm=arm,
                name=name,
                error=err,
                bound=float(bound),
                passes=bool(np.isfinite(err) and err <= bound),
            )
        )

    def numeric(name, x, y, key, arm, absolute=None, relative=None):
        bound = (BOUNDS["primitive_abs"] if absolute is None else absolute) + (
            BOUNDS["primitive_rel"] if relative is None else relative
        ) * float(np.max(abs(y)))
        check(name, x, y, bound, key, arm)

    for c, e in zip(d["cases"], st["cases"], strict=True):
        key = c["key"]
        p = run / "cases" / (key + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        with np.load(p) as z:
            a = {n: z[n] for n in z.files}
        m = read(p.with_suffix(".json"))
        bank[key] = a
        for r in m["checks"]:
            checks.append(dict(key=key, arm="reference", **r))
        for arm, meta in m["arms"].items():
            assert (
                meta["no_current_native_SiLU"]
                and meta["engine"]["current_affected_finite_SiLU_coordinates"] == 0
            )
            assert meta["engine"]["activation_calls"] == len(model.model.layers)
            for r in meta["checks"]:
                checks.append(dict(key=key, arm=arm, **r))
        for arm in (*ARMS, "F", "T"):
            numeric(
                "CPU64_head",
                cpu_head(head, a[arm + "_final"]),
                a[arm + "_scores"],
                key,
                arm,
                absolute=0.005,
                relative=0.0,
            )
        sites = sorted({0, len(model.model.layers) // 2, len(model.model.layers) - 1})
        for j in sites:
            layer = model.model.layers[j].mlp
            wg, wu, wd = (
                array(m.weight).astype(float)
                for m in (layer.gate_proj, layer.up_proj, layer.down_proj)
            )
            gf, uf, pf = (a[f"False{j}_" + n].astype(float) for n in ("g", "u", "phi"))
            slopeF = derivatives(gf)["d1"]
            for arm in ARMS:
                pre = f"{arm}_probe{j}_"
                g, u, ph, sl = (a[pre + n].astype(float) for n in ("g", "u", "phi", "slope"))
                lin = a[pre + "linear"].astype(float)
                wp = a[pre + "write"].astype(float)
                n = a[pre + "nonlinear"].astype(float)
                expected = (uf * slopeF * (g - gf) + pf * (u - uf)).astype(np.float32).astype(
                    float
                ) @ wd.T
                numeric(f"{j}_linear_projection", lin, expected, key, arm)
                false_write = a[f"P_probe{j}_False_write"].astype(float)
                check(f"{j}_closure", false_write + lin + n, wp, 1e-10, key, arm)
                # Complete ledger and installed restriction from saved component operands.
                if arm == "P":
                    chosen = n
                elif arm == "L":
                    chosen = np.zeros_like(n)
                elif arm == "F":
                    q = a[f"P_probe{j}_False_Q"]
                    chosen = np.einsum("rkh,rk->rh", q, np.einsum("rkh,rh->rk", q, n))
                else:
                    q = a[pre + "current_Q"]
                    proj = np.einsum("rkh,rk->rh", q, np.einsum("rkh,rh->rk", q, n))
                    if arm == "E":
                        chosen = proj
                    else:
                        o = a[pre + "current_omega"].astype(float)
                        qr = np.stack([np.linalg.qr(x.T, mode="reduced")[0].T for x in o])
                        rnd = np.einsum("rkh,rk->rh", qr, np.einsum("rkh,rh->rk", qr, n))
                        want = np.linalg.norm(proj, axis=1)
                        rn = np.linalg.norm(rnd, axis=1)
                        chosen = np.stack(
                            [
                                np.zeros(wd.shape[0])
                                if want[i] == 0
                                else (rnd[i] * want[i] / rn[i] if rn[i] > 0 else qr[i, 0] * want[i])
                                for i in range(2)
                            ]
                        )
                check(
                    f"{j}_saved_install",
                    a[pre + "installed"],
                    (false_write + lin + chosen).astype(np.float32),
                    0.0,
                    key,
                    arm,
                )
            for arm, family in [
                ("P", "False"),
                ("P", "current"),
                ("E", "current"),
                ("R", "current"),
            ]:
                pre = f"{arm}_probe{j}_"
                bp = pre + family + "_"
                g, u, ph = (
                    a[f"False{j}_" + n].astype(float)
                    if family == "False"
                    else a[pre + n].astype(float)
                    for n in ("g", "u", "phi")
                )
                sl = derivatives(g)["d1"] if family == "False" else a[pre + "slope"]
                aa, bb = a[bp + "a"].astype(float), a[bp + "b"].astype(float)
                numeric(f"{j}_{family}_coefficient_a", aa, u * sl, key, arm)
                numeric(f"{j}_{family}_coefficient_b", bb, ph, key, arm)

                def jac(x, transpose=False, wg=wg, wu=wu, wd=wd):
                    if transpose:
                        z = x @ wd
                        return (z * aa[:, None, :]) @ wg + (z * bb[:, None, :]) @ wu
                    return ((x @ wg.T) * aa[:, None, :] + (x @ wu.T) * bb[:, None, :]) @ wd.T

                numeric(
                    f"{j}_{family}_J", a[bp + "y0"], jac(a[bp + "omega"].astype(float)), key, arm
                )
                numeric(f"{j}_{family}_JT", a[bp + "z0"], jac(a[bp + "q0"], True), key, arm)

                def orth(x):
                    return np.stack([np.linalg.qr(v.T, mode="reduced")[0].T for v in x])

                q = orth(jac(a[bp + "omega"].astype(float)))
                for _ in range(2):
                    q = orth(jac(orth(jac(q, True))))
                saved = a[bp + "Q"]
                for row in range(2):
                    check(
                        f"{j}_{family}_Qorth_{row}",
                        saved[row] @ saved[row].T,
                        np.eye(len(saved[row])),
                        1e-10,
                        key,
                        arm,
                    )
                    # Direct difference avoids subtractive cancellation in projector distance.
                    err = np.linalg.norm(saved[row] - ((saved[row] @ q[row].T) @ q[row]))
                    check(f"{j}_{family}_Qrange_{row}", err, 0.0, BOUNDS["range"], key, arm)
                    v = a[bp + "omega"][row, 0].astype(float)
                    dg = wg @ v
                    du = wu @ v
                    gt = torch.tensor(g[row], dtype=torch.float64)
                    ut = torch.tensor(u[row], dtype=torch.float64)
                    pt = torch.tensor(ph[row], dtype=torch.float64)
                    dgt = torch.tensor(dg, dtype=torch.float64)
                    dut = torch.tensor(du, dtype=torch.float64)
                    w = torch.tensor(wd, dtype=torch.float64)
                    lam = (
                        torch.tensor(slopeF[row], dtype=torch.float64)
                        - torch.autograd.functional.jvp(
                            lambda x: torch_rational(x),
                            torch.tensor(gf[row], dtype=torch.float64),
                            torch.ones_like(gt),
                        )[1]
                    )

                    def finite(t):
                        delta = (
                            F.silu(gt + t * dgt) - F.silu(gt)
                            if family == "False"
                            else torch_rational(gt + t * dgt) - torch_rational(gt) + lam * t * dgt
                        )
                        return w @ ((pt + delta) * (ut + t * dut))

                    ideal = wd @ (u[row] * sl[row] * dg + ph[row] * du)
                    ad = array(
                        torch.autograd.functional.jvp(
                            finite,
                            torch.tensor(0.0, dtype=torch.float64),
                            torch.tensor(1.0, dtype=torch.float64),
                        )[1]
                    )
                    numeric(
                        f"{j}_{family}_local_AD_{row}",
                        ad,
                        ideal,
                        key,
                        arm,
                        absolute=1e-6,
                        relative=1e-5,
                    )
                    for step in (1e-4, 5e-5):
                        fd = array((finite(step) - finite(-step)) / (2 * step))
                        numeric(
                            f"{j}_{family}_FD_{row}_{step}",
                            fd,
                            ideal,
                            key,
                            arm,
                            absolute=1e-6,
                            relative=1e-5,
                        )
            del wg, wu, wd
        for arm in ARMS:
            rec = vector(
                a[arm + "_final"][1:].astype(float) - a["F_final"][1:],
                a["T_final"][1:].astype(float) - a["F_final"][1:],
            )
            ve = vector(
                a[arm + "_final"][-1].astype(float) - a["F_final"][-1],
                a["T_final"][-1].astype(float) - a["F_final"][-1],
            )
            err = abs(margin(a[arm + "_scores"]) - margin(a["T_scores"]))
            native.append(
                dict(
                    key=key,
                    **c["group"],
                    arm=arm,
                    receiving_relative=rec["relative"],
                    receiving_cosine=rec["cosine"],
                    verdict_relative=ve["relative"],
                    verdict_cosine=ve["cosine"],
                    margin_error=err,
                    passes=rec["passes"] and ve["passes"] and err <= 0.1,
                )
            )
        for r in m["arms"]["P"]["local"]:
            locals_.append(dict(key=key, **c["group"], **r))
        print("293 audit", len(bank), "/", len(d["cases"]), flush=True)
    for block in sorted({c["group"]["block"] for c in d["cases"]}):
        cs = sorted(
            [c for c in d["cases"] if c["group"]["block"] == block],
            key=lambda c: c["group"]["schema"],
        )
        assert len(cs) == 2 and {c["group"]["schema"] for c in cs} == {"generic", "descriptive"}
        a, b = (bank[c["key"]] for c in cs)
        for arm in ARMS:

            def delta(pack, arm, field):
                return pack[arm + "_" + field].astype(float) - pack["F_" + field].astype(float)

            source = vector(
                delta(a, arm, "first_source") - delta(b, arm, "first_source"),
                delta(a, "T", "first_source") - delta(b, "T", "first_source"),
                0.1,
                0.995,
            )
            verdict = (
                vector(
                    (delta(a, arm, "final") - delta(b, arm, "final"))[-1],
                    (delta(a, "T", "final") - delta(b, "T", "final"))[-1],
                    0.1,
                    0.995,
                )
                if a[arm + "_final"].shape == b[arm + "_final"].shape
                else vector(
                    delta(a, arm, "final")[-1] - delta(b, arm, "final")[-1],
                    delta(a, "T", "final")[-1] - delta(b, "T", "final")[-1],
                    0.1,
                    0.995,
                )
            )
            pred = (
                margin(a[arm + "_scores"])
                - margin(a["F_scores"])
                - margin(b[arm + "_scores"])
                + margin(b["F_scores"])
            )
            target = (
                margin(a["T_scores"])
                - margin(a["F_scores"])
                - margin(b["T_scores"])
                + margin(b["F_scores"])
            )
            pairs.append(
                dict(
                    block=block,
                    arm=arm,
                    source_relative=source["relative"],
                    source_cosine=source["cosine"],
                    verdict_relative=verdict["relative"],
                    verdict_cosine=verdict["cosine"],
                    prediction=pred,
                    target=target,
                    error=abs(pred - target),
                    passes=source["passes"] and verdict["passes"] and abs(pred - target) <= 0.1,
                )
            )
    tables = dict(checks=checks, native=native, pairs=pairs, local=locals_)
    for name, rr in tables.items():
        table(run / (name + ".csv"), rr)
    failed = [r for r in checks if not r["passes"]]
    outcomes = {}
    for arm in ARMS:
        rr = [r for r in native if r["arm"] == arm]
        pr = [r for r in pairs if r["arm"] == arm]
        layouts = {
            b: sum(r["passes"] for r in rr if r["block"] == b)
            for b in sorted({r["block"] for r in rr})
        }
        n = sum(layouts.values())
        pn = sum(r["passes"] for r in pr)
        outcomes[arm] = dict(
            native=n,
            total=len(rr),
            layouts=layouts,
            pairs=pn,
            pair_total=len(pr),
            sufficient=n >= 6 and min(layouts.values()) >= 1 and pn >= 3,
        )
    report = dict(
        experiment=293,
        status="failed" if failed else "passed",
        retrospective=True,
        ordinary_checkpoint_true=0,
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"),
        source_sha256=sha(Path(__file__)),
        required_checks=len(checks),
        failed_checks=len(failed),
        outcomes=outcomes,
        primary_pass=outcomes["E"]["sufficient"],
        adaptation_specific=outcomes["E"]["sufficient"]
        and outcomes["F"]["native"] <= 2
        and outcomes["R"]["native"] <= 2,
        general_question_B_solved=False,
        learned_circuit_identified=False,
        csv_sha256={n + ".csv": sha(run / (n + ".csv")) for n in tables},
    )
    save_json(run / "audit.json", report)
    print(report, flush=True)
    assert not failed, "Numerical qualification failed; do not interpret scientific outcomes"
    if not d["synthetic"]:
        assert outcomes["P"]["native"] == 8 and outcomes["P"]["pairs"] == 4, "P calibration failed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    main(p.parse_args().run.resolve())
