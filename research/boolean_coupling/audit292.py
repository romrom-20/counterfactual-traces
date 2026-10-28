"""Independent primitive qualification, replay, and sealed native-vector scoring."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .audit229 import sha
from .audit272 import read, table
from .audit287 import metrics
from .cache272 import save_json
from .core225b import array, load_model, observe
from .joint292 import ARMS, BOUNDS, ROOT, Engine, proposal
from .qualify287b import vector_stats
from .run285 import release
from .run286 import cpu_head, head_data
from .run292 import binding, entry
from .source279 import derivatives


def torch_rational(x, m=2):
    t = x / (2 ** (m + 1))
    t = torch.clamp((t**3 + 15 * t) / (6 * t * t + 15), -1.0, 1.0)
    for _ in range(m):
        t = 2 * t / (1 + t * t)
    return x * (1 + t) / 2


def primitive_rows(f, p):
    """No finite edited SiLU: only False SiLU derivatives and rational functions."""
    gf = f["witness_g"].astype(float)
    gp = p["witness_g"].astype(float)
    phi = f["witness_phi"].astype(float)
    result = []

    def check(name, actual, expected, bound):
        error = float(np.max(abs(np.asarray(actual, float) - np.asarray(expected, float))))
        result.append(
            dict(
                name=name,
                error=error,
                bound=bound,
                passes=bool(np.isfinite(error) and error <= bound),
            )
        )

    with torch.enable_grad():
        base = torch.tensor(gf, dtype=torch.float64, requires_grad=True)
        d1 = torch.autograd.grad(F.silu(base).sum(), base, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), base, create_graph=True)[0]
        d3 = torch.autograd.grad(d2.sum(), base)[0]
        ds = derivatives(gf)
        for name, actual in zip(("d1", "d2", "d3"), (d1, d2, d3), strict=True):
            check("False_" + name, ds[name], array(actual), 1e-8)
        rbase = torch_rational(base)
        rprime = torch.autograd.grad(rbase.sum(), base)[0].detach()
        f_d1 = d1.detach()
        const_phi = torch.tensor(phi, dtype=torch.float64)

        def independent(x):
            return (
                const_phi
                + torch_rational(x)
                - rbase.detach()
                + (f_d1 - rprime) * (x - base.detach())
            )

        false = dict(g=gf.astype(np.float32), phi=f["witness_phi"])
        for label, point in (("False", gf), ("forecast", gp)):
            x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            value = independent(x)
            slope = torch.autograd.grad(value.sum(), x)[0]
            expect, dslope = proposal("P", point, false)
            check(label + "_rational_value", expect, array(value), 1e-8)
            check(label + "_rational_AD", dslope, array(slope), BOUNDS["derivative_AD"])
            for step in (1e-4, 5e-5):
                with torch.no_grad():
                    fd = (independent(x.detach() + step) - independent(x.detach() - step)) / (
                        2 * step
                    )
                check(
                    label + "_rational_FD_" + str(step), dslope, array(fd), BOUNDS["derivative_FD"]
                )
        check("False_value_anchor", proposal("P", gf, false)[0], phi, 0.0)
        check("False_slope_anchor", proposal("P", gf, false)[1], ds["d1"], 1e-12)
    check(
        "installed_primary_phi",
        p["witness_phi"],
        proposal("P", gp, false)[0].astype(np.float32),
        0.0,
    )
    check("recorded_primary_slope", p["witness_slope"], proposal("P", gp, false)[1], 0.0)
    return result


def load_forecasts(run, d):
    st = read(run / "forecast-status.json")
    assert st["status"] == "complete" and st["ordinary_true"] == 0
    assert [e["key"] for e in st["cases"]] == [c["key"] for c in d["cases"]]
    assert st["raw_bytes"] == sum(e["bytes"] for e in st["cases"]) <= 256 * 1024**2
    assert st["counts"] == dict(
        full_F=len(d["cases"]),
        full_forecast=5 * len(d["cases"]),
        full_Fzero=5 * len(d["cases"]),
        affected_current_finite_SiLU=0,
    )
    assert sha(run / "head.npz") == read(run / "head.json")["arrays_sha256"]
    with np.load(run / "head.npz") as z:
        head = {n: z[n] for n in z.files}
    return st, head


def qualification(run, model=None):
    assert not (run / "forecast-audit.json").exists() and not (run / "native-started.json").exists()
    d = binding(run, model)
    st, head = load_forecasts(run, d)
    model = load_model(True) if model is None else model
    assert all(np.array_equal(v, head[n]) for n, v in head_data(model).items())
    checks, primitives, heads = [], [], []
    for case, e in zip(d["cases"], st["cases"], strict=True):
        key = case["key"]
        a = entry(run, "forecasts", e)
        m = read(run / "forecasts" / (key + ".json"))
        assert not any(n.startswith("T_") for n in a) and m["ordinary_true"] == 0
        assert set(m["arms"]) == {"F", *ARMS, *(arm + "0" for arm in ARMS)}
        layers = a["F_states"].shape[0]
        for name, meta in m["arms"].items():
            assert meta["ordinary_true"] == 0 and meta["activation_calls"] == layers
            if name != "F":
                assert (
                    meta["editing_hooks_applied"]
                    and meta["current_affected_finite_SiLU_coordinates"] == 0
                )
                assert meta["prefixes"] == m["arms"]["F"]["prefixes"]
            for row in meta["checks"]:
                assert row["bound"] in (0.0, 1e-4)
                assert row["passes"] == (row["error"] <= row["bound"])
                checks.append(dict(key=key, arm=name, **row))
        assert sum(len(meta["checks"]) for meta in m["arms"].values()) == 83 * layers + 5
        f = {n[2:]: v for n, v in a.items() if n.startswith("F_")}
        p = {n[2:]: v for n, v in a.items() if n.startswith("P_")}
        for row in primitive_rows(f, p):
            primitives.append(dict(key=key, **row))

        def check(name, x, y, bound=0.0):
            error = float(np.max(abs(np.asarray(x, float) - np.asarray(y, float))))
            checks.append(
                dict(
                    key=key,
                    arm="audit",
                    name=name,
                    error=error,
                    bound=bound,
                    passes=bool(np.isfinite(error) and error <= bound),
                )
            )

        check("False_observed_slope", f["witness_slope"], derivatives(f["witness_g"])["d1"])
        for arm in ARMS:
            if arm == "P":
                g, u, phi, product = (
                    a[arm + "_witness_" + n] for n in ("g", "u", "phi", "product")
                )
                af = dict(g=f["witness_g"], phi=f["witness_phi"])
                fup, fproduct = f["witness_u"], f["witness_product"]
            else:
                g, u, phi, product = (
                    a[arm + "_control_" + n] for n in ("g", "u", "phi", "product")
                )
                af = dict(g=f["witness_g"][-1, -1], phi=f["witness_phi"][-1, -1])
                fup, fproduct = f["witness_u"][-1, -1], f["witness_product"][-1, -1]
            ph = proposal(arm, g, af)[0].astype(np.float32)
            check(arm + "_phi_formula", phi, ph)
            expected = (
                (
                    fproduct.astype(float)
                    + fup.astype(float) * (ph.astype(float) - af["phi"])
                    + af["phi"].astype(float) * (u.astype(float) - fup)
                ).astype(np.float32)
                if arm == "A"
                else ph * u
            )
            check(arm + "_product_formula", product, expected, 1e-4)
        for arm in ("F", *ARMS):
            err = float(np.max(abs(cpu_head(head, a[arm + "_final"]) - a[arm + "_scores"])))
            heads.append(
                dict(
                    key=key, arm=arm, name="CPU64_head", error=err, bound=0.005, passes=err <= 0.005
                )
            )
        # Independently execute F/P and P's zero edit, never a finite edited SiLU.
        replay = Engine(model, case)
        for arm in ("F", "P"):
            again, meta = replay.run(arm)
            assert meta["parent_seals"] == m["arms"][arm]["parent_seals"]
            for n, v in again.items():
                check("replay_" + arm + "_" + n, v, a[arm + "_" + n])
        _, zm = replay.run("P", zero=True)
        assert all(r["passes"] for r in zm["checks"])
        release(model)
        print("292 qualification", len({x["key"] for x in heads}), "/", len(d["cases"]), flush=True)
    tables = dict(forecast_checks=checks, primitive_checks=primitives, head_checks=heads)
    for n, rows in tables.items():
        table(run / (n + ".csv"), rows)
    allrows = [r for rows in tables.values() for r in rows]
    failed = [r for r in allrows if not r["passes"]]
    audit = dict(
        experiment=292,
        status="failed" if failed else "passed",
        ordinary_true=0,
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "forecast-status.json"),
        source_sha256=sha(ROOT / "research/boolean_coupling/audit292.py"),
        required_checks=len(allrows),
        failed_checks=len(failed),
        counts=dict(full_F=len(d["cases"]), full_P=len(d["cases"]), full_Pzero=len(d["cases"])),
        csv_sha256={n + ".csv": sha(run / (n + ".csv")) for n in tables},
        general_question_B_solved=False,
    )
    save_json(run / "forecast-audit.json", audit)
    assert not failed, "Forecast qualification failed; no seal or native measurement"
    print(audit, flush=True)


def scored(pred, target, relative, cosine):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    met = metrics(pred, target)
    return dict(
        **met,
        **vector_stats(pred, target),
        passes=bool(
            met["norm"] >= 0.001 and met["relative"] <= relative and met["cosine"] >= cosine
        ),
    )


def outcome(run, model=None):
    assert not (run / "native-audit.json").exists()
    d = binding(run, model)
    fst, head = load_forecasts(run, d)
    st = read(run / "native-status.json")
    assert st["status"] == "complete" and st["ordinary_true"] == len(d["cases"])
    assert st["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    assert st["native_started_sha256"] == sha(run / "native-started.json")
    assert st["raw_bytes"] == sum(e["bytes"] for e in st["cases"])
    assert fst["raw_bytes"] + st["raw_bytes"] <= 256 * 1024**2
    model = load_model(True) if model is None else model
    checks, bank = [], {}
    for case, fe, te in zip(d["cases"], fst["cases"], st["cases"], strict=True):
        assert fe["key"] == te["key"] == case["key"]
        a = entry(run, "forecasts", fe)
        t = entry(run, "native", te)
        key = case["key"]
        meta = read(run / "native" / (key + ".json"))["arm"]
        assert not meta["editing_hooks_applied"] and meta["ordinary_true"] == 1
        fmeta = read(run / "forecasts" / (key + ".json"))["arms"]["F"]
        assert meta["prefixes"] == fmeta["prefixes"]
        # Ordinary unhooked full native replay; no approximation policy active.
        states, _, scores = observe(model, torch.tensor([case["true_ids"]], device=model.device))
        for name, x, y, bound in [
            (
                "unhooked_True_states",
                np.stack([array(s[0, case["position"] :]) for s in states]),
                t["states"],
                0.0,
            ),
            ("unhooked_True_scores", array(scores), t["scores"], 0.0),
            ("CPU64_True_head", cpu_head(head, t["final"]), t["scores"], 0.005),
        ]:
            err = float(np.max(abs(np.asarray(x, float) - np.asarray(y, float))))
            checks.append(dict(key=key, name=name, error=err, bound=bound, passes=err <= bound))
        assert np.array_equal(t["final"], t["states"][-1])
        bank[key] = (a, t)
        release(model)
    table(run / "native_checks.csv", checks)
    failed = [r for r in checks if not r["passes"]]
    audit = dict(
        experiment=292,
        status="failed" if failed else "passed",
        design_sha256=sha(run / "design.json"),
        native_status_sha256=sha(run / "native-status.json"),
        forecast_seal_sha256=sha(run / "forecast-seal.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        counts=dict(full_unhooked_true_replays=len(d["cases"])),
        source_sha256=sha(ROOT / "research/boolean_coupling/audit292.py"),
        general_question_B_solved=False,
        learned_circuit_identified=False,
    )
    tables = {"native_checks": checks}
    if not failed:
        native, trajectory, effects, coefficients = [], [], [], []
        for case in d["cases"]:
            key = case["key"]
            a, t = bank[key]
            group = case["group"]
            true = t["final"].astype(float) - a["F_final"].astype(float)
            fmargin = float(a["F_scores"][1]) - float(a["F_scores"][0])
            tm = float(t["scores"][1]) - float(t["scores"][0])
            for arm in (*ARMS, "T"):
                z = t["final"] if arm == "T" else a[arm + "_final"]
                scores = t["scores"] if arm == "T" else a[arm + "_scores"]
                delta = z.astype(float) - a["F_final"]
                receiving = scored(delta[1:], true[1:], 0.05, 0.99)
                verdict = scored(delta[-1], true[-1], 0.05, 0.99)
                err = abs(float(scores[1]) - float(scores[0]) - tm)
                native.append(
                    dict(
                        key=key,
                        **group,
                        arm=arm,
                        receiving_relative=receiving["relative"],
                        receiving_cosine=receiving["cosine"],
                        receiving_norm=receiving["norm"],
                        verdict_relative=verdict["relative"],
                        verdict_cosine=verdict["cosine"],
                        verdict_norm=verdict["norm"],
                        margin_error=err,
                        native_margin_change=tm - fmargin,
                        passes=receiving["passes"] and verdict["passes"] and err <= 0.1,
                    )
                )
                target = t["final"].astype(float) - a["A_final"]
                prediction = z.astype(float) - a["A_final"]
                em = (
                    float(scores[1])
                    - float(scores[0])
                    - float(a["A_scores"][1])
                    + float(a["A_scores"][0])
                )
                target_m = tm - float(a["A_scores"][1]) + float(a["A_scores"][0])
                rec = scored(prediction[1:], target[1:], 0.25, 0.95)
                ve = scored(prediction[-1], target[-1], 0.25, 0.95)
                Aerr = metrics(a["A_final"][1:].astype(float) - a["F_final"][1:], true[1:])[
                    "relative"
                ]
                Amerr = abs(float(a["A_scores"][1]) - float(a["A_scores"][0]) - tm)
                material = abs(target_m) >= 0.02 and (Aerr >= 0.1 or Amerr >= 0.2)
                effects.append(
                    dict(
                        key=key,
                        **group,
                        arm=arm,
                        receiving_relative=rec["relative"],
                        receiving_cosine=rec["cosine"],
                        receiving_norm=rec["norm"],
                        verdict_relative=ve["relative"],
                        verdict_cosine=ve["cosine"],
                        verdict_norm=ve["norm"],
                        prediction=em,
                        effect_target=target_m,
                        error=abs(em - target_m),
                        material=material,
                        passes=material
                        and rec["passes"]
                        and ve["passes"]
                        and abs(em - target_m) <= 0.005 + 0.25 * abs(target_m),
                    )
                )
            for name, pp, tt in [
                (
                    "source_states",
                    a["P_states"][:, 0].astype(float) - a["F_states"][:, 0],
                    t["states"][:, 0].astype(float) - a["F_states"][:, 0],
                ),
                (
                    "receiving_states",
                    a["P_states"][:, 1:].astype(float) - a["F_states"][:, 1:],
                    t["states"][:, 1:].astype(float) - a["F_states"][:, 1:],
                ),
                (
                    "source_verdict_MLP_writes",
                    a["P_writes"].astype(float) - a["F_writes"],
                    t["writes"].astype(float) - a["F_writes"],
                ),
            ]:
                trajectory.append(dict(key=key, **group, scope=name, **scored(pp, tt, 0.1, 0.995)))
            # Post-seal, same FORECAST input: not native receiving-state prediction.
            gf = a["F_witness_g"].astype(float)
            gp = a["P_witness_g"].astype(float)
            uf = a["F_witness_u"].astype(float)
            up = a["P_witness_u"].astype(float)
            pf = a["F_witness_phi"].astype(float)
            actual_phi = F.silu(torch.tensor(gp, dtype=torch.float64)).numpy()
            actual_slope = derivatives(gp)["d1"]
            fslope = derivatives(gf)["d1"]
            for j in range(len(gp)):
                for row in range(2):
                    pred = np.concatenate(
                        [
                            (up * a["P_witness_slope"] - uf * fslope)[j, row],
                            (a["P_witness_phi"].astype(float) - pf)[j, row],
                        ]
                    )
                    target = np.concatenate(
                        [(up * actual_slope - uf * fslope)[j, row], (actual_phi - pf)[j, row]]
                    )
                    coefficients.append(
                        dict(key=key, **group, site=j, row=row, **scored(pred, target, 0.1, 0.995))
                    )
        tables.update(
            native=native, trajectories=trajectory, effects=effects, coefficients=coefficients
        )
        for n, rr in tables.items():
            table(run / (n + ".csv"), rr)
        audit["native"] = {}
        audit["effects"] = {}

        def population(rr):
            layouts = {
                b: sum(r["passes"] for r in rr if r["block"] == b)
                for b in sorted({r["block"] for r in rr})
            }
            count = sum(r["passes"] for r in rr)
            return dict(
                passes=count,
                total=len(rr),
                layouts=layouts,
                population_pass=count >= 24 and all(n >= 6 for n in layouts.values()),
            )

        for arm in (*ARMS, "T"):
            audit["native"][arm] = population([r for r in native if r["arm"] == arm])
            audit["effects"][arm] = population([r for r in effects if r["arm"] == arm])
        h2 = [
            dict(
                key=c["key"],
                **c["group"],
                passes=all(r["passes"] for r in trajectory if r["key"] == c["key"])
                and next(r["passes"] for r in native if r["key"] == c["key"] and r["arm"] == "P"),
            )
            for c in d["cases"]
        ]
        audit["trajectory_joint"] = population(h2)
        audit["primary_pass"] = dict(
            H1_native=audit["native"]["P"]["population_pass"],
            H2_trajectory=audit["trajectory_joint"]["population_pass"],
            H3_joint_product_effect=audit["effects"]["P"]["population_pass"],
        )
    audit["csv_sha256"] = {n + ".csv": sha(run / (n + ".csv")) for n in tables}
    save_json(run / "native-audit.json", audit)
    print(audit, flush=True)
    assert not failed, "Native replay failed; outcome tables withheld"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("forecast", "native"))
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    if a.phase == "forecast":
        qualification(a.run.resolve())
    else:
        outcome(a.run.resolve())
