"""Separate283 staged derivatives, public primitive replay and frozen decisions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as fn

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .audit279 import coefficient_checks, maximum
from .cache272 import installed_sources, save_json
from .carried283 import ARMS, EVAL, PRODUCT_ARMS, ROOT, Observations, restore
from .checkpoint238 import locate, read_head, weights
from .composition280 import digest, padded, polynomial
from .core225b import array, load_model
from .run272 import stamp
from .source279 import derivatives


def good(v, relative=0.25, cosine=0.95):
    return bool(v["norm"] >= 0.001 and v["relative"] <= relative and v["cosine"] >= cosine)


def diff(d, a, b, name):
    return d[a + "_" + name].astype(float) - d[b + "_" + name].astype(float)


def outcome(key, group, arm, prediction, target, predicted_margin, target_margin, native=False):
    r, passes = dict(key=key, **group, arm=arm), []
    for scope, select in (("all", slice(1, None)), ("verdict", -1)):
        v = vector(prediction[select], target[select])
        r.update({scope + "_" + n: x for n, x in v.items()})
        passes.append(good(v, 0.05 if native else 0.25, 0.99 if native else 0.95))
    error = abs(predicted_margin - target_margin)
    eligible = bool(native or abs(target_margin) >= 0.02)
    r.update(
        target_margin=target_margin,
        margin_error=error,
        margin_eligible=eligible,
        passes=bool(
            all(passes)
            and eligible
            and error <= (0.1 if native else 0.005 + 0.25 * abs(target_margin))
        ),
    )
    return r


def staged_checks(data, w, independent, check):
    """DoubleAD and two finite differences of the explicitly anchored smooth map."""
    x0 = data["corner_C_mid"].astype(float)
    direction = data["point_U"].astype(float) - x0
    g0, u0 = (data["corner_C_" + n].astype(float) for n in ("gate", "up"))
    e0 = data["corner_C_phi"].astype(float) - data["corner_C_cubic"].astype(float)
    tx, tv = (torch.tensor(a, dtype=torch.float64) for a in (x0, direction))
    wn, wg, wu = (
        torch.tensor(w[n], dtype=torch.float64) for n in ("norm", "gate_weight", "up_weight")
    )

    def norm(x):
        return x / torch.sqrt(torch.mean(x * x) + float(w["eps"])) * wn

    with torch.enable_grad():
        z0 = norm(tx)
        _, dz = torch.autograd.functional.jvp(norm, tx, tv)
        dg, du = wg @ dz, wu @ dz
        gf = torch.tensor(data["F_node2_gate"].astype(float))
        gc, uc, ec = (torch.tensor(v) for v in (g0, u0, e0))
        d1, d2, d3 = (torch.tensor(independent[n]) for n in ("d1", "d2", "d3"))

        def poly(g):
            delta = g - gf
            return d1 * delta + 0.5 * d2 * delta**2 + d3 * delta**3 / 6

        def staged(x):
            change = norm(x) - z0
            g, u = gc + wg @ change, uc + wu @ change
            e = ec + (fn.silu(g) - fn.silu(gc)) - (poly(g) - poly(gc))
            return e * u

        _, product_ad = torch.autograd.functional.jvp(staged, tx, tv)
        gate_var = gc.clone().requires_grad_(True)
        prime = torch.autograd.grad(fn.silu(gate_var).sum(), gate_var)[0]
        step = gc - gf
        pc_prime = d1 + d2 * step + 0.5 * d3 * step**2
        gate_expected = (prime - pc_prime) * dg * uc
        up_expected = ec * du
    expected = dict(
        direction=direction,
        rms_jvp=dz.numpy(),
        dg=dg.numpy(),
        du=du.numpy(),
        e=e0,
        silu_prime=prime.detach().numpy(),
        cubic_prime=pc_prime.numpy(),
        gate_delta=gate_expected.detach().numpy(),
        up_delta=up_expected.numpy(),
        increment=product_ad.numpy(),
    )
    errors = {}
    for name, value in expected.items():
        err = maximum(data["response_" + name], value)
        check("staged_AD_" + name, err, 0.0, 1e-4)
        errors[name + "_AD"] = err
    check(
        "increment_sum",
        data["response_increment"],
        data["response_gate_delta"] + data["response_up_delta"],
    )
    check(
        "RMS_signed_sum",
        data["response_rms_jvp"],
        data["response_rms_direct"] + data["response_rms_scale"],
    )
    for h in (1e-4, 5e-5):
        norm_fd = ((norm(tx + h * tv) - norm(tx - h * tv)) / (2 * h)).detach().numpy()
        product_fd = ((staged(tx + h * tv) - staged(tx - h * tv)) / (2 * h)).detach().numpy()
        for name, actual, expected_value in (
            ("RMS", data["response_rms_jvp"], norm_fd),
            ("product", data["response_increment"], product_fd),
        ):
            err = maximum(actual, expected_value)
            check("staged_" + name + "_FD_" + str(h), err, 0.0, 1e-4)
            errors[name + "_FD_" + str(h)] = err
    return errors


def main(run, reference, upstream):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = (read(run / (n + ".json")) for n in ("design", "status"))
    assert (
        design["experiment"] == 283 and design["retrospective"] and status["status"] == "complete"
    )
    assert tuple(design["arms"]) == ARMS and len(status["cases"]) == 32
    assert design["installed_sources"] == installed_sources()
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    for directory, field in ((reference, "reference_hashes"), (upstream, "upstream_hashes")):
        for n, h in design[field].items():
            assert sha(directory / n) == h
    assert read(reference / "audit.json")["status"] == "passed"
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 1, "mlp")
    assert {n: digest(v) for n, v in w.items()} == read(run / "weight-hashes.json")
    model, head = load_model(True), read_head(folder)
    cases = {c["key"]: c for c in design["cases"]}
    archived = {e["key"]: e for e in read(reference / "status.json")["cases"]}
    (
        checks,
        raw,
        coefficients,
        derivs,
        heads,
        native,
        local,
        effect,
        incremental,
        incremental_effect,
        removal,
        paired,
        joint,
        ledger,
    ) = ([] for _ in range(14))
    bank, public_down_maps, public_corner_maps = {}, 0, 0

    def check(name, a, b, bound=0.0):
        error = maximum(a, b)
        checks.append(dict(key=key, name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(node, a, b, required=False):
        error = maximum(a, b)
        raw.append(
            dict(
                key=key,
                node=node,
                error=error,
                original_bound=0.0001,
                passes=bool(error <= 0.0001),
                required_bridge=required,
            )
        )
        if required:
            check("CPU64_" + node, error, 0.0, 0.0001)

    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        group, s, length = case["group"], case["position"], len(case["false_ids"])
        path = run / "cases" / (key + ".npz")
        assert (
            sha(path) == entry["arrays_sha256"]
            and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        )
        meta = read(path.with_suffix(".json"))
        assert meta["reference_entry"] == archived[key]
        assert all(
            meta[n]
            for n in (
                "own_B_inputs",
                "candidates_before_full_calibration",
                "archive_replays_exact",
                "staged_not_whole_model_derivative",
                "zero_exact",
                "full_control_exact",
            )
        )
        prior = Observations(reference, upstream, archived[key])
        d = restore(path, meta, prior)
        for arm in ("F", "B", "K", "C1", "T"):
            for name in ("source", "final", "scores", "write"):
                check("archive_" + arm + "_" + name, d[arm + "_" + name], prior[arm + "_" + name])
        for name in ("source", "final", "scores", "write"):
            check("zero_" + name, d["FZ_" + name], d["F_" + name])
            check("full_" + name, d["U_" + name], d["C1_" + name])
        coeff = {}
        for site in (1, 2):
            coeff[site], errs = coefficient_checks(
                d[f"F_node{site}_gate"], derivatives(d[f"F_node{site}_gate"])
            )
            coefficients.append(dict(key=key, site=site, bound=0.0001, **errs))
            for n, err in errs.items():
                check("False_" + str(site) + "_" + n, err, 0.0, 0.0001)
        derivs.append(dict(key=key, bound=0.0001, **staged_checks(d, w, coeff[2], check)))
        xf = d["F_node2_mid"].astype(float)
        carry = diff(d, "B", "F", "source")[0]
        attention = (d["B_node2_mid"].astype(float) - d["B_source"][0]) - (xf - d["F_source"][0])
        check("carry_input", d["component_carry"], carry)
        check("attention_input", d["component_attention"], attention)
        check("input_closure", xf + carry + attention, d["B_node2_mid"], 1e-10)
        check("carried_point", d["corner_C_mid"], (xf + carry).astype(np.float32))
        check("full_point", d["point_U"], d["B_node2_mid"])
        layer = model.model.layers[1]
        with torch.no_grad():
            full = padded(d["corner_C_mid"], length, s, model.device)
            z = layer.post_attention_layernorm(full)
            g, u = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
            phi = layer.mlp.act_fn(g)
            cubic = padded(
                polynomial(dict(gate=d["F_node2_gate"], phi=d["F_node2_phi"]), array(g[0, s]), 3),
                length,
                s,
                model.device,
            )
            pf, pc = phi * u, cubic * u
            wf, wc = layer.mlp.down_proj(pf), layer.mlp.down_proj(pc)
            for n, value in dict(
                norm=z,
                gate=g,
                up=u,
                phi=phi,
                cubic=cubic,
                finite_product=pf,
                cubic_product=pc,
                finite_write=wf,
                cubic_write=wc,
            ).items():
                check("public_C_" + n, array(value[0, s]), d["corner_C_" + n])
            public_corner_maps += 1
            public_down_maps += 2
            x = d["corner_C_mid"].astype(float)
            zz = x / np.sqrt(np.mean(x * x) + w["eps"]) * w["norm"]
            gg, uu = (zz @ w[n + "_weight"].T + w.get(n + "_bias", 0.0) for n in ("gate", "up"))
            ph64 = fn.silu(torch.tensor(d["corner_C_gate"].astype(float))).numpy()
            for n, value in dict(norm=zz, gate=gg, up=uu, phi=ph64).items():
                rawcheck(n, value, d["corner_C_" + n], n != "up")
            consumed = d["corner_C_norm"].astype(float)
            norm_error = np.abs(consumed - zz) @ np.abs(w["up_weight"]).T
            unit = 2.0**-24
            k = 2 * consumed.size + 3
            gamma = k * unit / (1 - k * unit)
            bound = norm_error + gamma * (
                np.abs(consumed) @ np.abs(w["up_weight"]).T + np.abs(w.get("up_bias", 0.0))
            )
            check(
                "up_rounding_envelope_ratio",
                float(
                    np.max(
                        np.abs(d["corner_C_up"].astype(float) - uu)
                        / np.maximum(bound, np.finfo(float).tiny)
                    )
                ),
                0.0,
                1.0,
            )
            rawcheck(
                "finite_product",
                ph64 * d["corner_C_up"].astype(float),
                d["corner_C_finite_product"],
            )
            for a in ("finite", "cubic"):
                rawcheck(
                    a + "_write",
                    d["corner_C_" + a + "_product"].astype(float) @ w["down_weight"].T,
                    d["corner_C_" + a + "_write"],
                )
            base = (d["B_node2_phi"] * d["B_node2_up"]).astype(np.float32)
            check("calibration_product", d["full_product"], prior["corner_U_finite_product"])
            rc = d["corner_C_finite_product"].astype(float) - d["corner_C_cubic_product"].astype(
                float
            )
            ru = d["full_product"].astype(float) - base.astype(float)
            check("full_remainder", d["full_remainder"], ru)
            gdelta, udelta = d["response_gate_delta"], d["response_up_delta"]
            predicted = gdelta + udelta
            plan = dict(
                K=rc,
                P=rc + predicted,
                H=rc + gdelta,
                J=rc + udelta,
                U=ru,
                N=ru - predicted,
                **{a: rc for a in ("G0", "G1", "G2")},
            )
            for arm in ARMS:
                am = meta["arms"][arm]
                assert am["prefix_hashes"] == meta["arms"]["F"]["prefix_hashes"]
                assert (
                    len(am["prefix_hashes"]) == len(model.model.layers)
                    and am["other_rows_unchanged"]
                )
                assert am["product_delta_installed"] == (arm in PRODUCT_ARMS)
                if arm not in ("F", "FZ", "T"):
                    check("same_source1", d[arm + "_source"][0], d["B_source"][0])
                if arm in plan:
                    product = (base.astype(float) + plan[arm]).astype(np.float32)
                elif arm in ("F", "FZ"):
                    product = (d["F_node2_phi"] * d["F_node2_up"]).astype(np.float32)
                elif arm == "B":
                    product = base
                elif arm == "C1":
                    product = d["full_product"]
                else:
                    product = None
                if product is not None:
                    assert digest(product) == am["product_hashes"][1]
                    replay = layer.mlp.down_proj(padded(product, length, s, model.device))
                    check("public_installed_down", array(replay[0, s]), d[arm + "_raw_write"])
                    public_down_maps += 1
                    if arm.startswith("G"):
                        i = int(arm[-1])
                        seed = int(
                            hashlib.sha256((key + f":283-increment:{i}").encode()).hexdigest()[:16],
                            16,
                        )
                        draw = np.random.default_rng(seed).normal(size=d["K_write"].shape)
                        incremental_write = diff(d, "P", "K", "write")
                        expected_draw = (draw * np.linalg.norm(incremental_write)) / np.linalg.norm(
                            draw
                        )
                        check("random_exact", d["draw_" + arm], expected_draw)
                        check(
                            "random_norm",
                            np.linalg.norm(expected_draw),
                            np.linalg.norm(incremental_write),
                            1e-10,
                        )
                        expected_write = (
                            d[arm + "_raw_write"].astype(float) + expected_draw
                        ).astype(np.float32)
                    else:
                        expected_write = d[arm + "_raw_write"]
                    check("installed_write", d[arm + "_write"], expected_write)
                    midpoint = d["F_node2_mid"] if arm in ("F", "FZ") else d["B_node2_mid"]
                    check("source2_addition", midpoint + d[arm + "_write"], d[arm + "_source"][1])
                if arm not in ("F", "C1", "T"):
                    assert am["actual_gate_hashes"] == meta["arms"]["F"]["actual_gate_hashes"]
                    assert (
                        am["primal_hashes"] == meta["arms"]["F"]["primal_hashes"]
                        and am["current_MLP2_finite_coordinates"] == 0
                    )
                tensor = torch.zeros(
                    (1, length, w["norm"].size), dtype=torch.float32, device=model.device
                )
                tensor[0, s:] = torch.tensor(d[arm + "_final"], device=model.device)
                score = array(model.lm_head(model.model.norm(tensor)[:, -1:])[0, -1, [895, 830]])
                check("public_full_head", score, d[arm + "_scores"])
                h = d[arm + "_final"][-1].astype(float)
                cpu = (
                    h
                    / np.sqrt(np.mean(h * h) + head["final_eps"])
                    * head["final_norm"]
                    @ head["final_head"].T
                )
                check("CPU64_head", cpu, d[arm + "_scores"], 0.005)
                heads.append(
                    dict(
                        key=key,
                        arm=arm,
                        public_error=maximum(score, d[arm + "_scores"]),
                        CPU64_error=maximum(cpu, d[arm + "_scores"]),
                    )
                )
        for arm in EVAL:
            native.append(
                outcome(
                    key,
                    group,
                    arm,
                    diff(d, arm, "F", "final"),
                    diff(d, "T", "F", "final"),
                    margin(d[arm + "_scores"]),
                    margin(d["T_scores"]),
                    True,
                )
            )
            v = vector(diff(d, arm, "B", "write"), diff(d, "C1", "B", "write"))
            local.append(dict(key=key, **group, arm=arm, **v, passes=good(v)))
            effect.append(
                outcome(
                    key,
                    group,
                    arm,
                    diff(d, arm, "B", "final"),
                    diff(d, "C1", "B", "final"),
                    margin(d[arm + "_scores"]) - margin(d["B_scores"]),
                    margin(d["C1_scores"]) - margin(d["B_scores"]),
                )
            )
            v = vector(diff(d, arm, "K", "write"), diff(d, "C1", "K", "write"))
            incremental.append(dict(key=key, **group, arm=arm, **v, passes=good(v)))
            incremental_effect.append(
                outcome(
                    key,
                    group,
                    arm,
                    diff(d, arm, "K", "final"),
                    diff(d, "C1", "K", "final"),
                    margin(d[arm + "_scores"]) - margin(d["K_scores"]),
                    margin(d["C1_scores"]) - margin(d["K_scores"]),
                )
            )
        v = vector(diff(d, "U", "N", "write"), diff(d, "U", "K", "write"))
        r = outcome(
            key,
            group,
            "N",
            diff(d, "U", "N", "final"),
            diff(d, "U", "K", "final"),
            margin(d["U_scores"]) - margin(d["N_scores"]),
            margin(d["U_scores"]) - margin(d["K_scores"]),
        )
        r.update({"local_" + n: x for n, x in v.items()})
        r["passes"] = bool(r["passes"] and good(v))
        removal.append(r)
        for quantity, field, select in (
            ("write", "write", slice(None)),
            ("receiving", "final", slice(1, None)),
            ("verdict", "final", -1),
        ):
            target = diff(d, "C1", "K", field)[select]
            parts = dict(
                gate=diff(d, "H", "K", field)[select],
                up=diff(d, "J", "K", field)[select],
                installation_or_downstream_nonadditivity=(
                    diff(d, "P", "K", field) - diff(d, "H", "K", field) - diff(d, "J", "K", field)
                )[select],
                unpredicted_remainder=diff(d, "C1", "P", field)[select],
            )
            check("signed_" + quantity + "_closure", sum(parts.values()), target, 1e-10)
            for component, value in parts.items():
                ledger.append(
                    dict(
                        key=key,
                        **group,
                        quantity=quantity,
                        component=component,
                        **vector(value, target),
                        component_norm=float(np.linalg.norm(value)),
                        signed_projection=float(
                            np.vdot(value, target) / max(np.vdot(target, target), 1e-30)
                        ),
                    )
                )
        bank[key] = {a: d[a + "_write"].astype(float) for a in EVAL}
        print("283 audited", len(bank), "/32", flush=True)
    groups = {}
    for key, case in cases.items():
        g = case["group"]
        index = tuple(g[n] for n in ("assignment", "block", "reverse", "target"))
        assert g["schema"] not in groups.setdefault(index, {})
        groups[index][g["schema"]] = key
    assert len(groups) == 16 and all(set(g) == {"generic", "descriptive"} for g in groups.values())
    pair_pass = {}
    for index, names in groups.items():
        desc, generic = (bank[names[n]] for n in ("descriptive", "generic"))
        target = (desc["C1"] - desc["B"]) - (generic["C1"] - generic["B"])
        for arm in EVAL:
            prediction = (desc[arm] - desc["B"]) - (generic[arm] - generic["B"])
            v = vector(prediction, target)
            paired.append(
                dict(
                    pair_key="|".join(map(str, index)),
                    generic_key=names["generic"],
                    descriptive_key=names["descriptive"],
                    arm=arm,
                    **v,
                    passes=good(v),
                )
            )
            for key in names.values():
                pair_pass[key, arm] = good(v)
    lookups = [
        {(r["key"], r["arm"]): r for r in data} for data in (native, local, effect, incremental)
    ]
    for key, case in cases.items():
        for arm in EVAL:
            flags = {
                n: index[key, arm]["passes"]
                for n, index in zip(
                    ("native_pass", "local_pass", "effect_pass", "incremental_local_pass"),
                    lookups,
                    strict=True,
                )
            }
            flags["paired_local_pass"] = pair_pass[key, arm]
            joint.append(
                dict(key=key, **case["group"], arm=arm, **flags, passes=bool(all(flags.values())))
            )
    datasets = dict(
        checks=checks,
        raw=raw,
        coefficients=coefficients,
        derivatives=derivs,
        heads=heads,
        native=native,
        local=local,
        effect=effect,
        incremental=incremental,
        incremental_effect=incremental_effect,
        removal=removal,
        paired=paired,
        joint=joint,
        ledger=ledger,
    )
    for name, data in datasets.items():
        table(run / (name + ".csv"), data)

    def counts(data):
        return {arm: sum(r["passes"] for r in data if r["arm"] == arm) for arm in EVAL}

    strata = {
        s + "/" + str(v): sum(
            r["passes"] for r in joint if r["arm"] == "P" and r["schema"] == s and r["reverse"] == v
        )
        for s in ("generic", "descriptive")
        for v in (0, 1)
    }
    save_json(
        run / "audit.json",
        dict(
            status="passed",
            experiment=283,
            retrospective=True,
            source_sha256=sha(Path(__file__)),
            design_sha256=sha(run / "design.json"),
            status_sha256=sha(run / "status.json"),
            numerical_checks=len(checks),
            False_anchors=len(coefficients),
            False_coefficient_vectors=3 * len(coefficients),
            staged_derivative_anchors=len(derivs),
            derivative_bound=0.0001,
            public_corner_maps=public_corner_maps,
            public_down_maps=public_down_maps,
            finite_head_maps=len(heads),
            raw_checks=dict(
                total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=0.0001
            ),
            **{
                n + "_counts": counts(v)
                for n, v in dict(
                    native=native,
                    local=local,
                    effect=effect,
                    incremental=incremental,
                    incremental_effect=incremental_effect,
                    paired=paired,
                    joint=joint,
                ).items()
            },
            P_strata=strata,
            primary_passes=bool(counts(joint)["P"] >= 24 and min(strata.values()) >= 6),
            removal_passes=sum(r["passes"] for r in removal),
            increment_margin_eligible=sum(
                r["margin_eligible"] for r in incremental_effect if r["arm"] == "P"
            ),
            csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
            ended=stamp(),
            boundary="Staged derivative of an explicitly anchored smooth surrogate, public Float32 map qualification; original CPU64up failures retained. Old dependent contexts, rich finite budget; no old Jacobian recertification, fresh/circuit/efficiency/generalQB claim.",
        ),
    )
    print(
        dict(
            primary=read(run / "audit.json")["primary_passes"],
            native=counts(native),
            incremental=counts(incremental),
            paired=counts(paired),
            joint=counts(joint),
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference-282", type=Path, required=True)
    p.add_argument("--reference-281d", type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.run.resolve(), a.reference_282.resolve(), a.reference_281d.resolve())
    except Exception as exc:
        if not (a.run / "failure-audit.json").exists():
            save_json(a.run / "failure-audit.json", dict(error=repr(exc), time=stamp()))
        raise
