"""Pre-native numerical seal, then independently checked native endpoints284."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as fn

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .audit279 import coefficient_checks, maximum
from .audit283 import diff, good, outcome, staged_checks
from .cache272 import installed_sources, save_json
from .carried283 import EVAL
from .carried284 import FORECAST, NATIVE, ROOT, bindings, load_forecast, load_native
from .checkpoint238 import locate, read_head, weights
from .composition280 import digest, padded, polynomial
from .core225b import array, load_model, observe
from .run272 import stamp
from .source279 import derivatives


def setup(run):
    design = bindings(run)
    assert design["installed_sources"] == installed_sources()
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 1, "mlp")
    assert {n: digest(v) for n, v in w.items()} == read(run / "weight-hashes.json")
    return design, w, load_model(True), read_head(folder)


def forecast(run):
    assert not (run / "forecast-audit.json").exists() and not (run / "native-started.json").exists()
    design, w, model, head = setup(run)
    status = read(run / "forecast-status.json")
    assert (
        status["status"] == "complete"
        and status["ordinary_true"] == status["full_current_source2_primal"] == 0
        and len(status["cases"]) == 32
    )
    cases = {c["key"]: c for c in design["cases"]}
    checks, raw, coefficients, derivs, heads, scores = [], [], [], [], [], []
    down_maps = 0

    def check(name, a, b, bound=0.0):
        error = maximum(a, b)
        checks.append(dict(key=key, name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    for entry in status["cases"]:
        key = entry["key"]
        case = cases[key]
        s = case["position"]
        length = len(case["false_ids"])
        d, meta = load_forecast(run, entry)
        assert meta["ordinary_true"] == meta["full_current_source2_primal"] == 0
        assert (
            meta["own_B_inputs"]
            and meta["zero_exact"]
            and meta["all_prefixes_exact"]
            and meta["staged_not_whole_model_derivative"]
        )
        assert set(meta["arms"]) == set(FORECAST) and not any(
            n.startswith(("T_", "C1_", "U_", "N_", "full_")) for n in d
        )
        for name in ("source", "final", "scores", "write"):
            check("zero_" + name, d["FZ_" + name], d["F_" + name])
        coeff = {}
        for site in (1, 2):
            coeff[site], errors = coefficient_checks(
                d[f"F_node{site}_gate"], derivatives(d[f"F_node{site}_gate"])
            )
            coefficients.append(dict(key=key, site=site, bound=0.0001, **errors))
            for name, error in errors.items():
                check("False_" + str(site) + "_" + name, error, 0.0, 0.0001)
        derivs.append(dict(key=key, bound=0.0001, **staged_checks(d, w, coeff[2], check)))
        xf = d["F_node2_mid"].astype(float)
        c = diff(d, "B", "F", "source")[0]
        a = (d["B_node2_mid"].astype(float) - d["B_source"][0]) - (xf - d["F_source"][0])
        check("carry", d["component_carry"], c)
        check("attention", d["component_attention"], a)
        check("input_closure", xf + c + a, d["B_node2_mid"], 1e-10)
        check("C_point", d["corner_C_mid"], (xf + c).astype(np.float32))
        check("U_point", d["point_U"], d["B_node2_mid"])
        layer = model.model.layers[1]
        with torch.no_grad():
            for site in (1, 2):
                current_layer = model.model.layers[site - 1]
                anchor = dict(gate=d[f"F_node{site}_gate"], phi=d[f"F_node{site}_phi"])
                for bit in ("F", "B"):
                    prefix = f"{bit}_node{site}_"
                    normalized = current_layer.post_attention_layernorm(
                        padded(d[prefix + "mid"], length, s, model.device)
                    )
                    gg0, uu0 = (
                        current_layer.mlp.gate_proj(normalized),
                        current_layer.mlp.up_proj(normalized),
                    )
                    check("own_input_gate", array(gg0[0, s]), d[prefix + "gate"])
                    check("own_input_up", array(uu0[0, s]), d[prefix + "up"])
                    false_primal = current_layer.mlp.act_fn(
                        padded(anchor["gate"], length, s, model.device)
                    )
                    check("observed_False_primal", array(false_primal[0, s]), anchor["phi"])
                    expected = (
                        anchor["phi"] if bit == "F" else polynomial(anchor, d[prefix + "gate"], 3)
                    )
                    check("installed_source_phi", expected, d[prefix + "phi"])
                    if bit == "B":
                        step = d[prefix + "gate"].astype(float) - anchor["gate"].astype(float)
                        independent_phi = (
                            anchor["phi"].astype(float)
                            + coeff[site]["d1"] * step
                            + 0.5 * coeff[site]["d2"] * step**2
                            + coeff[site]["d3"] * step**3 / 6
                        )
                        check(
                            "independent_source_cubic",
                            independent_phi.astype(np.float32),
                            d[prefix + "phi"],
                            0.0001,
                        )
                    installed = current_layer.mlp.down_proj(
                        padded(d[prefix + "phi"], length, s, model.device) * uu0
                    )
                    check("own_source_down", array(installed[0, s]), d[prefix + "raw_write"])
                    down_maps += 1
                    check(
                        "own_source_addition",
                        d[prefix + "mid"] + d[prefix + "raw_write"],
                        d[bit + "_source"][site - 1],
                    )
            z = layer.post_attention_layernorm(padded(d["corner_C_mid"], length, s, model.device))
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
            for name, value in dict(
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
                check("public_C_" + name, array(value[0, s]), d["corner_C_" + name])
            down_maps += 2
            x = d["corner_C_mid"].astype(float)
            zz = x / np.sqrt(np.mean(x * x) + w["eps"]) * w["norm"]
            gg, uu = (zz @ w[n + "_weight"].T + w.get(n + "_bias", 0.0) for n in ("gate", "up"))
            ph = fn.silu(torch.tensor(d["corner_C_gate"].astype(float))).numpy()
            raw_values = dict(
                norm=zz,
                gate=gg,
                up=uu,
                phi=ph,
                finite_product=ph * d["corner_C_up"].astype(float),
                finite_write=d["corner_C_finite_product"].astype(float) @ w["down_weight"].T,
                cubic_write=d["corner_C_cubic_product"].astype(float) @ w["down_weight"].T,
            )
            for name, value in raw_values.items():
                err = maximum(value, d["corner_C_" + name])
                required = name in ("norm", "gate", "phi")
                raw.append(
                    dict(
                        key=key,
                        node=name,
                        error=err,
                        original_bound=0.0001,
                        passes=bool(err <= 0.0001),
                        required_bridge=required,
                    )
                )
                if required:
                    check("CPU64_" + name, err, 0.0, 0.0001)
            consumed = d["corner_C_norm"].astype(float)
            unit = 2.0**-24
            k = 2 * consumed.size + 3
            gamma = k * unit / (1 - k * unit)
            bound = np.abs(consumed - zz) @ np.abs(w["up_weight"]).T + gamma * (
                np.abs(consumed) @ np.abs(w["up_weight"]).T + np.abs(w.get("up_bias", 0.0))
            )
            check(
                "up_envelope_ratio",
                float(
                    np.max(
                        np.abs(d["corner_C_up"].astype(float) - uu)
                        / np.maximum(bound, np.finfo(float).tiny)
                    )
                ),
                0.0,
                1.0,
            )
            base = (d["B_node2_phi"] * d["B_node2_up"]).astype(np.float32)
            rc = d["corner_C_finite_product"].astype(float) - d["corner_C_cubic_product"].astype(
                float
            )
            inc = d["response_gate_delta"] + d["response_up_delta"]
            plan = dict(
                K=rc,
                P=rc + inc,
                H=rc + d["response_gate_delta"],
                J=rc + d["response_up_delta"],
                **{a: rc for a in ("G0", "G1", "G2")},
            )
            for arm in FORECAST:
                am = meta["arms"][arm]
                assert am["prefix_hashes"] == meta["arms"]["F"]["prefix_hashes"] and len(
                    am["prefix_hashes"]
                ) == len(model.model.layers)
                assert am["other_rows_unchanged"] and am["product_delta_installed"] == (arm in plan)
                if arm not in ("F", "FZ"):
                    check("same_source1", d[arm + "_source"][0], d["B_source"][0])
                product = (
                    (base.astype(float) + plan[arm]).astype(np.float32)
                    if arm in plan
                    else (d["F_node2_phi"] * d["F_node2_up"]).astype(np.float32)
                    if arm in ("F", "FZ")
                    else base
                )
                assert digest(product) == am["product_hashes"][1]
                replay = layer.mlp.down_proj(padded(product, length, s, model.device))
                down_maps += 1
                check("public_installed_down", array(replay[0, s]), d[arm + "_raw_write"])
                if arm.startswith("G"):
                    from .carried283 import random_writes

                    draw = random_writes(case, diff(d, "P", "K", "write"))[arm]
                    check("random_exact", d["draw_" + arm], draw)
                    check(
                        "random_norm",
                        np.linalg.norm(draw),
                        np.linalg.norm(diff(d, "P", "K", "write")),
                        1e-10,
                    )
                    actual = (d[arm + "_raw_write"].astype(float) + draw).astype(np.float32)
                else:
                    actual = d[arm + "_raw_write"]
                check("installed_write", d[arm + "_write"], actual)
                midpoint = d["F_node2_mid"] if arm in ("F", "FZ") else d["B_node2_mid"]
                check("source2_add", midpoint + d[arm + "_write"], d[arm + "_source"][1])
                if arm != "F":
                    assert am["actual_gate_hashes"] == meta["arms"]["F"]["actual_gate_hashes"]
                    assert (
                        am["primal_hashes"] == meta["arms"]["F"]["primal_hashes"]
                        and am["current_MLP2_finite_coordinates"] == 0
                    )
                full = torch.zeros(
                    (1, length, w["norm"].size), dtype=torch.float32, device=model.device
                )
                full[0, s:] = torch.tensor(d[arm + "_final"], device=model.device)
                score = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check("public_head", score, d[arm + "_scores"])
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
                scores.append(
                    dict(
                        key=key,
                        **case["group"],
                        arm=arm,
                        score_false=float(d[arm + "_scores"][0]),
                        score_true=float(d[arm + "_scores"][1]),
                        margin=margin(d[arm + "_scores"]),
                    )
                )
        print("284 qualified forecasts", len(heads) // len(FORECAST), "/32", flush=True)
    data = dict(
        forecast_checks=checks,
        forecast_raw=raw,
        forecast_coefficients=coefficients,
        forecast_derivatives=derivs,
        forecast_heads=heads,
        forecast_scores=scores,
    )
    for name, rows in data.items():
        table(run / (name + ".csv"), rows)
    save_json(
        run / "forecast-audit.json",
        dict(
            status="passed",
            experiment=284,
            source_sha256=sha(Path(__file__)),
            design_sha256=sha(run / "design.json"),
            forecast_status_sha256=sha(run / "forecast-status.json"),
            ordinary_true=0,
            full_current_source2_primal=0,
            numerical_checks=len(checks),
            False_anchors=64,
            False_coefficient_vectors=192,
            staged_derivative_anchors=32,
            public_corner_maps=32,
            public_down_maps=down_maps,
            finite_head_maps=len(heads),
            raw_checks=dict(
                total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=0.0001
            ),
            csv_sha256={name + ".csv": sha(run / (name + ".csv")) for name in data},
            ended=stamp(),
            boundary="Pre-native qualification of declared staged derivative and public Float32 scope, failed rawup diagnostics retained; not old whole-block recertification.",
        ),
    )
    save_json(
        run / "forecast-seal.json",
        dict(
            experiment=284,
            globally_pre_native=True,
            ordinary_true=0,
            full_current_source2_primal=0,
            forecast_audit_sha256=sha(run / "forecast-audit.json"),
            forecast_status_sha256=sha(run / "forecast-status.json"),
            design_sha256=sha(run / "design.json"),
            source_hashes=design["source_hashes"],
            cases=status["cases"],
            ended=stamp(),
            requires_public_push_before_native=not (
                design["synthetic"] or design["published_literal_repetition"]
            ),
        ),
    )
    print(
        "284 all32 forecasts and numerical qualification sealed; ordinary nativeTrue=0.", flush=True
    )


def native(run):
    assert not (run / "native-audit.json").exists()
    design, w, model, head = setup(run)
    fs, ns, fa, seal, start = (
        read(run / (n + ".json"))
        for n in (
            "forecast-status",
            "native-status",
            "forecast-audit",
            "forecast-seal",
            "native-started",
        )
    )
    assert ns["status"] == "complete" and fs["status"] == "complete" and fa["status"] == "passed"
    assert len(ns["cases"]) == len(fs["cases"]) == 32
    assert ns["counts"]["full_forwards"] == 128 and ns["counts"]["ordinary_true"] == 32
    assert fs["ordinary_true"] == fs["full_current_source2_primal"] == 0
    assert fa["source_sha256"] == sha(Path(__file__))
    assert seal["design_sha256"] == fa["design_sha256"] == sha(run / "design.json")
    assert (
        sha(run / "forecast-seal.json")
        == start["forecast_seal_sha256"]
        == start["receipt"]["forecast_seal_sha256"]
    )
    assert seal["forecast_audit_sha256"] == sha(run / "forecast-audit.json") and seal[
        "forecast_status_sha256"
    ] == sha(run / "forecast-status.json")
    if not design["synthetic"] and not design["published_literal_repetition"]:
        import subprocess

        receipt = start["receipt"]
        assert receipt["remote_head_verified"]
        published = subprocess.check_output(
            ["git", "show", receipt["public_forecast_commit"] + ":" + receipt["public_seal_path"]],
            cwd=ROOT,
        )
        import hashlib

        assert hashlib.sha256(published).hexdigest() == sha(run / "forecast-seal.json")
    forecasts = {r["key"]: r for r in fs["cases"]}
    cases = {c["key"]: c for c in design["cases"]}
    (
        checks,
        heads,
        native_rows,
        local,
        effect,
        incremental,
        incremental_effect,
        removal,
        paired,
        joint,
        ledger,
    ) = ([] for _ in range(11))
    bank = {}

    def check(name, a, b, bound=0.0):
        err = maximum(a, b)
        checks.append(dict(key=key, name=name, error=err, bound=bound))
        assert np.isfinite(err) and err <= bound, checks[-1]

    for entry in ns["cases"]:
        key = entry["key"]
        case = cases[key]
        group = case["group"]
        s = case["position"]
        length = len(case["false_ids"])
        d, fm = load_forecast(run, forecasts[key])
        nd, nm = load_native(run, entry, d)
        d.update(nd)
        assert (
            nm["full_control_exact"]
            and nm["forecast_arrays_sha256"] == forecasts[key]["arrays_sha256"]
        )
        assert nm["forecast_metadata_sha256"] == forecasts[key]["metadata_sha256"]
        base = (d["B_node2_phi"] * d["B_node2_up"]).astype(np.float32)
        check(
            "full_finite_product",
            d["full_product"],
            (d["C1_node2_phi"] * d["C1_node2_up"]).astype(np.float32),
        )
        ru = d["full_product"].astype(float) - base.astype(float)
        check("full_remainder", d["full_remainder"], ru)
        with torch.no_grad():
            for name in ("source", "final", "scores", "write"):
                check("full_" + name, d["U_" + name], d["C1_" + name])
            actual_layer = model.model.layers[1]
            actual_norm = actual_layer.post_attention_layernorm(
                padded(d["B_node2_mid"], length, s, model.device)
            )
            actual_gate, actual_up = (
                actual_layer.mlp.gate_proj(actual_norm),
                actual_layer.mlp.up_proj(actual_norm),
            )
            actual_phi = actual_layer.mlp.act_fn(actual_gate)
            for name, value in dict(gate=actual_gate, up=actual_up, phi=actual_phi).items():
                check("native_full_" + name, array(value[0, s]), d["C1_node2_" + name])
            check(
                "native_public_full_product",
                array((actual_phi * actual_up)[0, s]),
                d["full_product"],
            )
            for arm in NATIVE:
                meta = nm["arms"][arm]
                assert (
                    meta["prefix_hashes"] == fm["arms"]["F"]["prefix_hashes"]
                    and meta["other_rows_unchanged"]
                )
                assert meta["product_delta_installed"] == (arm in ("U", "N"))
                if arm != "T":
                    check("same_source1", d[arm + "_source"][0], d["B_source"][0])
                    product = (
                        d["full_product"]
                        if arm == "C1"
                        else (
                            base.astype(float)
                            + ru
                            - (d["response_increment"] if arm == "N" else 0.0)
                        ).astype(np.float32)
                    )
                    assert digest(product) == meta["product_hashes"][1]
                    replay = model.model.layers[1].mlp.down_proj(
                        padded(product, length, s, model.device)
                    )
                    check("native_installed_down", array(replay[0, s]), d[arm + "_write"])
                    check(
                        "native_source2_add",
                        d["B_node2_mid"] + d[arm + "_write"],
                        d[arm + "_source"][1],
                    )
                full = torch.zeros(
                    (1, length, w["norm"].size), dtype=torch.float32, device=model.device
                )
                full[0, s:] = torch.tensor(d[arm + "_final"], device=model.device)
                score = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check("native_public_head", score, d[arm + "_scores"])
                h = d[arm + "_final"][-1].astype(float)
                cpu = (
                    h
                    / np.sqrt(np.mean(h * h) + head["final_eps"])
                    * head["final_norm"]
                    @ head["final_head"].T
                )
                check("native_CPU64_head", cpu, d[arm + "_scores"], 0.005)
                heads.append(
                    dict(
                        key=key,
                        arm=arm,
                        public_error=maximum(score, d[arm + "_scores"]),
                        CPU64_error=maximum(cpu, d[arm + "_scores"]),
                    )
                )
            # No editing hooks: independent ordinary native True replay after seal.
            states, _, score = observe(model, torch.tensor([case["true_ids"]], device=model.device))
            check("ordinary_True_replay", array(states[-1][0, s:]), d["T_final"])
            check("ordinary_True_scores", array(score), d["T_scores"])
        for arm in EVAL:
            native_rows.append(
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
        r.update({"local_" + n: v for n, v in v.items()})
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
        bank[key] = {arm: d[arm + "_write"].astype(float) for arm in EVAL}
        print("284 audited native", len(bank), "/32", flush=True)
    groups = {}
    for key, c in cases.items():
        g = c["group"]
        index = tuple(g[n] for n in ("assignment", "block", "reverse", "target"))
        assert g["schema"] not in groups.setdefault(index, {})
        groups[index][g["schema"]] = key
    assert len(groups) == 16
    pair_pass = {}
    for index, names in groups.items():
        desc, gen = (bank[names[n]] for n in ("descriptive", "generic"))
        target = (desc["C1"] - desc["B"]) - (gen["C1"] - gen["B"])
        for arm in EVAL:
            v = vector((desc[arm] - desc["B"]) - (gen[arm] - gen["B"]), target)
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
    indices = [
        {(r["key"], r["arm"]): r for r in data}
        for data in (native_rows, local, effect, incremental)
    ]
    for key, c in cases.items():
        for arm in EVAL:
            flags = {
                name: index[key, arm]["passes"]
                for name, index in zip(
                    ("native_pass", "local_pass", "effect_pass", "incremental_local_pass"),
                    indices,
                    strict=True,
                )
            }
            flags["paired_local_pass"] = pair_pass[key, arm]
            joint.append(
                dict(key=key, **c["group"], arm=arm, **flags, passes=bool(all(flags.values())))
            )
    data = dict(
        native_checks=checks,
        native_heads=heads,
        native=native_rows,
        local=local,
        effect=effect,
        incremental=incremental,
        incremental_effect=incremental_effect,
        removal=removal,
        paired=paired,
        joint=joint,
        ledger=ledger,
    )
    for name, rows in data.items():
        table(run / (name + ".csv"), rows)

    def counts(rows):
        return {arm: sum(r["passes"] for r in rows if r["arm"] == arm) for arm in EVAL}

    strata = {
        s + "/" + str(v): sum(
            r["passes"] for r in joint if r["arm"] == "P" and r["schema"] == s and r["reverse"] == v
        )
        for s in ("generic", "descriptive")
        for v in (0, 1)
    }
    save_json(
        run / "native-audit.json",
        dict(
            status="passed",
            experiment=284,
            source_sha256=sha(Path(__file__)),
            forecast_seal_sha256=sha(run / "forecast-seal.json"),
            native_status_sha256=sha(run / "native-status.json"),
            native_started_sha256=sha(run / "native-started.json"),
            **{
                name + "_counts": counts(rows)
                for name, rows in dict(
                    native=native_rows,
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
            numerical_checks=len(checks),
            finite_head_maps=len(heads),
            ordinary_True_replay_forwards=32,
            removal_passes=sum(r["passes"] for r in removal),
            increment_margin_eligible=sum(
                r["margin_eligible"] for r in incremental_effect if r["arm"] == "P"
            ),
            csv_sha256={name + ".csv": sha(run / (name + ".csv")) for name in data},
            ended=stamp(),
            boundary="New literal templates with pre-native sealed predictions; dependent backgrounds, additional finiteCmap/substantial finite work; staged surrogate derivative not whole-block/circuit/efficiency/generalQB closure.",
        ),
    )
    print(
        dict(
            primary=read(run / "native-audit.json")["primary_passes"],
            native=counts(native_rows),
            incremental=counts(incremental),
            paired=counts(paired),
            joint=counts(joint),
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("forecast", "native"))
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    try:
        (forecast if a.stage == "forecast" else native)(a.run.resolve())
    except Exception as exc:
        name = "failure-" + a.stage + "-audit.json"
        if not (a.run / name).exists():
            save_json(a.run / name, dict(error=repr(exc), time=stamp()))
        raise
