"""Qualify 285 forecasts before native measurement; then audit frozen native tests."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as fn
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .audit279 import coefficient_checks, maximum
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .composition280 import digest, padded
from .core225b import array, load_model, observe
from .run272 import stamp

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ("C", "L", "D", "M", "Z", "FZ")
ARMS = ("F", *FORECAST)
EVAL = ("C", "L", "D", "M", "Z", "T")
MATCH = ("schema", "reverse", "target", "assignment")


def bindings(run, model=None):
    design = read(run / "design.json")
    assert design["experiment"] == 285 and design["globally_pre_native"]
    assert tuple(design["arms"]) == ARMS
    assert design["installed_sources"] == installed_sources()
    for name, expected in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == expected, name
    assert sha(run / "bank-config.json") == design["bank_config_sha256"]
    assert read(run / "bank-config.json")["reference_hashes"] == design["reference_hashes"]
    if not design["synthetic"]:
        assert sha(ROOT / "research/boolean_coupling/bank285.json") == design["bank_config_sha256"]
    if model is None:
        assert not design["synthetic"]
        assert locate(ROOT)[1] == design["checkpoint"]
        model = load_model(True)
    else:
        assert design["synthetic"] and model.device.type == "cpu"
        signature = dict(
            synthetic=True,
            hidden=model.config.hidden_size,
            neurons=model.config.intermediate_size,
            layers=len(model.model.layers),
            weights={n: digest(array(p)) for n, p in model.named_parameters()},
        )
        assert design["checkpoint"] == signature
    assert not model.training and model.config._attn_implementation == "eager"
    cases = design["cases"]
    assert len({c["key"] for c in cases}) == len(cases) == 32
    assert len({c["group"]["block"] for c in cases}) == 4
    assert set(Counter(c["group"]["block"] for c in cases).values()) == {8}
    seen = set()
    historical = set(design.get("historical_input_hashes", []))
    for case in cases:
        f, t, s = case["false_ids"], case["true_ids"], case["position"]
        assert len(f) == len(t) and 0 <= s < len(f) - 1
        assert [i for i, (a, b) in enumerate(zip(f, t, strict=True)) if a != b] == [s]
        assert f[s] == 895 and t[s] == 830
        hashes = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in (f, t)
        ]
        assert case["input_hashes"] == hashes and not set(hashes) & (seen | historical)
        seen.update(hashes)
    return design, model


def historical_bank(run, reference):
    """Reconstruct the fixed bank separately, without opening native observations."""
    cfg = read(run / "bank-config.json")
    assert cfg["experiment"] == 285 and cfg["reference_experiment"] == 279
    assert tuple(cfg["match_fields"]) == MATCH
    assert set(cfg["reference_hashes"]) == {
        "design.json",
        "forecast-status.json",
        "forecast-audit.json",
        "forecast-seal.json",
    }
    for name, expected in cfg["reference_hashes"].items():
        assert sha(reference / name) == expected, name
    old = read(reference / "design.json")
    status = read(reference / "forecast-status.json")
    audit = read(reference / "forecast-audit.json")
    seal = read(reference / "forecast-seal.json")
    assert (
        old["experiment"] == 279 and status["status"] == "complete" and audit["status"] == "passed"
    )
    assert status["ordinary_true"] == audit["ordinary_true"] == seal["ordinary_true"] == 0
    assert seal["cases"] == status["cases"]
    assert seal["design_sha256"] == audit["design_sha256"] == sha(reference / "design.json")
    assert (
        seal["status_sha256"] == audit["status_sha256"] == sha(reference / "forecast-status.json")
    )
    assert seal["audit_sha256"] == sha(reference / "forecast-audit.json")
    indexed = {e["key"]: e for e in status["cases"]}
    cases = {c["key"]: c for c in old["cases"]}
    assert len(indexed) == len(status["cases"]) == len(cases) == 32
    result, selected = {}, set()
    for group in cfg["groups"]:
        match = tuple(group["group"][n] for n in MATCH)
        assert match not in result and match[-1] == 0 and len(group["donors"]) == 2
        deltas, blocks = [], set()
        for donor in group["donors"]:
            key = donor["key"]
            assert key not in selected and indexed[key] == donor
            selected.add(key)
            assert tuple(cases[key]["group"][n] for n in MATCH) == match
            blocks.add(cases[key]["group"]["block"])
            p = reference / "forecasts" / (key + ".npz")
            assert sha(p) == donor["arrays_sha256"] and p.stat().st_size == donor["bytes"]
            assert sha(p.with_suffix(".json")) == donor["metadata_sha256"]
            meta = read(p.with_suffix(".json"))
            assert meta["arrays_sha256"] == donor["arrays_sha256"] and meta["ordinary_true"] == 0
            with np.load(p) as data:
                a, b = data["writes_C"], data["false_write"]
                assert a.dtype == b.dtype == np.float32 and a.shape == b.shape and a.ndim == 1
                deltas.append(a.astype(float) - b.astype(float))
        assert len(blocks) == 2
        result[match] = (np.stack(deltas).mean(0), [r["key"] for r in group["donors"]])
    assert len(result) == 8 and len(selected) == 16
    return result


def case_data(run, folder, entry):
    p = run / folder / (entry["key"] + ".npz")
    assert sha(p) == entry["arrays_sha256"] and p.stat().st_size == entry["bytes"]
    assert sha(p.with_suffix(".json")) == entry["metadata_sha256"]
    meta = read(p.with_suffix(".json"))
    assert meta["key"] == entry["key"] and meta["arrays_sha256"] == entry["arrays_sha256"]
    with np.load(p) as saved:
        data = {n: saved[n] for n in saved.files}
    assert all(np.isfinite(v).all() for v in data.values())
    return data, meta


def head(model, case, residual):
    h = model.config.hidden_size
    full = torch.zeros((1, len(case["false_ids"]), h), dtype=torch.float32, device=model.device)
    full[0, case["position"] :] = torch.tensor(residual, device=model.device)
    score = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
    last = residual[-1].astype(float)
    norm = array(model.model.norm.weight).astype(float)
    weight = array(model.lm_head.weight[[895, 830]]).astype(float)
    cpu = last / np.sqrt(np.mean(last * last) + model.config.rms_norm_eps) * norm @ weight.T
    return score, cpu


def forecast(run, reference, model=None):
    assert not (run / "forecast-audit.json").exists() and not (run / "native-started.json").exists()
    design, model = bindings(run, model)
    status = read(run / "forecast-status.json")
    assert status["status"] == "complete" and status["ordinary_true"] == 0
    assert len(status["cases"]) == 32
    bank = historical_bank(run, reference)
    layer = model.model.layers[0]
    checks, coefficients, raw, heads, scores = [], [], [], [], []

    def check(key, name, a, b, bound=0.0):
        error = maximum(a, b)
        row = dict(key=key, name=name, error=error, bound=bound)
        checks.append(row)
        assert np.isfinite(error) and error <= bound, row

    cases = {c["key"]: c for c in design["cases"]}
    assert [e["key"] for e in status["cases"]] == list(cases)
    with torch.no_grad():
        for entry in status["cases"]:
            key, case = entry["key"], cases[entry["key"]]
            start, length = case["position"], len(case["false_ids"])
            d, meta = case_data(run, "forecasts", entry)
            width, neurons = model.config.hidden_size, model.config.intermediate_size
            for arm in ARMS:
                for name in (
                    "before",
                    "mid",
                    "raw_write",
                    "write",
                    "source",
                    "z",
                    "norm",
                    "attwrite",
                ):
                    assert d[arm + "_" + name].shape == (width,)
                for name in ("gate", "up", "phi"):
                    assert d[arm + "_" + name].shape == (neurons,)
                assert d[arm + "_final"].shape == (length - start, width)
                assert d[arm + "_scores"].shape == (2,)
            assert meta["ordinary_true"] == meta["full_current_first_source_SiLU"] == 0
            assert set(meta["arms"]) == set(ARMS) and not any(n.startswith("T_") for n in d)
            delta, donor_keys = bank[tuple(case["group"][n] for n in MATCH)]
            assert meta["donor_keys"] == donor_keys and meta["bank_delta_sha256"] == digest(delta)
            check(key, "independent_bank", d["donor_delta"], delta)
            independent, errs = coefficient_checks(
                d["F_gate"], {n: d["coeff_" + n] for n in ("d1", "d2", "d3")}
            )
            coefficients.append(dict(key=key, bound=0.0001, **errs))
            for n, err in errs.items():
                check(key, "False_coefficient_" + n, err, 0, 0.0001)
            false_gate = padded(d["F_gate"], length, start, model.device)
            check(
                key,
                "observed_False_SiLU",
                array(layer.mlp.act_fn(false_gate)[0, start]),
                d["F_phi"],
            )
            check(
                key,
                "CPU64_False_SiLU",
                fn.silu(torch.tensor(d["F_gate"].astype(float))).numpy(),
                d["F_phi"],
                0.0001,
            )
            for arm, ids in (("F", case["false_ids"]), ("C", case["true_ids"])):
                full = model.model.embed_tokens(torch.tensor([ids], device=model.device))
                z0 = layer.input_layernorm(full)
                hd, nq, nk = (
                    layer.self_attn.head_dim,
                    model.config.num_attention_heads,
                    model.config.num_key_value_heads,
                )
                q = layer.self_attn.q_proj(z0).view(1, length, nq, hd).transpose(1, 2)
                k = layer.self_attn.k_proj(z0).view(1, length, nk, hd).transpose(1, 2)
                v = layer.self_attn.v_proj(z0).view(1, length, nk, hd).transpose(1, 2)
                cos, sin = (
                    torch.tensor(d[n], device=model.device) for n in ("rope_cos", "rope_sin")
                )
                q, k = qwen.apply_rotary_pos_emb(q, k, cos, sin)
                for n, value in dict(
                    before=full[0, start],
                    norm=z0[0, start],
                    q=q[0, :, start],
                    K=k[0, :, start],
                    V=v[0, :, start],
                    full_k=k[0],
                    full_v=v[0],
                ).items():
                    check(key, "known_" + arm + "_" + n, array(value), d[arm + "_" + n])
                mask = torch.tensor(d["context_mask"], device=model.device)
                y, p = qwen.eager_attention_forward(
                    layer.self_attn, q, k, v, mask, scaling=hd**-0.5, dropout=0.0
                )
                out = layer.self_attn.o_proj(y.reshape(1, length, nq * hd))
                check(
                    key,
                    "attention_" + arm + "_probability",
                    array(p[0, :, start]),
                    d[arm + "_p"],
                    0.0001,
                )
                check(key, "attention_" + arm + "_PV", array(y[0, start]), d[arm + "_y"], 0.0001)
                check(
                    key,
                    "attention_" + arm + "_write",
                    array(out[0, start]),
                    d[arm + "_attwrite"],
                    0.0001,
                )
                check(
                    key,
                    "attention_" + arm + "_mid",
                    array(full[0, start] + out[0, start]),
                    d[arm + "_mid"],
                    0.0001,
                )
                check(
                    key,
                    "saved_mid_" + arm,
                    d[arm + "_before"] + d[arm + "_attwrite"],
                    d[arm + "_mid"],
                )
                normalized = layer.post_attention_layernorm(
                    padded(d[arm + "_mid"], length, start, model.device)
                )
                gate, up = layer.mlp.gate_proj(normalized), layer.mlp.up_proj(normalized)
                for n, value in dict(z=normalized, gate=gate, up=up).items():
                    check(key, "public_" + arm + "_" + n, array(value[0, start]), d[arm + "_" + n])
                x = d[arm + "_mid"].astype(float)
                cpu_z = (
                    x
                    / np.sqrt(np.mean(x * x) + model.config.rms_norm_eps)
                    * array(layer.post_attention_layernorm.weight).astype(float)
                )
                check(key, "CPU64_norm_" + arm, cpu_z, d[arm + "_z"], 0.0001)
                for n in ("gate", "up"):
                    module = getattr(layer.mlp, n + "_proj")
                    cpu = cpu_z @ array(module.weight).astype(float).T
                    if module.bias is not None:
                        cpu += array(module.bias).astype(float)
                    error = maximum(cpu, d[arm + "_" + n])
                    raw.append(
                        dict(
                            key=key,
                            arm=arm,
                            node=n,
                            error=error,
                            original_bound=0.0001,
                            passes=error <= 0.0001,
                        )
                    )
            dg = d["C_gate"].astype(float) - d["F_gate"].astype(float)
            linear = d["F_phi"].astype(float) + independent["d1"] * dg
            cubic = linear + 0.5 * independent["d2"] * dg**2 + independent["d3"] * dg**3 / 6
            check(key, "independent_C_phi", cubic.astype(np.float32), d["C_phi"], 0.0001)
            check(key, "independent_L_phi", linear.astype(np.float32), d["L_phi"], 0.0001)
            expected_writes = {
                "D": (d["F_write"].astype(float) + delta).astype(np.float32),
                "Z": d["F_write"],
                "FZ": d["F_write"],
            }
            ratio = np.linalg.norm(
                d["C_write"].astype(float) - d["F_write"].astype(float)
            ) / np.linalg.norm(delta)
            assert np.isfinite(ratio) and np.linalg.norm(delta) > 0
            expected_writes["M"] = (d["F_write"].astype(float) + delta * ratio).astype(np.float32)
            for arm in ARMS:
                am = meta["arms"][arm]
                for n in ("prefix_hashes", "K_prefix", "V_prefix"):
                    assert am[n] == meta["arms"]["F"][n] and len(am[n]) == len(model.model.layers)
                assert am["current_source_finite_sites"] == len(model.model.layers) - (arm != "F")
                assert am["false_primal_checks"] == int(arm != "F")
                assert am["ordinary_true"] == 0 and am["only_first_source_policy"] is True
                assert am["full_current_first_source_activation"] == int(arm == "F")
                origin = "F" if arm in ("F", "FZ") else "C"
                for n in ("before", "mid", "gate", "up", "z"):
                    check(
                        key, "same_incoming_" + arm + "_" + n, d[arm + "_" + n], d[origin + "_" + n]
                    )
                phi_origin = "F" if arm in ("F", "FZ") else "L" if arm == "L" else "C"
                check(key, "same_installed_phi_" + arm, d[arm + "_phi"], d[phi_origin + "_phi"])
                product = padded(d[arm + "_phi"], length, start, model.device) * padded(
                    d[arm + "_up"], length, start, model.device
                )
                write = layer.mlp.down_proj(product)
                check(key, "public_raw_write_" + arm, array(write[0, start]), d[arm + "_raw_write"])
                expected = expected_writes.get(arm, d[arm + "_raw_write"])
                check(key, "installed_write_" + arm, d[arm + "_write"], expected)
                check(key, "source_addition_" + arm, d[arm + "_mid"] + expected, d[arm + "_source"])
                assert am["product_hashes"][0] == digest(array(product[0, start]))
                if arm in ("C", "L"):
                    ph = cubic if arm == "C" else linear
                    cpu_product = ph * d[arm + "_up"].astype(float)
                    cpu_write = (
                        array(product[0, start]).astype(float)
                        @ array(layer.mlp.down_proj.weight).astype(float).T
                    )
                    if layer.mlp.down_proj.bias is not None:
                        cpu_write += array(layer.mlp.down_proj.bias).astype(float)
                    for n, a, b in (
                        ("phi", ph, d[arm + "_phi"]),
                        ("product", cpu_product, array(product[0, start])),
                        ("write", cpu_write, d[arm + "_raw_write"]),
                    ):
                        error = maximum(a, b)
                        raw.append(
                            dict(
                                key=key,
                                arm=arm,
                                node=n,
                                error=error,
                                original_bound=0.0001,
                                passes=error <= 0.0001,
                            )
                        )
                score, cpu_score = head(model, case, d[arm + "_final"])
                check(key, "public_head_" + arm, score, d[arm + "_scores"])
                check(key, "CPU64_head_" + arm, cpu_score, d[arm + "_scores"], 0.005)
                heads.append(
                    dict(
                        key=key,
                        arm=arm,
                        public_error=maximum(score, d[arm + "_scores"]),
                        CPU64_error=maximum(cpu_score, d[arm + "_scores"]),
                    )
                )
                scores.append(
                    dict(
                        key=key,
                        **case["group"],
                        arm=arm,
                        false_score=float(score[0]),
                        true_score=float(score[1]),
                        margin=margin(score),
                    )
                )
            for n in (
                "before",
                "mid",
                "gate",
                "up",
                "phi",
                "raw_write",
                "write",
                "source",
                "final",
                "scores",
                "z",
            ):
                check(key, "False_zero_" + n, d["FZ_" + n], d["F_" + n])
            print("285 qualified forecasts", len(coefficients), "/32", flush=True)
    outputs = dict(
        forecast_checks=checks,
        forecast_coefficients=coefficients,
        forecast_raw=raw,
        forecast_heads=heads,
        forecast_scores=scores,
    )
    for name, rows in outputs.items():
        table(run / (name + ".csv"), rows)
    csv_hashes = {name + ".csv": sha(run / (name + ".csv")) for name in outputs}
    save_json(
        run / "forecast-audit.json",
        dict(
            status="passed",
            experiment=285,
            source_sha256=sha(Path(__file__)),
            design_sha256=sha(run / "design.json"),
            forecast_status_sha256=sha(run / "forecast-status.json"),
            bank_config_sha256=sha(run / "bank-config.json"),
            ordinary_true=0,
            full_current_first_source_SiLU=0,
            numerical_checks=len(checks),
            False_anchors=32,
            False_coefficient_vectors=96,
            finite_head_maps=len(heads),
            raw_checks=dict(
                total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=0.0001
            ),
            csv_sha256=csv_hashes,
            ended=stamp(),
            boundary="Declared False coefficients and exact public Float32 operations; raw CPU64 projections retained diagnostically, no current source SiLU before seal or whole-block derivative claim.",
        ),
    )
    save_json(
        run / "forecast-seal.json",
        dict(
            experiment=285,
            globally_pre_native=True,
            ordinary_true=0,
            full_current_first_source_SiLU=0,
            forecast_audit_sha256=sha(run / "forecast-audit.json"),
            forecast_status_sha256=sha(run / "forecast-status.json"),
            design_sha256=sha(run / "design.json"),
            bank_config_sha256=sha(run / "bank-config.json"),
            source_hashes=design["source_hashes"],
            cases=status["cases"],
            csv_sha256=csv_hashes,
            ended=stamp(),
            requires_public_push_before_native=not design["synthetic"],
        ),
    )


def good(v, relative, cosine):
    return bool(v["norm"] >= 0.001 and v["relative"] <= relative and v["cosine"] >= cosine)


def endpoint(case, arm, d, target_arm, baseline, native=False):
    row = dict(key=case["key"], **case["group"], arm=arm)
    flags = []
    for scope, selection in (("all", slice(1, None)), ("verdict", -1)):
        prediction = d[arm + "_final"].astype(float) - d[baseline + "_final"].astype(float)
        target = d[target_arm + "_final"].astype(float) - d[baseline + "_final"].astype(float)
        v = vector(prediction[selection], target[selection])
        row.update({scope + "_" + n: value for n, value in v.items()})
        flags.append(good(v, 0.05 if native else 0.25, 0.99 if native else 0.95))
    target_margin = margin(d[target_arm + "_scores"]) - margin(d[baseline + "_scores"])
    error = abs(margin(d[arm + "_scores"]) - margin(d[target_arm + "_scores"]))
    eligible = native or abs(target_margin) >= 0.02
    row.update(
        target_margin=target_margin,
        margin_error=error,
        margin_eligible=eligible,
        passes=bool(
            all(flags)
            and eligible
            and error <= (0.1 if native else 0.005 + 0.25 * abs(target_margin))
        ),
    )
    return row


def native(run, reference, model=None):
    assert not (run / "native-audit.json").exists()
    design, model = bindings(run, model)
    historical_bank(run, reference)
    status, seal, start = (
        read(run / (n + ".json")) for n in ("native-status", "forecast-seal", "native-started")
    )
    fa = read(run / "forecast-audit.json")
    assert (
        status["status"] == "complete"
        and status["ordinary_true"] == 32
        and len(status["cases"]) == 32
    )
    assert (
        fa["status"] == "passed"
        and fa["ordinary_true"] == seal["ordinary_true"] == start["ordinary_true_before"] == 0
    )
    assert seal["ended"] <= start["began"]
    assert start["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    assert seal["forecast_audit_sha256"] == sha(run / "forecast-audit.json")
    assert seal["forecast_status_sha256"] == sha(run / "forecast-status.json")
    assert seal["design_sha256"] == sha(run / "design.json")
    assert seal["source_hashes"] == design["source_hashes"]
    assert fa["forecast_status_sha256"] == sha(run / "forecast-status.json")
    assert fa["design_sha256"] == sha(run / "design.json")
    assert fa["source_sha256"] == sha(Path(__file__))
    forecast_status = read(run / "forecast-status.json")
    assert seal["cases"] == forecast_status["cases"]
    assert seal["csv_sha256"] == fa["csv_sha256"]
    assert seal["bank_config_sha256"] == fa["bank_config_sha256"] == sha(run / "bank-config.json")
    if not design["synthetic"]:
        assert start["remote_head_verified"] is True
        proof = subprocess.check_output(
            [
                "git",
                "show",
                start["public_commit"]
                + ":results/native-structure-285/pre-native/forecast-seal.json",
            ],
            cwd=ROOT,
        )
        assert hashlib.sha256(proof).hexdigest() == sha(run / "forecast-seal.json")
    for name, expected in fa["csv_sha256"].items():
        assert sha(run / name) == expected
    forecasts = {e["key"]: e for e in seal["cases"]}
    cases = {c["key"]: c for c in design["cases"]}
    assert [e["key"] for e in status["cases"]] == list(cases)
    checks, heads, bank = [], [], {}

    def check(key, name, a, b, bound=0.0):
        error = maximum(a, b)
        row = dict(key=key, name=name, error=error, bound=bound)
        checks.append(row)
        assert np.isfinite(error) and error <= bound, row

    layer = model.model.layers[0]
    with torch.no_grad():
        for entry in status["cases"]:
            key, case = entry["key"], cases[entry["key"]]
            f, fm = case_data(run, "forecasts", forecasts[key])
            t, tm = case_data(run, "native", entry)
            assert tm.get("measured", tm.get("began", start["began"])) >= start["began"]
            d = {**f, **t}
            assert tm["ordinary_true"] is True and tm["editing_hooks_applied"] is False
            assert tm["arm"]["false_primal_checks"] == 0
            assert tm["arm"]["current_source_finite_sites"] == len(model.model.layers)
            for name in ("prefix_hashes", "K_prefix", "V_prefix"):
                assert tm["arm"][name] == fm["arms"]["F"][name]
            for n in ("before", "mid", "gate", "up", "z"):
                check(key, "prepared_native_" + n, d["T_" + n], d["C_" + n])
            length, s = len(case["false_ids"]), case["position"]
            point = padded(d["T_mid"], length, s, model.device)
            z = layer.post_attention_layernorm(point)
            gate, up = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
            phi = layer.mlp.act_fn(gate)
            product = phi * up
            write = layer.mlp.down_proj(product)
            for n, value in dict(
                z=z, gate=gate, up=up, phi=phi, raw_write=write, write=write
            ).items():
                check(key, "native_public_" + n, array(value[0, s]), d["T_" + n])
            check(key, "native_source_addition", d["T_mid"] + d["T_write"], d["T_source"])
            # Ordinary replay has no source intervention or attention wrapper.
            states, _, scores = observe(
                model, torch.tensor([case["true_ids"]], device=model.device)
            )
            check(key, "ordinary_True_source", array(states[0][0, s]), d["T_source"])
            check(key, "ordinary_True_final", array(states[-1][0, s:]), d["T_final"])
            check(key, "ordinary_True_scores", array(scores), d["T_scores"])
            assert [digest(array(v[0, :s])) for v in states] == fm["arms"]["F"]["prefix_hashes"]
            for arm in (*ARMS, "T"):
                score, cpu = head(model, case, d[arm + "_final"])
                check(key, "public_head_" + arm, score, d[arm + "_scores"])
                check(key, "CPU64_head_" + arm, cpu, d[arm + "_scores"], 0.005)
                heads.append(
                    dict(
                        key=key,
                        arm=arm,
                        public_error=maximum(score, d[arm + "_scores"]),
                        CPU64_error=maximum(cpu, d[arm + "_scores"]),
                    )
                )
            bank[key] = d
            print("285 audited native", len(bank), "/32", flush=True)
    # No outcome is computed until every case's numerical qualification passes.
    natives, locals_, h1, effects, repairs, h2 = [], [], [], [], [], []
    for key, case in cases.items():
        d = bank[key]
        for arm in EVAL:
            n = endpoint(case, arm, d, "T", "F", True)
            v = vector(
                d[arm + "_write"].astype(float) - d["F_write"].astype(float),
                d["T_write"].astype(float) - d["F_write"].astype(float),
            )
            loc = dict(key=key, **case["group"], arm=arm, **v, passes=good(v, 0.10, 0.995))
            eff = endpoint(case, arm, d, "T", "M")
            rv = vector(
                d[arm + "_write"].astype(float) - d["M_write"].astype(float),
                d["T_write"].astype(float) - d["M_write"].astype(float),
            )
            repair = dict(key=key, **case["group"], arm=arm, **rv, passes=good(rv, 0.25, 0.95))
            natives.append(n)
            locals_.append(loc)
            effects.append(eff)
            repairs.append(repair)
            h1.append(
                dict(key=key, **case["group"], arm=arm, passes=bool(n["passes"] and loc["passes"]))
            )
            h2.append(
                dict(
                    key=key,
                    **case["group"],
                    arm=arm,
                    passes=bool(eff["passes"] and repair["passes"]),
                )
            )

    def counts(rows):
        return {a: sum(r["passes"] for r in rows if r["arm"] == a) for a in EVAL}

    def strata(rows):
        return {
            b: sum(r["passes"] for r in rows if r["arm"] == "C" and r["block"] == b)
            for b in sorted({c["group"]["block"] for c in cases.values()})
        }

    h1s, h2s = strata(h1), strata(h2)
    outputs = dict(
        native_checks=checks,
        native_heads=heads,
        native=natives,
        local=locals_,
        h1=h1,
        effect=effects,
        repair=repairs,
        h2=h2,
    )
    for name, rows in outputs.items():
        table(run / (name + ".csv"), rows)
    save_json(
        run / "native-audit.json",
        dict(
            status="passed",
            experiment=285,
            source_sha256=sha(Path(__file__)),
            design_sha256=sha(run / "design.json"),
            native_status_sha256=sha(run / "native-status.json"),
            native_started_sha256=sha(run / "native-started.json"),
            forecast_seal_sha256=sha(run / "forecast-seal.json"),
            numerical_checks=len(checks),
            finite_head_maps=len(heads),
            ordinary_True_replay_forwards=32,
            native_counts=counts(natives),
            local_counts=counts(locals_),
            H1_counts=counts(h1),
            H1_C_layouts=h1s,
            H1_C_passes=bool(counts(h1)["C"] >= 24 and min(h1s.values()) >= 6),
            effect_counts=counts(effects),
            repair_counts=counts(repairs),
            H2_counts=counts(h2),
            H2_C_layouts=h2s,
            H2_C_passes=bool(counts(h2)["C"] >= 24 and min(h2s.values()) >= 6),
            H2_margin_eligible=sum(r["margin_eligible"] for r in effects if r["arm"] == "T"),
            retained_forecast_raw_checks=fa["raw_checks"],
            csv_sha256={name + ".csv": sha(run / (name + ".csv")) for name in outputs},
            ended=stamp(),
            boundary="Structural transfer with a historical False-forecast donor bank. Substantial finite later-source/receiver computation; no learned-circuit, efficiency, novelty-priority or general Question B closure claim.",
        ),
    )
    print(
        {
            "H1": counts(h1),
            "H2": counts(h2),
            "H2_margin_eligible": sum(r["margin_eligible"] for r in effects if r["arm"] == "T"),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "native"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-279", type=Path, required=True)
    args = parser.parse_args()
    try:
        globals()[args.phase](args.run.resolve(), args.reference_279.resolve())
    except Exception as exc:
        target = args.run / ("failure-audit-" + args.phase + ".json")
        if args.run.exists() and not target.exists():
            save_json(target, dict(error=repr(exc), time=stamp()))
        raise
