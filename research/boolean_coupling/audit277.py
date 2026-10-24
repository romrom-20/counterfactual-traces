"""Separate pre-native numerical qualification, then frozen fresh-target evaluation277."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head, weights
from .core225b import array, load_model
from .run272 import stamp

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("C", "A", "L", "S", "G0", "G1", "G2", "Z", "FZ")
EVAL = (*ARMS[:-1], "T")


def maximum(a, b):
    return float(np.max(abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def bindings(run):
    design = read(run / "design.json")
    assert design["experiment"] == 277 and design["globally_pre_native"]
    assert tuple(design["arms"]) == ARMS and design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    return design, folder


def qualification(run):
    design, folder = bindings(run)
    assert not (run / "forecast-audit.json").exists() and not (run / "native-started.json").exists()
    status = read(run / "forecast-status.json")
    assert status["status"] == "complete" and status["ordinary_true"] == 0 and len(status["cases"]) == 32
    historical = set(read(ROOT / "research/boolean_coupling/historical-input-hashes-277.json")["hashes"])
    seen = set()
    for case in design["cases"]:
        hashes = [hashlib.sha256(__import__("json").dumps(case[n], separators=(",", ":")).encode()).hexdigest()
                  for n in ("false_ids", "true_ids")]
        assert hashes == case["input_hashes"] and not set(hashes) & (historical | seen)
        seen.update(hashes)
        changed = [i for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"], strict=True)) if a != b]
        assert changed == [case["position"]]
    model = load_model(True)
    layer = model.model.layers[0]
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 0, "mlp")
    tw = {n: torch.tensor(v, dtype=torch.float64) for n, v in w.items()}
    head = read_head(folder)
    checks, raw, derivatives, heads, scores = ([] for _ in range(5))

    def check(name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, point, node, a, b):
        error = maximum(a, b)
        raw.append(dict(key=key, point=point, node=node, error=error, original_bound=.0001, passes=error <= .0001))

    def smooth(x):
        z = x * torch.rsqrt(x.square().mean() + tw["eps"]) * tw["norm"]
        return F.linear(F.silu(F.linear(z, tw["gate_weight"])) * F.linear(z, tw["up_weight"]), tw["down_weight"])

    cases = {c["key"]: c for c in design["cases"]}
    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        start, length = case["position"], len(case["false_ids"])
        path = run / "forecasts" / (key + ".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert meta["ordinary_true"] == meta["source_True_MLP_primal_evaluations"] == 0
        assert meta["all_forecast_prefixes_exact"] and meta["source_only_install"] and meta["prepared_attention_only"]
        with np.load(path) as d, torch.no_grad():
            for prefix, ids in (("false", case["false_ids"]), ("prepared", case["true_ids"])):
                # First attention ONLY: no counterfactual source MLP1 target is evaluated.
                full = model.model.embed_tokens(torch.tensor([ids], device="mps"))
                z = layer.input_layernorm(full)
                cos, sin = (torch.tensor(d[n], device="mps") for n in ("rope_cos", "rope_sin"))
                q = layer.self_attn.q_proj(z).view(1, length, 12, 128).transpose(1, 2)
                k = layer.self_attn.k_proj(z).view(1, length, 2, 128).transpose(1, 2)
                q, k = qwen.apply_rotary_pos_emb(q, k, cos, sin)
                v = layer.self_attn.v_proj(z).view(1, length, 2, 128).transpose(1, 2)
                for name, val in dict(before=full[0, start], norm=z[0, start], q=q[0, :, start],
                    k=k[0, :, start], v=v[0, :, start], full_k=k[0], full_v=v[0]).items():
                    check("known_embedding_" + prefix + "_" + name, array(val), d[prefix + "_" + name])
                mask = torch.zeros((1, 1, length, length), device="mps")
                mask[0, 0, start] = torch.tensor(d[prefix + "_mask"], device="mps")
                y, p = qwen.eager_attention_forward(layer.self_attn, q, k, v, mask, scaling=128**-.5)
                check("public_probability", array(p[0, :, start]), d[prefix + "_p"], 1e-4)
                check("public_PV", array(y[0, start]), d[prefix + "_y"], 1e-4)
                write = layer.self_attn.o_proj(y.reshape(1, length, 1536))
                check("public_attention_write", array(write[0, start]), d[prefix + "_attwrite"], 1e-4)
                check("attention_mid_addition", d[prefix + "_before"] + d[prefix + "_attwrite"], d[prefix + "_mid"])
                kk, vv = d[prefix + "_full_k"].astype(float), d[prefix + "_full_v"].astype(float)
                qq, pp = d[prefix + "_q"].astype(float), d[prefix + "_p"].astype(float)
                logits = np.einsum("hd,hnd->hn", qq, np.repeat(kk, 6, axis=0)) * 128**-.5 + d[prefix + "_mask"].astype(float)
                softmax = np.exp(logits - logits.max(-1, keepdims=True))
                softmax /= softmax.sum(-1, keepdims=True)
                yy = np.einsum("hn,hnd->hd", pp, np.repeat(vv, 6, axis=0))
                rawcheck(key, prefix, "attention_softmax", softmax, pp)
                rawcheck(key, prefix, "attention_PV", yy, d[prefix + "_y"])
                ow = array(layer.self_attn.o_proj.weight).astype(float)
                rawcheck(key, prefix, "attention_output", d[prefix + "_y"].astype(float).reshape(-1) @ ow.T, d[prefix + "_attwrite"])
                if prefix == "prepared":
                    p[0, :, start] = torch.tensor(d["false_p"], device="mps")
                    for corner in ("W", "B"):
                        values = v.clone()
                        if corner == "B":
                            values[0, :, start] = torch.tensor(d["false_v"], device="mps")
                        cy = (p @ qwen.repeat_kv(values, 6)).transpose(1, 2).contiguous()
                        cw = layer.self_attn.o_proj(cy.reshape(1, length, 1536))
                        check("public_corner_PV", array(cy[0, start]), d["prepared_" + corner + "_y"], 1e-4)
                        check("public_corner_output", array(cw[0, start]), d["prepared_" + corner + "_write"], 1e-4)
            check("causal_prefix_K", d["prepared_full_k"][:, :start], d["false_full_k"][:, :start])
            check("causal_prefix_V", d["prepared_full_v"][:, :start], d["false_full_v"][:, :start])
            check("corner_B_y", d["prepared_B_y"], d["false_y"])
            check("corner_B_write", d["prepared_B_write"], d["false_attwrite"])
            embedding = d["prepared_before"].astype(float) - d["false_before"].astype(float)
            value = d["prepared_W_write"].astype(float) - d["prepared_B_write"].astype(float)
            xc = (d["false_mid"].astype(float) + embedding + value).astype(np.float32)
            r = d["prepared_mid"].astype(float) - xc.astype(float)
            points = dict(F=d["false_mid"], C=xc, R=(d["false_mid"].astype(float) + r).astype(np.float32), T=d["prepared_mid"])
            for name, point in points.items():
                check("point_provenance_" + name, point, d["point_" + name])
            for name, val in dict(embedding=embedding, self_value=value, carrier=xc.astype(float) - d["false_mid"].astype(float), read=r).items():
                check("component_provenance", val, d["component_" + name])
            for name in ("F", "C", "R"):
                full = torch.zeros((1, length, 1536), device="mps")
                full[0, start] = torch.tensor(points[name], device="mps")
                z = layer.post_attention_layernorm(full)
                g, u = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
                phi = layer.mlp.act_fn(g)
                product = phi * u
                write = layer.mlp.down_proj(product)
                for node, val in dict(x=full, z=z, gate=g, up=u, phi=phi, product=product, write=write).items():
                    check("public_MLP_" + node, array(val[0, start]), d["node_" + name + "_" + node])
                xt = torch.tensor(points[name].astype(float), dtype=torch.float64)
                zz = xt * torch.rsqrt(xt.square().mean() + tw["eps"]) * tw["norm"]
                gg, uu = F.linear(zz, tw["gate_weight"]), F.linear(zz, tw["up_weight"])
                ph = F.silu(gg)
                pp = ph * uu
                ww = F.linear(pp, tw["down_weight"])
                for node, val in dict(z=zz, gate=gg, up=uu, phi=ph, product=pp, write=ww).items():
                    rawcheck(key, name, node, val.numpy(), d["node_" + name + "_" + node])
                check("CPU64_write_bridge", ww.numpy(), d["node_" + name + "_write"], 1e-4)
            for name in ("z", "gate", "up", "phi", "product", "write"):
                check("False_observed_MLP", d["false_" + name], d["node_F_" + name])
            for anchor in ("F", "C"):
                xt, rt = (torch.tensor(v, dtype=torch.float64) for v in (points[anchor].astype(float), r))
                primal, jvp = torch.func.jvp(smooth, (xt,), (rt,))
                saved = d["derivative_" + anchor + "_write"]
                errors = dict(jvp_error=maximum(jvp.numpy(), saved))
                for eps in (1e-4, 1e-5):
                    fd = (smooth(xt + eps * rt) - smooth(xt - eps * rt)) / (2 * eps)
                    errors["fd_" + str(eps)] = maximum(fd.numpy(), saved)
                # no_grad from the public replay must not disable the reverse check
                with torch.enable_grad():
                    for number, function in enumerate((torch.sin, torch.cos)):
                        cot = function(torch.arange(1536, dtype=torch.float64) * (17 if number == 0 else 29))
                        variable = xt.clone().requires_grad_(True)
                        rev = torch.autograd.grad((smooth(variable) * cot).sum(), variable)[0] @ rt
                        errors["reverse_" + str(number)] = abs(float(rev) - float(cot @ torch.tensor(saved)))
                for name, error in errors.items():
                    check("smooth_CPU64_derivative_" + name, error, 0., 1e-4)
                derivatives.append(dict(key=key, anchor=anchor, bound=.0001, **errors))
                check("derivative_anchor", d["derivative_" + anchor + "_x"], points[anchor])
                check("derivative_direction", d["derivative_" + anchor + "_direction"], r)
                check("smooth_primal_bridge", primal.numpy(), d["node_" + anchor + "_write"], 1e-4)
            mf, mc, mr = (d["node_" + n + "_write"].astype(float) for n in ("F", "C", "R"))
            low, shifted = (mc + d["derivative_" + n + "_write"] for n in ("F", "C"))
            gain = shifted - low
            check("gain_identity", d["gain"], gain)
            proposals = dict(C=mc, A=mc + mr - mf, L=low, S=shifted, Z=mf, FZ=mf)
            for number in range(3):
                arm = "G" + str(number)
                seed = int(hashlib.sha256((key + f":276gain:{number}").encode()).hexdigest()[:16], 16)
                draw = np.random.default_rng(seed).normal(size=1536)
                draw *= np.linalg.norm(gain) / np.linalg.norm(draw)
                check("random_direction", draw, d["random_" + arm])
                check("random_norm", np.linalg.norm(draw), np.linalg.norm(gain), 1e-12)
                proposals[arm] = low + draw
            for arm, candidate in proposals.items():
                check("candidate_rule", candidate, d["proposal_" + arm])
                branch = "F" if arm in ("Z", "FZ") else "C"
                check("actual_branch", d[arm + "_used_point"], points[branch])
                check("actual_carrier_primal", d[arm + "_raw_write"], d["node_" + branch + "_write"])
                native = points["F" if arm == "FZ" else "T"]
                check("actual_attention_mid", d[arm + "_native_mid"], native)
                check("actual_install", d[arm + "_installed_write"], candidate.astype(np.float32))
                check("actual_source_addition", d[arm + "_source"][0], (native + candidate.astype(np.float32)).astype(np.float32))
            for node in ("source", "final", "scores"):
                check("zero_edit", d["FZ_" + node], d["F_" + node])
            for arm in ("F", *ARMS):
                full = torch.zeros((1, length, 1536), device="mps")
                full[0, start:] = torch.tensor(d[arm + "_final"], device="mps")
                expected = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check("finite_full_vocab_head", expected, d[arm + "_scores"])
                h = d[arm + "_final"][-1].astype(float)
                cpu = h / np.sqrt(np.mean(h * h) + head["final_eps"]) * head["final_norm"] @ head["final_head"].T
                check("CPU64_head_bridge", cpu, d[arm + "_scores"], .005)
                heads.append(dict(key=key, arm=arm, public_error=maximum(expected, d[arm + "_scores"]), cpu_error=maximum(cpu, d[arm + "_scores"])))
                scores.append(dict(key=key, arm=arm, false_score=float(d[arm + "_scores"][0]), true_score=float(d[arm + "_scores"][1]), margin=margin(d[arm + "_scores"])))
        print("277 pre-native audit", len(heads) // 10, "/32", flush=True)
    for name, records in dict(raw_primitives=raw, derivatives=derivatives, heads=heads, scores=scores).items():
        table(run / ("forecast-" + name.replace("_", "-") + ".csv"), records)
    save_json(run / "forecast-audit.json", dict(status="passed", experiment=277, ordinary_true=0,
        design_sha256=sha(run / "design.json"), status_sha256=sha(run / "forecast-status.json"), source_sha256=sha(__file__),
        numerical_checks=len(checks), numerical_max_error=max(c["error"] for c in checks), derivative_anchors=len(derivatives),
        raw_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        csv_sha256={p.name: sha(p) for p in run.glob("forecast-*.csv")}, ended=stamp(),
        boundary="All forecasts and this qualification precede native True. Smooth CPU64 local MLP derivatives only; old failures not recertified; raw coordinate outcomes retained."))


def outcomes(run):
    design, folder = bindings(run)
    assert not (run / "audit.json").exists()
    status, seal = read(run / "native-status.json"), read(run / "forecast-seal.json")
    assert status["status"] == "complete" and status["ordinary_true"] == 32
    assert status["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    assert seal["status_sha256"] == sha(run / "forecast-status.json") and seal["audit_sha256"] == sha(run / "forecast-audit.json")
    assert read(run / "forecast-audit.json")["ordinary_true"] == 0
    started = read(run / "native-started.json")
    assert started["ordinary_true_before"] == 0 and seal["created"] < started["started"]
    assert started["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    cases = {c["key"]: c for c in design["cases"]}
    forecasts = {c["key"]: c for c in seal["cases"]}
    model = load_model(True)
    layer, head = model.model.layers[0], read_head(folder)
    native, gains, local_gain, bank, heads, checks = [], [], [], {}, [], []
    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        start, length = case["position"], len(case["false_ids"])
        fp, tp = (run / part / (key + ".npz") for part in ("forecasts", "observations"))
        assert sha(fp) == forecasts[key]["arrays_sha256"] and sha(fp.with_suffix(".json")) == forecasts[key]["metadata_sha256"]
        assert sha(tp) == entry["arrays_sha256"] and sha(tp.with_suffix(".json")) == entry["metadata_sha256"]
        assert read(tp.with_suffix(".json"))["measured"] >= started["started"]
        with np.load(fp) as f, np.load(tp) as t, torch.no_grad():
            for name in ("before", "norm", "q", "k", "v", "full_k", "full_v", "p", "mask", "scaling", "y", "attwrite", "mid"):
                assert np.array_equal(t["native_" + name], f["prepared_" + name])
                checks.append("native_first_attention_exact")
            full = torch.zeros((1, length, 1536), device="mps")
            full[0, start] = torch.tensor(t["native_mid"], device="mps")
            z = layer.post_attention_layernorm(full)
            g, u = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
            phi = layer.mlp.act_fn(g)
            product = phi * u
            write = layer.mlp.down_proj(product)
            for name, value in dict(z=z, gate=g, up=u, phi=phi, product=product, write=write).items():
                assert np.array_equal(array(value[0, start]), t["native_" + name])
                checks.append("native_MLP_public_exact")
            assert np.array_equal(t["T_source"][0], (t["native_mid"] + t["native_write"]).astype(np.float32))
            full[0, start:] = torch.tensor(t["T_final"], device="mps")
            expected = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
            assert np.array_equal(expected, t["T_scores"])
            h = t["T_final"][-1].astype(float)
            cpu = h / np.sqrt(np.mean(h * h) + head["final_eps"]) * head["final_norm"] @ head["final_head"].T
            assert maximum(cpu, t["T_scores"]) <= .005
            heads.append(dict(key=key, public_error=maximum(expected, t["T_scores"]), cpu_error=maximum(cpu, t["T_scores"])))
            all_data = {"T": dict(source=t["T_source"][0].astype(float), final=t["T_final"].astype(float), margin=margin(t["T_scores"]))}
            all_data.update({arm: dict(source=f[arm + "_source"][0].astype(float), final=f[arm + "_final"].astype(float), margin=margin(f[arm + "_scores"])) for arm in ("F", *ARMS)})
            bank[key] = all_data
            target = all_data["T"]["final"] - all_data["F"]["final"]
            correction = all_data["T"]["final"] - all_data["L"]["final"]
            mt, ml = all_data["T"]["margin"], all_data["L"]["margin"]
            low = f["proposal_L"]
            for arm in EVAL:
                nrow, grow = (dict(key=key, **case["group"], arm=arm) for _ in range(2))
                nok, gok = [], []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    n = vector((all_data[arm]["final"] - all_data["F"]["final"])[pick], target[pick])
                    g = vector((all_data[arm]["final"] - all_data["L"]["final"])[pick], correction[pick])
                    nrow.update({scope + "_" + k: v for k, v in n.items()})
                    grow.update({scope + "_" + k: v for k, v in g.items()})
                    nok.append(n["norm"] >= .001 and n["relative"] <= .05 and n["cosine"] >= .99)
                    gok.append(g["norm"] >= .001 and g["relative"] <= .25 and g["cosine"] >= .95)
                err = abs(all_data[arm]["margin"] - mt)
                nrow.update(margin_error=err, passes=bool(all(nok) and err <= .1))
                grow.update(margin_error=err, target_margin=mt - ml, margin_bound=.005 + .25 * abs(mt - ml),
                    passes=bool(all(gok) and abs(mt - ml) >= .02 and err <= .005 + .25 * abs(mt - ml)))
                native.append(nrow)
                gains.append(grow)
                candidate = t["native_write"].astype(float) if arm == "T" else f["proposal_" + arm]
                v = vector(candidate - low, t["native_write"].astype(float) - low)
                local_gain.append(dict(key=key, **case["group"], arm=arm, **v,
                    passes=bool(v["norm"] >= .001 and v["relative"] <= .25 and v["cosine"] >= .95)))
        print("277 native audit", len(bank), "/32", flush=True)
    pairs, combined, explanation = [], [], []
    for key, own in bank.items():
        donor = bank[design["partners"][key]]
        ts = (own["T"]["source"] - own["F"]["source"]) - (donor["T"]["source"] - donor["F"]["source"])
        tv = (own["T"]["final"][-1] - own["F"]["final"][-1]) - (donor["T"]["final"][-1] - donor["F"]["final"][-1])
        tm = (own["T"]["margin"] - own["F"]["margin"]) - (donor["T"]["margin"] - donor["F"]["margin"])
        for arm in EVAL:
            s = vector((own[arm]["source"] - own["F"]["source"]) - (donor[arm]["source"] - donor["F"]["source"]), ts)
            v = vector((own[arm]["final"][-1] - own["F"]["final"][-1]) - (donor[arm]["final"][-1] - donor["F"]["final"][-1]), tv)
            m = (own[arm]["margin"] - own["F"]["margin"]) - (donor[arm]["margin"] - donor["F"]["margin"])
            good = all(x["norm"] >= .001 and x["relative"] <= .25 and x["cosine"] >= .95 for x in (s, v))
            row = dict(key=key, donor=design["partners"][key], **cases[key]["group"], arm=arm,
                **{"source_" + k: x for k, x in s.items()}, **{"verdict_" + k: x for k, x in v.items()},
                target_margin=tm, margin_error=abs(m - tm), margin_bound=.005 + .25 * abs(tm),
                passes=bool(good and abs(tm) >= .02 and abs(m - tm) <= .005 + .25 * abs(tm)))
            pairs.append(row)
            nr = next(r for r in native if r["key"] == key and r["arm"] == arm)
            combined.append(dict(key=key, schema=cases[key]["group"]["schema"], reverse=cases[key]["group"]["reverse"], arm=arm,
                native_pass=nr["passes"], paired_pass=row["passes"], passes=nr["passes"] and row["passes"]))
        gain = next(r for r in gains if r["key"] == key and r["arm"] == "S")
        local = next(r for r in local_gain if r["key"] == key and r["arm"] == "S")
        failures = sum(not r["passes"] for r in gains if r["key"] == key and r["arm"] in ("G0", "G1", "G2"))
        explanation.append(dict(key=key, schema=cases[key]["group"]["schema"], reverse=cases[key]["group"]["reverse"],
            local_pass=local["passes"], downstream_pass=gain["passes"], random_failures=failures,
            passes=bool(local["passes"] and gain["passes"] and failures >= 2)))
    for name, records in dict(native=native, paired=pairs, combined=combined, gain=gains,
        local_gain=local_gain, explanation=explanation, heads=heads).items():
        table(run / (name.replace("_", "-") + ".csv"), records)
    def strata(records, arm=None):
        return {f"{s}/{rev}": sum(r["passes"] for r in records if r["schema"] == s and r["reverse"] == rev and (arm is None or r["arm"] == arm))
                for s in ("generic", "descriptive") for rev in (0, 1)}
    nc = {a: sum(r["passes"] for r in native if r["arm"] == a) for a in EVAL}
    cc = {a: sum(r["passes"] for r in combined if r["arm"] == a) for a in EVAL}
    ss, es = strata(combined, "S"), strata(explanation)
    ec = sum(r["passes"] for r in explanation)
    primary, gain = cc["S"] >= 24 and min(ss.values()) >= 6, ec >= 24 and min(es.values()) >= 6
    save_json(run / "audit.json", dict(status="passed", experiment=277, prospective=True,
        design_sha256=sha(run / "design.json"), forecast_seal_sha256=sha(run / "forecast-seal.json"),
        native_status_sha256=sha(run / "native-status.json"), source_sha256=sha(__file__), public_receipt_commit=started["public_commit"],
        native_counts=nc, combined_counts=cc, S_strata=ss, explanation_count=ec, explanation_strata=es,
        primary_passes=primary, gain_explanation_passes=gain, generative_account_passes=primary and gain,
        native_ordinary_true=32, exact_native_checks=len(checks), csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Unchanged single-site rule, fresh literals/dependent templates, all other model computation finite. Failed276 gain explanation remains failed. No compact full predictor, learned circuit or general QB solution."))
    print(dict(native=nc, combined=cc, primary=primary, gain_explanation=ec), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "native"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        (qualification if args.phase == "forecast" else outcomes)(args.run.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / ("failure-audit-" + args.phase + ".json")).exists():
            save_json(args.run / ("failure-audit-" + args.phase + ".json"), dict(error=repr(exc), time=stamp()))
        raise
