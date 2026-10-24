"""Independent polynomial/selection qualification before native, then fresh endpoints279."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .core225b import array, load_model
from .run272 import stamp

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("L", "Q", "C", "S", "R0", "R1", "R2", "Z", "FZ")
EVAL = (*ARMS[:-1], "T")


def maximum(a, b):
    return float(np.max(abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def bindings(run):
    design = read(run / "design.json")
    assert design["experiment"] == 279 and design["globally_pre_native"] and tuple(design["arms"]) == ARMS
    assert design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    folder, checkpoint = locate(ROOT)
    assert design["checkpoint"] == checkpoint
    return design, folder


def differentiated(g):
    """Coordinate derivatives from independent Torch autograd of SiLU itself."""
    with torch.enable_grad():
        x = torch.tensor(g, dtype=torch.float64, requires_grad=True)
        d1 = torch.autograd.grad(F.silu(x).sum(), x, create_graph=True)[0]
        d2 = torch.autograd.grad(d1.sum(), x, create_graph=True)[0]
        d3 = torch.autograd.grad(d2.sum(), x)[0]
    return {"d1": d1.detach().numpy(), "d2": d2.detach().numpy(), "d3": d3.detach().numpy()}


def coefficient_checks(g, saved):
    independent = differentiated(g.astype(float))
    errors = {name + "_AD": maximum(saved[name], independent[name]) for name in independent}
    for eps in (1e-4, 1e-5):
        xp, xm = torch.tensor(g.astype(float) + eps), torch.tensor(g.astype(float) - eps)
        fd1 = ((F.silu(xp) - F.silu(xm)) / (2 * eps)).numpy()
        plus, minus = differentiated(g.astype(float) + eps), differentiated(g.astype(float) - eps)
        for name, fd in (("d1", fd1), ("d2", (plus["d1"] - minus["d1"]) / (2 * eps)), ("d3", (plus["d2"] - minus["d2"]) / (2 * eps))):
            errors[name + "_FD_" + str(eps)] = maximum(saved[name], fd)
    assert all(np.isfinite(e) and e <= 1e-4 for e in errors.values()), errors
    return independent, errors


def qualification(run):
    design, folder = bindings(run)
    assert not (run / "forecast-audit.json").exists() and not (run / "native-started.json").exists()
    status = read(run / "forecast-status.json")
    assert status["status"] == "complete" and status["ordinary_true"] == 0 and len(status["cases"]) == 32
    historical = set(read(ROOT / "research/boolean_coupling/historical-input-hashes-279.json")["hashes"])
    seen = set()
    for case in design["cases"]:
        hs = [hashlib.sha256(json.dumps(case[n], separators=(",", ":")).encode()).hexdigest() for n in ("false_ids", "true_ids")]
        assert hs == case["input_hashes"] and not set(hs) & (historical | seen)
        seen.update(hs)
        changes = [i for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"], strict=True)) if a != b]
        assert changes == [case["position"]]
    model, head = load_model(True), read_head(folder)
    layer = model.model.layers[0]
    column_norm = np.linalg.norm(array(layer.mlp.down_proj.weight).astype(float), axis=0)
    checks, coefficients, raw, heads, scores = ([] for _ in range(5))
    cases = {c["key"]: c for c in design["cases"]}

    def check(name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, arm, node, a, b):
        error = maximum(a, b)
        raw.append(dict(key=key, arm=arm, node=node, error=error, original_bound=.0001, passes=error <= .0001))

    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        start, length = case["position"], len(case["false_ids"])
        path = run / "forecasts" / (key + ".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert meta["ordinary_true"] == meta["full_True_source_activation_primal_count"] == 0
        assert meta["all_prefixes_exact"] and meta["False_zero_exact"] and meta["False_source_primal_for_all_forecasts"]
        assert meta["only_first_source_MLP_write_changed"] and meta["controller_current_coordinates_per_mask"] == 90
        with np.load(path) as d, torch.no_grad():
            for prefix, ids in (("false", case["false_ids"]), ("prepared", case["true_ids"])):
                full = model.model.embed_tokens(torch.tensor([ids], device="mps"))
                z = layer.input_layernorm(full)
                cos, sin = (torch.tensor(d[n], device="mps") for n in ("rope_cos", "rope_sin"))
                q = layer.self_attn.q_proj(z).view(1, length, 12, 128).transpose(1, 2)
                k = layer.self_attn.k_proj(z).view(1, length, 2, 128).transpose(1, 2)
                q, k = qwen.apply_rotary_pos_emb(q, k, cos, sin)
                v = layer.self_attn.v_proj(z).view(1, length, 2, 128).transpose(1, 2)
                for name, val in dict(before=full[0, start], norm=z[0, start], q=q[0, :, start], k=k[0, :, start], v=v[0, :, start], full_k=k[0], full_v=v[0]).items():
                    check("known_embedding_" + name, array(val), d[prefix + "_" + name])
                mask = torch.zeros((1, 1, length, length), device="mps")
                mask[0, 0, start] = torch.tensor(d[prefix + "_mask"], device="mps")
                y, p = qwen.eager_attention_forward(layer.self_attn, q, k, v, mask, scaling=128**-.5)
                check("source_probability", array(p[0, :, start]), d[prefix + "_p"], 1e-4)
                check("source_PV", array(y[0, start]), d[prefix + "_y"], 1e-4)
                out = layer.self_attn.o_proj(y.reshape(1, length, 1536))
                check("source_attention_write", array(out[0, start]), d[prefix + "_attwrite"], 1e-4)
                check("mid_addition", d[prefix + "_before"] + d[prefix + "_attwrite"], d[prefix + "_mid"])
                # Only norm/gate/up at current input; never full current SiLU here.
                point = torch.zeros((1, length, 1536), device="mps")
                point[0, start] = torch.tensor(d[prefix + "_mid"], device="mps")
                zz = layer.post_attention_layernorm(point)
                gg, uu = layer.mlp.gate_proj(zz), layer.mlp.up_proj(zz)
                for name, val in dict(z=zz, gate=gg, up=uu).items():
                    check("source_MLP_input_" + name, array(val[0, start]), d[prefix + "_" + name])
                if prefix == "false":
                    phi = layer.mlp.act_fn(gg)
                    prod = phi * uu
                    ww = layer.mlp.down_proj(prod)
                    for name, val in dict(phi=phi, product=prod, write=ww).items():
                        check("observed_False_MLP", array(val[0, start]), d["false_" + name])
            check("causal_prefix_K", d["prepared_full_k"][:, :start], d["false_full_k"][:, :start])
            check("causal_prefix_V", d["prepared_full_v"][:, :start], d["false_full_v"][:, :start])
            coeff, errs = coefficient_checks(d["false_gate"], {n: d["derivatives_" + n] for n in ("d1", "d2", "d3")})
            coefficients.append(dict(key=key, bound=.0001, **errs))
            for name, error in errs.items():
                check("False_coefficient_" + name, error, 0., 1e-4)
            check("False_SiLU_Double_primal", F.silu(torch.tensor(d["false_gate"].astype(float))).numpy(), d["false_phi"], 1e-4)
            dg = d["prepared_gate"].astype(float) - d["false_gate"].astype(float)
            check("gate_displacement", d["dg"], dg)
            check("output_column_norm", d["column_norm"], column_norm)
            score = abs(dg)**4 * abs(d["prepared_up"].astype(float)) * column_norm
            check("mask_score", d["mask_score"], score)
            mask = np.sort(np.lexsort((np.arange(8960), -score))[:90])
            expected_phi = {}
            expected_phi["L"] = d["false_phi"].astype(float) + coeff["d1"] * dg
            expected_phi["Q"] = expected_phi["L"] + .5 * coeff["d2"] * dg**2
            expected_phi["C"] = expected_phi["Q"] + coeff["d3"] * dg**3 / 6
            for arm in ("S", "R0", "R1", "R2"):
                if arm == "S":
                    indices = mask
                else:
                    number = int(arm[-1])
                    seed = int(hashlib.sha256((key + f":279-mask:{number}").encode()).hexdigest()[:16], 16)
                    indices = np.sort(np.random.default_rng(seed).choice(8960, 90, replace=False))
                check("selected_indices", d["masks_" + arm], indices)
                mixed = d["false_gate"].copy()
                mixed[indices] = d["prepared_gate"][indices]
                check("only_selected_current_gates", d["mixed_inputs_" + arm], mixed)
                full = torch.zeros((1, length, 8960), device="mps")
                full[0, start] = torch.tensor(mixed, device="mps")
                selected = array(layer.mlp.act_fn(full)[0, start])[indices]
                check("partial_current_activation", d["selected_phi_" + arm], selected)
                check("selected_Double_bridge", F.silu(torch.tensor(d["prepared_gate"][indices].astype(float))).numpy(), selected, 1e-4)
                expected_phi[arm] = expected_phi["C"].copy()
                expected_phi[arm][indices] = selected
            down = array(layer.mlp.down_proj.weight).astype(float)
            for arm, phi in expected_phi.items():
                check("independent_polynomial_phi", d["phi_" + arm], phi.astype(np.float32), 1e-4)
                ph = torch.zeros((1, length, 8960), device="mps")
                up = torch.zeros_like(ph)
                ph[0, start] = torch.tensor(d["phi_" + arm], device="mps")
                up[0, start] = torch.tensor(d["prepared_up"], device="mps")
                prod = ph * up
                write = layer.mlp.down_proj(prod)
                check("public_finite_product", array(prod[0, start]), d["products_" + arm])
                check("public_down_projection", array(write[0, start]), d["writes_" + arm])
                rawcheck(key, arm, "phi", phi, d["phi_" + arm])
                rawcheck(key, arm, "product", phi * d["prepared_up"].astype(float), d["products_" + arm])
                rawcheck(key, arm, "write", (phi * d["prepared_up"].astype(float)) @ down.T, d["writes_" + arm])
            for arm in ("Z", "FZ"):
                check("zero_write", d["writes_" + arm], d["false_write"])
            for arm in ARMS:
                expected_mid = d["false_mid"] if arm == "FZ" else d["prepared_mid"]
                check("native_outer_mid", d[arm + "_actual_native_mid"], expected_mid)
                check("False_branch_point", d[arm + "_actual_used_point"], d["false_mid"])
                check("False_branch_primal", d[arm + "_actual_False_primal_write"], d["false_write"])
                check("actual_write", d[arm + "_actual_installed_write"], d["writes_" + arm])
                check("source_addition", d[arm + "_source"], (expected_mid + d["writes_" + arm]).astype(np.float32))
            check("False_source_addition", d["F_source"], (d["false_mid"] + d["false_write"]).astype(np.float32))
            for name in ("source", "final", "scores"):
                check("zero_edit_full", d["FZ_" + name], d["F_" + name])
            for arm in ("F", *ARMS):
                full = torch.zeros((1, length, 1536), device="mps")
                full[0, start:] = torch.tensor(d[arm + "_final"], device="mps")
                expected = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check("public_full_vocab_head", expected, d[arm + "_scores"])
                h = d[arm + "_final"][-1].astype(float)
                cpu = h / np.sqrt(np.mean(h * h) + head["final_eps"]) * head["final_norm"] @ head["final_head"].T
                check("CPU64_head_bridge", cpu, d[arm + "_scores"], .005)
                heads.append(dict(key=key, arm=arm, public_error=maximum(expected, d[arm + "_scores"]), cpu_error=maximum(cpu, d[arm + "_scores"])))
                scores.append(dict(key=key, arm=arm, false_score=float(d[arm + "_scores"][0]), true_score=float(d[arm + "_scores"][1]), margin=margin(d[arm + "_scores"])))
        print("279 pre-native audit", len(coefficients), "/32", flush=True)
    for name, records in dict(coefficients=coefficients, raw=raw, heads=heads, scores=scores).items():
        table(run / ("forecast-" + name + ".csv"), records)
    save_json(run / "forecast-audit.json", dict(status="passed", experiment=279, ordinary_true=0,
        design_sha256=sha(run / "design.json"), status_sha256=sha(run / "forecast-status.json"), source_sha256=sha(__file__),
        numerical_checks=len(checks), numerical_max_error=max(r["error"] for r in checks), False_gate_anchors=len(coefficients),
        coefficient_vectors_checked=3 * len(coefficients), raw_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        csv_sha256={p.name: sha(p) for p in run.glob("forecast-*.csv")}, ended=stamp(),
        boundary="No complete native True source activation before seal. Defined False coefficients independently differentiated; exact public product/down/head checks. Raw CPU64 forecast differences retained, not whole-model derivative validation."))


def outcomes(run):
    design, folder = bindings(run)
    assert not (run / "audit.json").exists()
    status, seal, started = (read(run / n) for n in ("native-status.json", "forecast-seal.json", "native-started.json"))
    assert status["status"] == "complete" and status["ordinary_true"] == 32
    assert seal["ordinary_true"] == started["ordinary_true_before"] == 0 and seal["created"] < started["started"]
    assert status["forecast_seal_sha256"] == started["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    assert seal["status_sha256"] == sha(run / "forecast-status.json") and seal["audit_sha256"] == sha(run / "forecast-audit.json")
    cases = {c["key"]: c for c in design["cases"]}
    forecasts = {c["key"]: c for c in seal["cases"]}
    model, head = load_model(True), read_head(folder)
    native, effect, local, correction, bank, heads = [], [], [], [], {}, []
    checks = 0
    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        start, length = case["position"], len(case["false_ids"])
        fp, tp = (run / part / (key + ".npz") for part in ("forecasts", "observations"))
        assert sha(fp) == forecasts[key]["arrays_sha256"] and sha(fp.with_suffix(".json")) == forecasts[key]["metadata_sha256"]
        assert sha(tp) == entry["arrays_sha256"] and sha(tp.with_suffix(".json")) == entry["metadata_sha256"]
        assert read(tp.with_suffix(".json"))["measured"] >= started["started"]
        with np.load(fp) as f, np.load(tp) as t, torch.no_grad():
            for name in ("before", "norm", "q", "k", "v", "full_k", "full_v", "p", "mask", "scaling", "y", "attwrite", "mid", "z", "gate", "up"):
                assert np.array_equal(t["native_" + name], f["prepared_" + name]), name
                checks += 1
            layer = model.model.layers[0]
            point = torch.zeros((1, length, 1536), device="mps")
            point[0, start] = torch.tensor(t["native_mid"], device="mps")
            z = layer.post_attention_layernorm(point)
            g, u = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
            phi = layer.mlp.act_fn(g)
            prod = phi * u
            write = layer.mlp.down_proj(prod)
            for name, val in dict(z=z, gate=g, up=u, phi=phi, product=prod, write=write).items():
                assert np.array_equal(array(val[0, start]), t["native_" + name])
                checks += 1
            assert np.array_equal(t["T_source"], (t["native_mid"] + t["native_write"]).astype(np.float32))
            for arm in ("S", "R0", "R1", "R2"):
                assert np.array_equal(t["native_phi"][f["masks_" + arm]], f["selected_phi_" + arm])
                checks += 1
            full = torch.zeros((1, length, 1536), device="mps")
            full[0, start:] = torch.tensor(t["T_final"], device="mps")
            expected = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
            assert np.array_equal(expected, t["T_scores"])
            h = t["T_final"][-1].astype(float)
            cpu = h / np.sqrt(np.mean(h * h) + head["final_eps"]) * head["final_norm"] @ head["final_head"].T
            assert maximum(cpu, t["T_scores"]) <= .005
            heads.append(dict(key=key, public_error=0., cpu_error=maximum(cpu, t["T_scores"])))
            own = {"T": dict(source=t["T_source"].astype(float), final=t["T_final"].astype(float), margin=margin(t["T_scores"]), write=t["native_write"].astype(float))}
            own.update({arm: dict(source=f[arm + "_source"].astype(float), final=f[arm + "_final"].astype(float),
                margin=margin(f[arm + "_scores"]), write=f["false_write"].astype(float) if arm == "F" else f["writes_" + arm].astype(float)) for arm in ("F", *ARMS)})
            bank[key] = own
            mt, mz, mc = (own[a]["margin"] for a in ("T", "Z", "C"))
            for arm in EVAL:
                nr, er, cr = (dict(key=key, **case["group"], arm=arm) for _ in range(3))
                ngood, egood, cgood = [], [], []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    n = vector((own[arm]["final"] - own["F"]["final"])[pick], (own["T"]["final"] - own["F"]["final"])[pick])
                    e = vector((own[arm]["final"] - own["Z"]["final"])[pick], (own["T"]["final"] - own["Z"]["final"])[pick])
                    c = vector((own[arm]["final"] - own["C"]["final"])[pick], (own["T"]["final"] - own["C"]["final"])[pick])
                    for row, val in ((nr, n), (er, e), (cr, c)):
                        row.update({scope + "_" + k: v for k, v in val.items()})
                    ngood.append(n["norm"] >= .001 and n["relative"] <= .05 and n["cosine"] >= .99)
                    egood.append(e["norm"] >= .001 and e["relative"] <= .25 and e["cosine"] >= .95)
                    cgood.append(c["norm"] >= .001 and c["relative"] <= .25 and c["cosine"] >= .95)
                error = abs(own[arm]["margin"] - mt)
                nr.update(margin_error=error, passes=bool(all(ngood) and error <= .1))
                er.update(margin_error=error, target_margin=mt - mz, margin_bound=.005 + .25 * abs(mt - mz), passes=bool(all(egood) and abs(mt - mz) >= .02 and error <= .005 + .25 * abs(mt - mz)))
                cr.update(margin_error=error, target_margin=mt - mc, margin_bound=.005 + .25 * abs(mt - mc), passes=bool(all(cgood) and abs(mt - mc) >= .02 and error <= .005 + .25 * abs(mt - mc)))
                native.append(nr)
                effect.append(er)
                correction.append(cr)
                v = vector(own[arm]["write"] - own["F"]["write"], own["T"]["write"] - own["F"]["write"])
                c = vector(own[arm]["write"] - own["C"]["write"], own["T"]["write"] - own["C"]["write"])
                local.append(dict(key=key, **case["group"], arm=arm, **v, correction_norm=c["norm"], correction_relative=c["relative"], correction_cosine=c["cosine"],
                    passes=bool(v["norm"] >= .001 and v["relative"] <= .1 and v["cosine"] >= .995),
                    correction_pass=bool(c["norm"] >= .001 and c["relative"] <= .25 and c["cosine"] >= .95)))
        print("279 native audit", len(bank), "/32", flush=True)
    paired, combined, restoration = [], [], []
    for key, own in bank.items():
        donor = bank[design["partners"][key]]
        ts = (own["T"]["source"] - own["F"]["source"]) - (donor["T"]["source"] - donor["F"]["source"])
        tv = (own["T"]["final"][-1] - own["F"]["final"][-1]) - (donor["T"]["final"][-1] - donor["F"]["final"][-1])
        tm = (own["T"]["margin"] - own["F"]["margin"]) - (donor["T"]["margin"] - donor["F"]["margin"])
        for arm in EVAL:
            s = vector((own[arm]["source"] - own["F"]["source"]) - (donor[arm]["source"] - donor["F"]["source"]), ts)
            v = vector((own[arm]["final"][-1] - own["F"]["final"][-1]) - (donor[arm]["final"][-1] - donor["F"]["final"][-1]), tv)
            m = (own[arm]["margin"] - own["F"]["margin"]) - (donor[arm]["margin"] - donor["F"]["margin"])
            ok = all(x["norm"] >= .001 and x["relative"] <= .25 and x["cosine"] >= .95 for x in (s, v)) and abs(tm) >= .02 and abs(m - tm) <= .005 + .25 * abs(tm)
            row = dict(key=key, **cases[key]["group"], arm=arm, **{"source_" + k: x for k, x in s.items()}, **{"verdict_" + k: x for k, x in v.items()}, target_margin=tm, margin_error=abs(m - tm), margin_bound=.005 + .25 * abs(tm), passes=bool(ok))
            paired.append(row)
            n, e, loc = (next(r for r in rows if r["key"] == key and r["arm"] == arm) for rows in (native, effect, local))
            combined.append(dict(key=key, schema=row["schema"], reverse=row["reverse"], arm=arm, native_pass=n["passes"], effect_pass=e["passes"], local_pass=loc["passes"], paired_pass=row["passes"], passes=bool(n["passes"] and e["passes"] and loc["passes"] and ok)))
            corr = next(r for r in correction if r["key"] == key and r["arm"] == arm)
            failures = sum(not r["passes"] for r in correction if r["key"] == key and r["arm"] in ("R0", "R1", "R2"))
            restoration.append(dict(key=key, schema=row["schema"], reverse=row["reverse"], arm=arm, local_pass=loc["correction_pass"], downstream_pass=corr["passes"], random_failures=failures, passes=bool(loc["correction_pass"] and corr["passes"] and failures >= 2)))
    for name, records in dict(native=native, effect=effect, local=local, correction=correction, paired=paired, combined=combined, restoration=restoration, heads=heads).items():
        table(run / (name + ".csv"), records)
    def counts(rows):
        return {a: sum(r["passes"] for r in rows if r["arm"] == a) for a in EVAL}
    def strata(rows, arm):
        return {f"{s}/{rev}": sum(r["passes"] for r in rows if r["schema"] == s and r["reverse"] == rev and r["arm"] == arm) for s in ("generic", "descriptive") for rev in (0, 1)}
    cc, rc = counts(combined), counts(restoration)
    primary = cc["C"] >= 24 and min(strata(combined, "C").values()) >= 6
    sparse = cc["S"] >= 24 and min(strata(combined, "S").values()) >= 6
    sparse_account = rc["S"] >= 24 and min(strata(restoration, "S").values()) >= 6
    save_json(run / "audit.json", dict(status="passed", experiment=279, prospective=True,
        design_sha256=sha(run / "design.json"), forecast_seal_sha256=sha(run / "forecast-seal.json"), native_status_sha256=sha(run / "native-status.json"), source_sha256=sha(__file__),
        public_receipt_commit=started["public_commit"], native_counts=counts(native), effect_counts=counts(effect), local_counts=counts(local), paired_counts=counts(paired), combined_counts=cc, restoration_counts=rc,
        C_strata=strata(combined, "C"), S_strata=strata(combined, "S"), S_restoration_strata=strata(restoration, "S"),
        primary_passes=primary, sparse_prediction_passes=sparse, sparse_correction_account_passes=sparse_account,
        exact_native_operand_checks=checks, ordinary_true=32, csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Fresh literals/dependent templates, source-only activation rule. No priority, learned circuit, useful whole-network efficiency or general Q B solution. Failed278 normalization-only claim and277 gain account stay failed."))
    print(dict(native=counts(native), combined=cc, primary=primary, sparse=sparse, sparse_correction_account=sparse_account), flush=True)


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
