"""Independent native-source attention operand audit and frozen decisions for275."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .run272 import stamp
from .source273 import paired

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARMS = ("F", "N", "W", "V", "B", "FB")
VALUE_ARMS = ("U", "W", "V", "B", "Z", "FZ", "T")


def maximum(a, b):
    return float(np.max(abs(a.astype(float) - b.astype(float))))


def projection(value, target):
    return float(np.vdot(value.ravel(), target.ravel())) / max(float(np.vdot(target.ravel(), target.ravel())), 1e-30)


def main(run, reference):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 275 and design["retrospective"] and status["status"] == "complete"
    assert design["cases"] == read(reference / "design.json")["cases"]
    assert design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    for name, h in design["reference_hashes"].items():
        assert sha(reference / name) == h
    _, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    original = {r["key"]: r for r in read(reference / "status.json")["cases"]}
    cases = {c["key"]: c for c in design["cases"]}
    checks, raw, native, effects, emission, layers, heads, ledger, local = ([] for _ in range(9))
    parts = {}

    def check(name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, arm, i, node, a, b):
        error = maximum(a, b)
        raw.append(dict(key=key, arm=arm, layer=i + 1, node=node, error=error,
                        original_bound=.0001, passes=error <= .0001))

    model = load_model(True)
    norms = [array(layer.input_layernorm.weight).astype(float) for layer in model.model.layers]
    vweights = [array(layer.self_attn.v_proj.weight).astype(float) for layer in model.model.layers]
    vbias = [array(layer.self_attn.v_proj.bias).astype(float) for layer in model.model.layers]
    oweights = [array(layer.self_attn.o_proj.weight).astype(float) for layer in model.model.layers]
    final_gamma = array(model.model.norm.weight).astype(float)
    final_w = array(model.lm_head.weight[[895, 830]]).astype(float)
    eps = model.config.rms_norm_eps
    with torch.no_grad():
        for entry in status["cases"]:
            key = entry["key"]
            case = cases[key]
            start, length = case["position"], len(case["false_ids"])
            path = run / "cases" / (key + ".npz")
            assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            meta = read(path.with_suffix(".json"))
            assert all(meta[k] for k in ("false_and_native_exact", "unchanged_prefix_all_layers", "source_query_only_install",
                "native_corner_B_matches_F", "native_value_KV_exact", "self_zero_archived_exact", "full_factor_zero_edit_exact"))
            old_path = reference / "cases" / (key + ".npz")
            assert sha(old_path) == original[key]["arrays_sha256"] == meta["input_sha256"]
            with np.load(old_path) as old, np.load(path) as data:
                for a, b in (("F", "F"), ("N", "T"), ("value_U", "T"), ("value_T", "T"), ("value_Z", "Z"), ("value_FZ", "F"), ("FB", "F")):
                    for node in ("states", "scores"):
                        check("archived_" + a + "_" + node, data[a + "_" + node], old[b + "_" + node])
                for a, b in (("F", "F"), ("N", "G")):
                    for n, m in (("norm", "norm"), ("k", "K"), ("v", "V")):
                        check("native_" + a + "_" + n, data[a + "_" + n], old[b + "_" + m])
                cos = torch.tensor(data["rope_cos"], device="mps")
                sin = torch.tensor(data["rope_sin"], device="mps")
                for arm in SOURCE_ARMS:
                    before, states = data[arm + "_before"], data[arm + "_states"]
                    check("initial_edit_" + arm, before[0], array(model.model.embed_tokens.weight[895 if arm in ("F", "FB") else 830]))
                    check("carried_source_" + arm, before[1:], states[:-1, 0])
                    for i, layer in enumerate(model.model.layers):
                        def node(name):
                            return data[arm + "_" + name][i]

                        full = torch.zeros((1, length, 1536), device="mps")
                        full[0, start] = torch.tensor(before[i], device="mps")
                        z = layer.input_layernorm(full)
                        check("public_norm", array(z[0, start]), node("norm"))
                        q = layer.self_attn.q_proj(z).view(1, length, 12, 128).transpose(1, 2)
                        k = layer.self_attn.k_proj(z).view(1, length, 2, 128).transpose(1, 2)
                        q, k = qwen.apply_rotary_pos_emb(q, k, cos, sin)
                        v = layer.self_attn.v_proj(z).view(1, length, 2, 128).transpose(1, 2)
                        check("public_source_Q", array(q[0, :, start]), node("q"))
                        check("public_source_K", array(k[0, :, start]), node("k"))
                        check("public_source_V", array(v[0, :, start]), node("v"))
                        kfull = torch.tensor(data["F_full_k"][i][None], device="mps")
                        vfull = torch.tensor(data["F_full_v"][i][None], device="mps")
                        kfull[0, :, start] = k[0, :, start]
                        vfull[0, :, start] = v[0, :, start]
                        mask = torch.zeros((1, 1, length, length), device="mps")
                        mask[0, 0, start] = torch.tensor(node("mask"), device="mps")
                        _, prob = qwen.eager_attention_forward(layer.self_attn, q, kfull, vfull, mask, scaling=float(node("scaling")))
                        check("public_source_probability", array(prob[0, :, start]), node("p"), 1e-4)
                        expected_p = data["F_p"][i] if arm in ("W", "B", "FB") else node("p")
                        expected_v = data["F_v"][i] if arm in ("V", "B", "FB") else node("v")
                        check("probability_clamp", node("used_p"), expected_p)
                        check("self_value_clamp", node("used_self_v"), expected_v)
                        prob[0, :, start] = torch.tensor(expected_p, device="mps")
                        vfull[0, :, start] = torch.tensor(expected_v, device="mps")
                        y = (prob @ qwen.repeat_kv(vfull, 6)).transpose(1, 2).contiguous()
                        write = layer.self_attn.o_proj(y.reshape(1, length, 1536))
                        check("public_source_PV", array(y[0, start]), node("y"), 1e-4)
                        check("public_source_output", array(write[0, start]), node("attwrite"), 1e-4)
                        saved_write = torch.zeros_like(full)
                        saved_write[0, start] = torch.tensor(node("attwrite"), device="mps")
                        mid = full + saved_write
                        check("source_mid_identity", array(mid[0, start]), node("mid"))
                        mlp = layer.mlp(layer.post_attention_layernorm(mid))
                        check("public_finite_MLP", array(mlp[0, start]), node("mlpwrite"))
                        check("source_block_identity", array((mid + mlp)[0, start]), states[i, 0])
                        x = before[i].astype(float)
                        zz = x / np.sqrt(np.mean(x ** 2) + eps) * norms[i]
                        rawcheck(key, arm, i, "RMS", zz, node("norm"))
                        rawcheck(key, arm, i, "V", zz @ vweights[i].T + vbias[i], node("v").reshape(-1))
                        kk = data["F_full_k"][i].astype(float).copy()
                        kk[:, start] = node("k").astype(float)
                        q0 = node("q").astype(float)
                        logits = np.einsum('hd,hnd->hn', q0, np.repeat(kk, 6, axis=0)) * float(node("scaling")) + node("mask").astype(float)
                        pp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        pp /= pp.sum(axis=-1, keepdims=True)
                        rawcheck(key, arm, i, "softmax", pp, node("p"))
                        vv = data["F_full_v"][i].astype(float).copy()
                        vv[:, start] = expected_v.astype(float)
                        yy = np.einsum('hn,hnd->hd', expected_p.astype(float), np.repeat(vv, 6, axis=0))
                        rawcheck(key, arm, i, "PV", yy, node("y"))
                        rawcheck(key, arm, i, "output", node("y").astype(float).reshape(-1) @ oweights[i].T, node("attwrite"))
                        if arm == "N":
                            for corner in ("W", "V", "B"):
                                cp = data["F_p"][i] if corner in ("W", "B") else node("p")
                                cv = data["F_v"][i] if corner in ("V", "B") else node("v")
                                prob[0, :, start] = torch.tensor(cp, device="mps")
                                vfull[0, :, start] = torch.tensor(cv, device="mps")
                                cy = (prob @ qwen.repeat_kv(vfull, 6)).transpose(1, 2).contiguous()
                                cw = layer.self_attn.o_proj(cy.reshape(1, length, 1536))
                                check("public_corner_PV", array(cy[0, start]), node("corner_" + corner + "_y"), 1e-4)
                                check("public_corner_output", array(cw[0, start]), node("corner_" + corner + "_write"), 1e-4)
                            check("native_corner_B_PV", node("corner_B_y"), data["F_y"][i])
                            check("native_corner_B_output", node("corner_B_write"), data["F_attwrite"][i])
                for arm in ("U", "W", "V", "B", "Z", "FZ"):
                    candidate = (old["F_V"].astype(float) + data["delta_" + arm]).astype(np.float32)
                    check("actual_value_install", data["value_" + arm + "_installed_V"], candidate)
                    target = old["F_states"][:, 0] if arm == "FZ" else old["G_source"]
                    check("value_path_source_native", data["value_" + arm + "_states"][:, 0], target)
                for arm in (*SOURCE_ARMS, *("value_" + a for a in VALUE_ARMS)):
                    full = torch.zeros((1, length, 1536), device="mps")
                    full[0, start:] = torch.tensor(data[arm + "_states"][-1], device="mps")
                    public = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                    check("actual_finite_head", public, data[arm + "_scores"])
                    x = data[arm + "_states"][-1, -1].astype(float)
                    cpu = (x / np.sqrt(np.mean(x ** 2) + eps) * final_gamma) @ final_w.T
                    check("CPU64_head_bridge", cpu, data[arm + "_scores"], .005)
                    heads.append(dict(key=key, arm=arm, public_error=maximum(public, data[arm + "_scores"]), cpu_error=maximum(cpu, data[arm + "_scores"])))
                native_target = data["N_states"][-1].astype(float) - data["F_states"][-1].astype(float)
                effect_target = data["value_U_states"][-1].astype(float) - data["value_Z_states"][-1].astype(float)
                mt, mz = margin(data["N_scores"]), margin(data["value_Z_scores"])
                for prefix in ("N", "W", "V", "B", "value_U", "value_W", "value_V", "value_B", "value_Z"):
                    row = dict(key=key, **case["group"], arm=prefix)
                    good = True
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = vector((data[prefix + "_states"][-1].astype(float) - data["F_states"][-1].astype(float))[pick], native_target[pick])
                        row.update({scope + "_" + k: v for k, v in m.items()})
                        good &= m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99
                    row.update(margin_error=abs(margin(data[prefix + "_scores"]) - mt))
                    row["passes"] = bool(good and row["margin_error"] <= .1)
                    native.append(row)
                for arm in ("U", "W", "V", "B", "Z"):
                    row = dict(key=key, **case["group"], arm=arm)
                    good = True
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = vector((data["value_" + arm + "_states"][-1].astype(float) - data["value_Z_states"][-1].astype(float))[pick], effect_target[pick])
                        row.update({scope + "_" + k: v for k, v in m.items()})
                        good &= m["norm"] >= .001 and m["relative"] <= .25 and m["cosine"] >= .95
                    row.update(target_margin=mt - mz, margin_error=abs(margin(data["value_" + arm + "_scores"]) - mt), margin_bound=.005 + .25 * abs(mt - mz))
                    row["passes"] = bool(good and abs(mt - mz) >= .02 and row["margin_error"] <= row["margin_bound"])
                    effects.append(row)
                for arm in ("N", "W", "V", "B"):
                    check("emission_identity", data["delta_" + arm], data[arm + "_v"].astype(float) - data["F_v"].astype(float))
                    m = vector(data["delta_" + arm], data["delta_N"])
                    good = 0
                    for i in range(28):
                        v = vector(data["delta_" + arm][i], data["delta_N"][i])
                        ok = v["norm"] >= .001 and v["relative"] <= .1 and v["cosine"] >= .995
                        layers.append(dict(key=key, **case["group"], arm=arm, layer=i + 1, **v, passes=bool(ok)))
                        good += ok
                    emission.append(dict(key=key, **case["group"], arm=arm, **m, good_layers=int(good),
                        passes=bool(m["norm"] >= .001 and m["relative"] <= .1 and m["cosine"] >= .995 and good >= 24)))
                before = data["N_before"].astype(float) - data["F_before"].astype(float)
                h = data["N_states"][:, 0].astype(float) - data["F_states"][:, 0].astype(float)
                a = data["N_attwrite"].astype(float) - data["F_attwrite"].astype(float)
                m = data["N_mlpwrite"].astype(float) - data["F_mlpwrite"].astype(float)
                rounding = h - before - a - m
                parts[key] = dict(h=h, attention=a, mlp=m, rounding=rounding, initial=before[0])
                for i in range(28):
                    for name, value in (("carry", before[i]), ("attention", a[i]), ("MLP", m[i]), ("rounding", rounding[i])):
                        ledger.append(dict(key=key, **case["group"], layer=i + 1, term=name,
                            target_norm=float(np.linalg.norm(h[i])), term_norm=float(np.linalg.norm(value)), signed_projection=projection(value, h[i])))
                    n = data["N_attwrite"][i].astype(float)
                    b = data["N_corner_B_write"][i].astype(float)
                    w = data["N_corner_W_write"][i].astype(float)
                    v = data["N_corner_V_write"][i].astype(float)
                    for name, value in (("reweight", v - b), ("self_value", w - b), ("mixed", n - v - w + b)):
                        local.append(dict(key=key, **case["group"], layer=i + 1, term=name,
                            target_norm=float(np.linalg.norm(n - b)), term_norm=float(np.linalg.norm(value)), signed_projection=projection(value, n - b)))
            print("275 audit", len(heads) // 13, "/32", flush=True)
    combined = []
    for e in emission:
        name = "U" if e["arm"] == "N" else e["arm"]
        f = next(r for r in effects if r["key"] == e["key"] and r["arm"] == name)
        combined.append(dict(key=e["key"], schema=e["schema"], reverse=e["reverse"], arm=e["arm"],
            emitter_pass=e["passes"], effect_pass=f["passes"], passes=e["passes"] and f["passes"]))
    pair_ledger = []
    partners = {a["key"]: b["key"] for pair in paired(design["cases"]) for a, b in (pair, pair[::-1])}
    for key, donor in partners.items():
        a, b = parts[key], parts[donor]
        target = a["h"] - b["h"]
        cumulative = {n: np.cumsum(a[n] - b[n], axis=0) for n in ("attention", "mlp", "rounding")}
        initial = a["initial"] - b["initial"]
        check("paired_cumulative_identity", sum(cumulative.values()) + initial, target, 1e-10)
        for n, values in cumulative.items():
            for i, value in enumerate(values):
                pair_ledger.append(dict(key=key, donor=donor, **cases[key]["group"], layer=i + 1, term=n,
                    target_norm=float(np.linalg.norm(target[i])), term_norm=float(np.linalg.norm(value)), signed_projection=projection(value, target[i])))
    tables = dict(native=native, effects=effects, emission=emission, emission_layers=layers,
        raw_primitives=raw, heads=heads, combined=combined, source_ledger=ledger,
        local_attention=local, paired_source_ledger=pair_ledger)
    for name, records in tables.items():
        table(run / (name.replace("_", "-") + ".csv"), records)
    cc = {a: sum(r["passes"] for r in combined if r["arm"] == a) for a in ("N", "W", "V", "B")}
    nc = {a: sum(r["passes"] for r in native if r["arm"] == a) for a in ("N", "W", "V", "B", "value_U", "value_W", "value_V", "value_B", "value_Z")}
    ec = {a: sum(r["passes"] for r in effects if r["arm"] == a) for a in ("U", "W", "V", "B", "Z")}
    strata = {f"{s}/{rev}": sum(r["passes"] for r in combined if r["arm"] == "V" and r["schema"] == s and r["reverse"] == rev)
              for s in ("generic", "descriptive") for rev in (0, 1)}
    primary = cc["V"] >= 24 and min(strata.values()) >= 6 and cc["W"] <= 8
    save_json(run / "audit.json", dict(status="passed", experiment=275, retrospective=True,
        design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"), source_sha256=sha(__file__),
        numerical_checks=len(checks), numerical_max_error=max(r["error"] for r in checks),
        raw_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        native_counts=nc, effect_counts=ec, combined_counts=cc, V_strata=strata, primary_passes=primary,
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Old contexts, expensive full finite interventions and conditional value-only continuation. Probability is not query alone. Algebraic ledgers are retrospective; no fresh/compact predictor or learned circuit. Old numerical failures unchanged."))
    print(dict(native=nc, effects=ec, combined=cc, primary=primary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-273", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.reference_273.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / "failure-audit.json").exists():
            save_json(args.run / "failure-audit.json", dict(error=repr(exc), time=stamp()))
        raise
