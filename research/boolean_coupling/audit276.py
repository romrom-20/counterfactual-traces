"""Separate smooth-derivative, public primitive and causal-decision audit for276."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head, weights
from .core225b import array, load_model
from .run272 import stamp

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "U", "C", "R", "A", "I", "L", "S", "G0", "G1", "G2", "Z", "FZ", "T")
EVAL = ("U", "C", "R", "A", "I", "L", "S", "G0", "G1", "G2", "Z")


def maximum(a, b):
    return float(np.max(abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def primal_name(arm):
    if arm in ("F", "FZ", "Z"):
        return "F"
    if arm in ("U", "T", "I"):
        return "T"
    return "R" if arm == "R" else "C"


def main(run, reference):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 276 and design["retrospective"] and status["status"] == "complete"
    assert design["cases"] == read(reference / "design.json")["cases"] and tuple(design["arms"]) == ARMS
    assert design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    for name, h in design["reference_hashes"].items():
        assert sha(reference / name) == h
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config = read(folder / "config.json")
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, config, 0, "mlp")
    tw = {name: torch.tensor(value, dtype=torch.float64) for name, value in w.items()}
    head = read_head(folder)
    model = load_model(True)
    layer = model.model.layers[0]
    original = {e["key"]: e for e in read(reference / "status.json")["cases"]}
    cases = {c["key"]: c for c in design["cases"]}
    checks, raw, derivative, native, gains, local_gain, heads = ([] for _ in range(7))
    final_bank = {}

    def check(name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def smooth(x):
        z = x * torch.rsqrt(x.square().mean() + tw["eps"]) * tw["norm"]
        gate = F.linear(z, tw["gate_weight"])
        up = F.linear(z, tw["up_weight"])
        return F.linear(F.silu(gate) * up, tw["down_weight"])

    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        start, length = case["position"], len(case["false_ids"])
        path, old_path = run / "cases" / (key + ".npz"), reference / "cases" / (key + ".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert sha(old_path) == original[key]["arrays_sha256"] == meta["input_sha256"]
        assert all(meta[n] for n in ("native_F_T_U_exact", "prefix_all_arms_exact", "site_only_install", "carrier_primal_for_S_L", "self_and_zero_exact"))
        assert len(meta["full_prefix_hashes"]) == 28
        with np.load(path) as data, np.load(old_path) as old:
            x, true = old["F_mid"][0].astype(float), old["N_mid"][0].astype(float)
            embedding = old["N_before"][0].astype(float) - old["F_before"][0].astype(float)
            self_v = old["N_corner_W_write"][0].astype(float) - old["N_corner_B_write"][0].astype(float)
            xc = (x + embedding + self_v).astype(np.float32)
            c, r = xc.astype(float) - x, true - xc.astype(float)
            points = dict(F=x.astype(np.float32), C=xc, R=(x + r).astype(np.float32), T=true.astype(np.float32))
            for name, value in points.items():
                check("point_provenance_" + name, data["point_" + name], value)
            for name, value in dict(embedding=embedding, self_value=self_v, carrier=c, read=r).items():
                check("component_provenance_" + name, data["component_" + name], value)
            check("absolute_native_input", points["C"].astype(float) + r, true)
            for name, point in points.items():
                with torch.no_grad():
                    full = torch.zeros((1, length, 1536), device="mps")
                    full[0, start] = torch.tensor(point, device="mps")
                    z = layer.post_attention_layernorm(full)
                    gate, up = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
                    phi = layer.mlp.act_fn(gate)
                    product = phi * up
                    write = layer.mlp.down_proj(product)
                public = dict(x=full, z=z, gate=gate, up=up, phi=phi, product=product, write=write)
                for node, value in public.items():
                    check("public_first_MLP_" + node, array(value[0, start]), data["node_" + name + "_" + node])
                # Independent CPU64 primal arithmetic, separate from the forecast's tangent.
                xt = torch.tensor(point.astype(float), dtype=torch.float64)
                zz = xt * torch.rsqrt(xt.square().mean() + tw["eps"]) * tw["norm"]
                gg, uu = F.linear(zz, tw["gate_weight"]), F.linear(zz, tw["up_weight"])
                ph = F.silu(gg)
                pp = ph * uu
                ww = F.linear(pp, tw["down_weight"])
                for node, value in dict(z=zz, gate=gg, up=uu, phi=ph, product=pp, write=ww).items():
                    error = maximum(value.numpy(), data["node_" + name + "_" + node])
                    raw.append(dict(key=key, point=name, node=node, error=error, original_bound=.0001, passes=error <= .0001))
                check("CPU64_write_bridge", ww.numpy(), data["node_" + name + "_write"], 1e-4)
            check("archived_F_MLP", data["node_F_write"], old["F_mlpwrite"][0])
            check("archived_T_MLP", data["node_T_write"], old["N_mlpwrite"][0])
            # JVP, two central finite differences and standard first-reverse tests.
            for anchor in ("F", "C"):
                xt = torch.tensor(points[anchor].astype(float), dtype=torch.float64)
                rt = torch.tensor(r, dtype=torch.float64)
                primal, jvp = torch.func.jvp(smooth, (xt,), (rt,))
                saved = data["derivative_" + anchor + "_write"]
                errors = dict(jvp_error=maximum(jvp.numpy(), saved))
                for eps in (1e-4, 1e-5):
                    fd = (smooth(xt + eps * rt) - smooth(xt - eps * rt)) / (2 * eps)
                    errors["fd_" + str(eps)] = maximum(fd.numpy(), saved)
                for number, function in enumerate((torch.sin, torch.cos)):
                    cotangent = function(torch.arange(1536, dtype=torch.float64) * (17 if number == 0 else 29))
                    variable = xt.clone().requires_grad_(True)
                    reverse = torch.autograd.grad((smooth(variable) * cotangent).sum(), variable)[0] @ rt
                    errors["reverse_" + str(number)] = abs(float(reverse) - float(cotangent @ torch.tensor(saved)))
                for name, error in errors.items():
                    check("CPU64_derivative_" + name, error, 0., 1e-4)
                derivative.append(dict(key=key, anchor=anchor, bound=.0001, **errors))
                check("derivative_point", data["derivative_" + anchor + "_x"], points[anchor])
                check("derivative_direction", data["derivative_" + anchor + "_direction"], r)
                check("finite_derivative_primal", primal.detach().numpy(), data["node_" + anchor + "_write"], 1e-4)
            mf, mc, mq, mt = (data["node_" + n + "_write"].astype(float) for n in ("F", "C", "R", "T"))
            low = mc + data["derivative_F_write"]
            shifted = mc + data["derivative_C_write"]
            gain = shifted - low
            check("gain_identity", data["gain"], gain)
            expected = dict(U=mt, C=mc, R=mq, A=mc + mq - mf, I=mt - mc - mq + 2 * mf,
                            L=low, S=shifted, Z=mf, FZ=mf)
            for number in range(3):
                arm = "G" + str(number)
                seed = int(hashlib.sha256((key + f":276gain:{number}").encode()).hexdigest()[:16], 16)
                draw = np.random.default_rng(seed).normal(size=1536)
                draw *= np.linalg.norm(gain) / np.linalg.norm(draw)
                check("random_direction", draw, data["random_" + arm])
                check("random_norm", np.linalg.norm(draw), np.linalg.norm(gain), 1e-12)
                expected[arm] = low + draw
            for arm, value in expected.items():
                check("candidate_identity_" + arm, value, data["proposal_" + arm])
            for arm in ARMS:
                source, final, scores = data[arm + "_source"], data[arm + "_final"], data[arm + "_scores"]
                family = primal_name(arm)
                check("actual_point_" + arm, data[arm + "_used_point"], points[family])
                check("actual_primal_" + arm, data[arm + "_raw_write"], data["node_" + family + "_write"])
                target_mid = points["F" if arm in ("F", "FZ") else "T"]
                check("native_mid_" + arm, data[arm + "_native_mid"], target_mid)
                candidate = expected.get(arm, mf if arm == "F" else mt).astype(np.float32)
                check("actual_install_" + arm, data[arm + "_installed_write"], candidate)
                check("source_first_addition", source[0], (target_mid + candidate).astype(np.float32))
                with torch.no_grad():
                    full = torch.zeros((1, length, 1536), device="mps")
                    full[0, start:] = torch.tensor(final, device="mps")
                    public_score = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check("finite_full_vocab_head", public_score, scores)
                h = final[-1].astype(float)
                cpu = h / np.sqrt(np.mean(h * h) + head["final_eps"]) * head["final_norm"] @ head["final_head"].T
                check("CPU64_head_bridge", cpu, scores, .005)
                heads.append(dict(key=key, arm=arm, public_error=maximum(public_score, scores), cpu_error=maximum(cpu, scores)))
            for a, b in (("FZ", "F"), ("U", "T")):
                for name in ("source", "final", "scores"):
                    check("self_zero_" + name, data[a + "_" + name], data[b + "_" + name])
            for a, b in (("F", "F"), ("T", "N"), ("U", "N")):
                check("archived_source", data[a + "_source"], old[b + "_states"][:, 0])
                check("archived_final", data[a + "_final"], old[b + "_states"][-1])
                check("archived_scores", data[a + "_scores"], old[b + "_scores"])
            final_bank[key] = {arm: dict(source=data[arm + "_source"][0].astype(float),
                final=data[arm + "_final"].astype(float), margin=margin(data[arm + "_scores"])) for arm in ARMS}
            target = data["T_final"].astype(float) - data["F_final"].astype(float)
            gain_target = data["U_final"].astype(float) - data["L_final"].astype(float)
            mu, ml = margin(data["U_scores"]), margin(data["L_scores"])
            for arm in EVAL:
                row = dict(key=key, **case["group"], arm=arm)
                nr, gr = dict(row), dict(row)
                ngood, ggood = [], []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    n = vector((data[arm + "_final"].astype(float) - data["F_final"].astype(float))[pick], target[pick])
                    g = vector((data[arm + "_final"].astype(float) - data["L_final"].astype(float))[pick], gain_target[pick])
                    nr.update({scope + "_" + k: v for k, v in n.items()})
                    gr.update({scope + "_" + k: v for k, v in g.items()})
                    ngood.append(n["norm"] >= .001 and n["relative"] <= .05 and n["cosine"] >= .99)
                    ggood.append(g["norm"] >= .001 and g["relative"] <= .25 and g["cosine"] >= .95)
                error = abs(margin(data[arm + "_scores"]) - mu)
                nr.update(margin_error=error, passes=bool(all(ngood) and error <= .1))
                gr.update(margin_error=error, target_margin=mu - ml, margin_bound=.005 + .25 * abs(mu - ml),
                    passes=bool(all(ggood) and abs(mu - ml) >= .02 and error <= .005 + .25 * abs(mu - ml)))
                native.append(nr)
                gains.append(gr)
                v = vector(expected[arm] - low, mt - low)
                local_gain.append(dict(row, **v, passes=bool(v["norm"] >= .001 and v["relative"] <= .25 and v["cosine"] >= .95)))
        print("276 audit", len(heads) // len(ARMS), "/32", flush=True)
    pairs = []
    for key, own in final_bank.items():
        donor = final_bank[design["partners"][key]]
        target_source = (own["T"]["source"] - own["F"]["source"]) - (donor["T"]["source"] - donor["F"]["source"])
        target_verdict = (own["T"]["final"][-1] - own["F"]["final"][-1]) - (donor["T"]["final"][-1] - donor["F"]["final"][-1])
        target_margin = (own["T"]["margin"] - own["F"]["margin"]) - (donor["T"]["margin"] - donor["F"]["margin"])
        for arm in EVAL:
            s = vector((own[arm]["source"] - own["F"]["source"]) - (donor[arm]["source"] - donor["F"]["source"]), target_source)
            v = vector((own[arm]["final"][-1] - own["F"]["final"][-1]) - (donor[arm]["final"][-1] - donor["F"]["final"][-1]), target_verdict)
            m = (own[arm]["margin"] - own["F"]["margin"]) - (donor[arm]["margin"] - donor["F"]["margin"])
            good = all(a["norm"] >= .001 and a["relative"] <= .25 and a["cosine"] >= .95 for a in (s, v))
            pairs.append(dict(key=key, donor=design["partners"][key], **cases[key]["group"], arm=arm,
                **{"source_" + k: x for k, x in s.items()}, **{"verdict_" + k: x for k, x in v.items()},
                target_margin=target_margin, margin_error=abs(m - target_margin), margin_bound=.005 + .25 * abs(target_margin),
                passes=bool(good and abs(target_margin) >= .02 and abs(m - target_margin) <= .005 + .25 * abs(target_margin))))
    combined, explanation = [], []
    for key, case in cases.items():
        for arm in EVAL:
            n = next(r for r in native if r["key"] == key and r["arm"] == arm)
            p = next(r for r in pairs if r["key"] == key and r["arm"] == arm)
            combined.append(dict(key=key, schema=case["group"]["schema"], reverse=case["group"]["reverse"], arm=arm,
                native_pass=n["passes"], paired_pass=p["passes"], passes=n["passes"] and p["passes"]))
        g = next(r for r in gains if r["key"] == key and r["arm"] == "S")
        local = next(r for r in local_gain if r["key"] == key and r["arm"] == "S")
        random_fails = sum(not r["passes"] for r in gains if r["key"] == key and r["arm"] in ("G0", "G1", "G2"))
        explanation.append(dict(key=key, schema=case["group"]["schema"], reverse=case["group"]["reverse"],
            local_pass=local["passes"], downstream_pass=g["passes"], random_failures=random_fails,
            passes=local["passes"] and g["passes"] and random_fails >= 2))
    for name, records in dict(native=native, paired=pairs, gain=gains, local_gain=local_gain,
        combined=combined, explanation=explanation, derivatives=derivative, raw_primitives=raw, heads=heads).items():
        table(run / (name.replace("_", "-") + ".csv"), records)
    nc = {a: sum(r["passes"] for r in native if r["arm"] == a) for a in EVAL}
    cc = {a: sum(r["passes"] for r in combined if r["arm"] == a) for a in EVAL}
    def strata(records, arm=None):
        return {f"{s}/{rev}": sum(r["passes"] for r in records if r["schema"] == s and r["reverse"] == rev and (arm is None or r["arm"] == arm))
                for s in ("generic", "descriptive") for rev in (0, 1)}
    sstrata, estrata = strata(combined, "S"), strata(explanation)
    ec = sum(r["passes"] for r in explanation)
    save_json(run / "audit.json", dict(status="passed", experiment=276, retrospective=True,
        design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"), source_sha256=sha(__file__),
        numerical_checks=len(checks), numerical_max_error=max(r["error"] for r in checks),
        raw_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        derivative_anchors=len(derivative), native_counts=nc, combined_counts=cc, S_strata=sstrata,
        explanation_count=ec, explanation_strata=estrata,
        primary_passes=cc["S"] >= 24 and min(sstrata.values()) >= 6,
        gain_explanation_passes=ec >= 24 and min(estrata.values()) >= 6,
        generative_account_passes=cc["S"] >= 24 and min(sstrata.values()) >= 6 and ec >= 24 and min(estrata.values()) >= 6,
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Old single-site diagnostic; smooth CPU64 first-MLP tangents independently checked, not Float32/whole-block derivative certification. Other model computation finite. No fresh/compact full predictor or learned circuit."))
    print(dict(native=nc, combined=cc, primary=cc["S"] >= 24 and min(sstrata.values()) >= 6,
               gain_explanation=ec), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-275", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.reference_275.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / "failure-audit.json").exists():
            save_json(args.run / "failure-audit.json", dict(error=repr(exc), time=stamp()))
        raise
