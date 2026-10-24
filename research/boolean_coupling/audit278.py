"""Separate primitive qualification, signed factorial ledger and native-effect audit278."""

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
BITS = ("000", "001", "010", "011", "100", "101", "110", "111")
ARMS = ("F", "Z", "FZ", *(p + b for p in ("N", "C") for b in BITS), "G0", "G1", "G2", "T")
EVAL = (*(p + b for p in ("N", "C") for b in BITS), "G0", "G1", "G2", "Z", "T")


def maximum(a, b):
    return float(np.max(abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def projection(a, b):
    return float(np.vdot(np.ravel(a), np.ravel(b))) / max(float(np.vdot(np.ravel(b), np.ravel(b))), 1e-30)


def primitive_checks(anchor, nodes, gamma, eps):
    """Independent Torch Double JVP/FD/reverse, with explicit NumPy derivatives."""
    x0, g0, p0, u0 = (anchor[n].astype(float) for n in ("x", "gate", "phi", "up"))
    jobs = []
    sigma = np.sqrt(np.mean(x0 * x0) + eps)
    sig = 1 / (1 + np.exp(-g0))
    slope = sig * (1 + g0 * (1 - sig))
    for prefix in ("N", "C"):
        dx = nodes[prefix + "111"]["x"].astype(float) - x0
        analytic = gamma * (dx / sigma - x0 * np.mean(x0 * dx) / sigma**3)
        def rms(x):
            return x * torch.rsqrt(x.square().mean() + eps) * torch.tensor(gamma)
        jobs.append((prefix + "_RMS", rms, (x0,), (dx,), analytic))
        for n in ("0", "1"):
            dg = nodes[prefix + n + "11"]["gate"].astype(float) - g0
            jobs.append((prefix + n + "_SiLU", F.silu, (g0,), (dg,), slope * dg))
            for s in ("0", "1"):
                node = nodes[prefix + n + s + "1"]
                dphi, dup = node["phi"].astype(float) - p0, node["up"].astype(float) - u0
                def multiply(p, u):
                    return p * u
                jobs.append((prefix + n + s + "_product", multiply, (p0, u0), (dphi, dup), u0 * dphi + p0 * dup))
    records = []
    for name, function, parents, directions, analytic in jobs:
        tx, td = (tuple(torch.tensor(v, dtype=torch.float64) for v in seq) for seq in (parents, directions))
        primal, jvp = torch.func.jvp(function, tx, td)
        errors = dict(jvp_error=maximum(jvp.detach().numpy(), analytic))
        for step in (1e-4, 1e-5):
            fd = (function(*(p + step * d for p, d in zip(tx, td, strict=True)))
                  - function(*(p - step * d for p, d in zip(tx, td, strict=True)))) / (2 * step)
            errors["fd_" + str(step)] = maximum(fd.detach().numpy(), analytic)
        with torch.enable_grad():
            for number, trig in enumerate((torch.sin, torch.cos)):
                cot = trig(torch.arange(analytic.size, dtype=torch.float64) * (17 if number == 0 else 29))
                variables = tuple(p.clone().requires_grad_(True) for p in tx)
                grads = torch.autograd.grad((function(*variables) * cot).sum(), variables)
                reverse = sum(float((g * d).sum()) for g, d in zip(grads, td, strict=True))
                errors["reverse_" + str(number)] = abs(reverse - float(np.vdot(cot.numpy(), analytic)))
        assert all(np.isfinite(e) and e <= 1e-4 for e in errors.values()), (name, errors)
        if name.endswith("RMS"):
            expected = anchor["z"]
        elif name.endswith("SiLU"):
            expected = anchor["phi"]
        else:
            expected = anchor["product"]
        primal_error = maximum(primal.detach().numpy(), expected)
        assert primal_error <= 1e-4, (name, primal_error)
        records.append(dict(primitive=name, bound=.0001, primal_error=primal_error, **errors))
    assert len(records) == 14
    return records


def main(run, reference):
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 278 and design["retrospective"] and status["status"] == "complete"
    assert design["cases"] == read(reference / "design.json")["cases"] and tuple(design["arms"]) == ARMS
    assert design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    for name, h in design["reference_hashes"].items():
        assert sha(reference / name) == h
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 0, "mlp")
    model = load_model(True)
    layer, head = model.model.layers[0], read_head(folder)
    gamma, eps = array(layer.post_attention_layernorm.weight).astype(float), layer.post_attention_layernorm.variance_epsilon
    original = {r["key"]: r for r in read(reference / "status.json")["cases"]}
    cases = {c["key"]: c for c in design["cases"]}
    checks, raw, derivatives, native, effect, nonlinear, local_rows, ledger, heads = ([] for _ in range(9))
    bank, writes_bank = {}, {}

    def check(name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, node, field, a, b):
        error = maximum(a, b)
        raw.append(dict(key=key, node=node, field=field, error=error, original_bound=.0001, passes=error <= .0001))

    for entry in status["cases"]:
        key, case = entry["key"], cases[entry["key"]]
        start, length = case["position"], len(case["false_ids"])
        path, prior = run / "cases" / (key + ".npz"), reference / "cases" / (key + ".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert sha(prior) == original[key]["arrays_sha256"] == meta["input_sha256"]
        assert all(meta[n] for n in ("ordinary_F_T_unhooked", "native_replay_exact", "native_self_and_zero_full_tails_exact",
            "archived_Z_C111_exact", "all_F_primitive_zero_edits_exact", "only_source_MLP1_primitives_changed"))
        assert len(meta["prefix_hashes"]) == 28
        with np.load(path) as d, np.load(prior) as old:
            anchor = {n: d["anchor_" + n] for n in ("x", "z", "gate", "up", "phi", "product", "write")}
            nodes = {p + b: {n: d["node_" + p + b + "_" + n] for n in
                ("x", "raw_z", "z", "gate", "up", "raw_phi", "phi", "raw_product", "product", "write")}
                for p in ("F", "N", "C") for b in BITS}
            for name in anchor:
                check("anchor_old", anchor[name], old["node_F_" + name])
            points = {p: d["point_" + p] for p in ("F", "N", "C")}
            for p, q in (("F", "F"), ("N", "T"), ("C", "C")):
                check("point_old", points[p], old["point_" + q])
            for row in primitive_checks(anchor, nodes, gamma, eps):
                derivatives.append(dict(key=key, **row))
            with torch.no_grad():
                for name, node in nodes.items():
                    p, bits = name[0], name[1:]
                    n, s, k = map(int, bits)
                    full = torch.zeros((1, length, len(gamma)), device="mps")
                    full[0, start] = torch.tensor(points[p], device="mps")
                    rz = layer.post_attention_layernorm(full)
                    z = rz.clone()
                    x0, delta = anchor["x"].astype(float), points[p].astype(float) - anchor["x"].astype(float)
                    sigma = np.sqrt(np.mean(x0 * x0) + eps)
                    dz = gamma * (delta / sigma - x0 * np.mean(x0 * delta) / sigma**3)
                    zr = array(rz[0, start]) if n else (anchor["z"].astype(float) + dz).astype(np.float32)
                    z[0, start] = torch.tensor(zr, device="mps")
                    g, u = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
                    rphi = layer.mlp.act_fn(g)
                    phi = rphi.clone()
                    sigmoid = 1 / (1 + np.exp(-anchor["gate"].astype(float)))
                    slope = sigmoid * (1 + anchor["gate"].astype(float) * (1 - sigmoid))
                    ph = array(rphi[0, start]) if s else (anchor["phi"].astype(float) + slope * (array(g[0, start]).astype(float) - anchor["gate"].astype(float))).astype(np.float32)
                    phi[0, start] = torch.tensor(ph, device="mps")
                    rproduct = phi * u
                    product = rproduct.clone()
                    pr = array(rproduct[0, start]) if k else (anchor["product"].astype(float)
                        + anchor["up"].astype(float) * (ph.astype(float) - anchor["phi"].astype(float))
                        + anchor["phi"].astype(float) * (array(u[0, start]).astype(float) - anchor["up"].astype(float))).astype(np.float32)
                    product[0, start] = torch.tensor(pr, device="mps")
                    output = layer.mlp.down_proj(product)
                    public = dict(x=full, raw_z=rz, z=z, gate=g, up=u, raw_phi=rphi, phi=phi,
                                  raw_product=rproduct, product=product, write=output)
                    for field, val in public.items():
                        check("public_policy_" + field, array(val[0, start]), node[field])
                    if p == "F":
                        for field in anchor:
                            check("False_zero_primitive", node[field], anchor[field])
                    # Independent whole-policy CPU64 raw comparison, never silently retoleranced.
                    xx = points[p].astype(float)
                    zz = xx / np.sqrt(np.mean(xx * xx) + eps) * gamma if n else anchor["z"].astype(float) + dz
                    gg, uu = (zz @ w[t + "_weight"].T for t in ("gate", "up"))
                    pp = gg / (1 + np.exp(-gg)) if s else anchor["phi"].astype(float) + slope * (gg - anchor["gate"].astype(float))
                    prod = pp * uu if k else anchor["product"].astype(float) + anchor["up"].astype(float) * (pp - anchor["phi"].astype(float)) + anchor["phi"].astype(float) * (uu - anchor["up"].astype(float))
                    ww = prod @ w["down_weight"].T
                    for field, val in dict(z=zz, gate=gg, up=uu, phi=pp, product=prod, write=ww).items():
                        rawcheck(key, name, field, val, node[field])
                    if bits == "111":
                        check("finite_whole_MLP_write_bridge", ww, node["write"], 1e-4)
            # Actual source primitive records versus independently replayed policies.
            rng_outputs = {}
            delta_write = nodes["N111"]["write"].astype(float) - nodes["N000"]["write"].astype(float)
            for number in range(3):
                arm = "G" + str(number)
                seed = int(hashlib.sha256((key + f":278-write:{number}").encode()).hexdigest()[:16], 16)
                draw = np.random.default_rng(seed).normal(size=delta_write.shape)
                draw *= np.linalg.norm(delta_write) / np.linalg.norm(draw)
                check("matched_random", draw, d["random_" + arm])
                check("matched_output_norm", np.linalg.norm(draw), np.linalg.norm(delta_write), 1e-12)
                rng_outputs[arm] = nodes["N000"]["write"].astype(float) + draw
                check("random_proposal", rng_outputs[arm], d["proposal_" + arm])
            local_writes = {}
            for arm in ARMS:
                if arm in ("F", "T"):
                    family = "F111" if arm == "F" else "N111"
                    for field in ("gate", "up", "phi", "product", "write"):
                        check("unhooked_native_operand", d[arm + "_actual_" + field], nodes[family][field])
                    mid = points["F" if arm == "F" else "N"]
                    check("unhooked_native_mid", d[arm + "_actual_mid"], mid)
                    candidate = nodes[family]["write"]
                else:
                    family = "F000" if arm == "FZ" else "F111" if arm == "Z" else "N000" if arm.startswith("G") else arm
                    mid = points["F" if arm == "FZ" else "N"]
                    check("outer_mid_stays_native", d[arm + "_actual_native_mid"], mid)
                    check("actual_MLP_point", d[arm + "_actual_used_point"], nodes[family]["x"])
                    for field in ("raw_z", "z", "gate", "up", "raw_phi", "phi", "raw_product", "product"):
                        check("actual_primitive", d[arm + "_actual_" + field], nodes[family][field])
                    check("actual_raw_write", d[arm + "_actual_raw_write"], nodes[family]["write"])
                    candidate = rng_outputs[arm].astype(np.float32) if arm.startswith("G") else nodes[family]["write"]
                    check("actual_write_install", d[arm + "_actual_installed_write"], candidate)
                check("actual_source_addition", d[arm + "_source"], (mid + candidate).astype(np.float32))
                local_writes[arm] = candidate.astype(float)
                with torch.no_grad():
                    full = torch.zeros((1, length, len(gamma)), device="mps")
                    full[0, start:] = torch.tensor(d[arm + "_final"], device="mps")
                    expected = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check("finite_full_vocab_head", expected, d[arm + "_scores"])
                h = d[arm + "_final"][-1].astype(float)
                cpu = h / np.sqrt(np.mean(h * h) + head["final_eps"]) * head["final_norm"] @ head["final_head"].T
                check("CPU64_head_bridge", cpu, d[arm + "_scores"], .005)
                heads.append(dict(key=key, arm=arm, public_error=maximum(expected, d[arm + "_scores"]), cpu_error=maximum(cpu, d[arm + "_scores"])))
            for a, b in (("N111", "T"), ("FZ", "F")):
                for field in ("source", "final", "scores"):
                    check("self_zero", d[a + "_" + field], d[b + "_" + field])
            for a, b in (("F", "F"), ("T", "T"), ("Z", "Z"), ("C111", "C")):
                check("archived_source", d[a + "_source"], old[b + "_source"][0])
                for field in ("final", "scores"):
                    check("archived_endpoint", d[a + "_" + field], old[b + "_" + field])
            own = {a: dict(source=d[a + "_source"].astype(float), final=d[a + "_final"].astype(float), margin=margin(d[a + "_scores"])) for a in ARMS}
            bank[key], writes_bank[key] = own, local_writes
            target = own["T"]["final"] - own["F"]["final"]
            effect_target = own["T"]["final"] - own["Z"]["final"]
            nonlinear_target = own["T"]["final"] - own["N000"]["final"]
            mt, mz, ml = (own[a]["margin"] for a in ("T", "Z", "N000"))
            for arm in EVAL:
                nr, er, rr = (dict(key=key, **case["group"], arm=arm) for _ in range(3))
                nok, eok, rok = [], [], []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    n = vector((own[arm]["final"] - own["F"]["final"])[pick], target[pick])
                    e = vector((own[arm]["final"] - own["Z"]["final"])[pick], effect_target[pick])
                    r = vector((own[arm]["final"] - own["N000"]["final"])[pick], nonlinear_target[pick])
                    for row, val in ((nr, n), (er, e), (rr, r)):
                        row.update({scope + "_" + k: v for k, v in val.items()})
                    nok.append(n["norm"] >= .001 and n["relative"] <= .05 and n["cosine"] >= .99)
                    eok.append(e["norm"] >= .001 and e["relative"] <= .25 and e["cosine"] >= .95)
                    rok.append(r["norm"] >= .001 and r["relative"] <= .25 and r["cosine"] >= .95)
                error = abs(own[arm]["margin"] - mt)
                nr.update(margin_error=error, passes=bool(all(nok) and error <= .1))
                er.update(margin_error=error, target_margin=mt - mz, margin_bound=.005 + .25 * abs(mt - mz),
                    passes=bool(all(eok) and abs(mt - mz) >= .02 and error <= .005 + .25 * abs(mt - mz)))
                rr.update(margin_error=error, target_margin=mt - ml, margin_bound=.005 + .25 * abs(mt - ml),
                    passes=bool(all(rok) and abs(mt - ml) >= .02 and error <= .005 + .25 * abs(mt - ml)))
                native.append(nr)
                effect.append(er)
                nonlinear.append(rr)
                v = vector(local_writes[arm] - local_writes["F"], local_writes["T"] - local_writes["F"])
                lv = vector(local_writes[arm] - local_writes["N000"], local_writes["T"] - local_writes["N000"])
                local_rows.append(dict(key=key, **case["group"], arm=arm, **v,
                    nonlinear_relative=lv["relative"], nonlinear_cosine=lv["cosine"], nonlinear_norm=lv["norm"],
                    passes=bool(v["norm"] >= .001 and v["relative"] <= .1 and v["cosine"] >= .995),
                    nonlinear_pass=bool(lv["norm"] >= .001 and lv["relative"] <= .25 and lv["cosine"] >= .95)))
            # Exact signed factorial, not a unique circuit or attribution of learned features.
            for p in ("N", "C"):
                for scope in ("write", "receiving", "verdict"):
                    vals = {b: local_writes[p + b] if scope == "write" else own[p + b]["final"][1:] if scope == "receiving" else own[p + b]["final"][-1] for b in BITS}
                    total = vals["111"] - vals["000"]
                    pieces = []
                    for bits in BITS[1:]:
                        selected = [b for b in BITS if all(y <= x for x, y in zip(bits, b, strict=True))]
                        term = sum(((-1) ** (bits.count("1") - b.count("1"))) * vals[b] for b in selected)
                        pieces.append(term)
                        ledger.append(dict(key=key, **case["group"], point=p, scope=scope, term=bits,
                            term_norm=float(np.linalg.norm(term)), target_norm=float(np.linalg.norm(total)),
                            signed_projection=projection(term, total)))
                    check("signed_Mobius_closure", sum(pieces), total, 1e-10)
        print("278 audit", len(bank), "/32", flush=True)
    pairs, combined, restoration = [], [], []
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
            row = dict(key=key, **cases[key]["group"], arm=arm, **{"source_" + k: x for k, x in s.items()},
                **{"verdict_" + k: x for k, x in v.items()}, target_margin=tm, margin_error=abs(m - tm), margin_bound=.005 + .25 * abs(tm),
                passes=bool(good and abs(tm) >= .02 and abs(m - tm) <= .005 + .25 * abs(tm)))
            pairs.append(row)
            n, e, loc = (next(r for r in rows if r["key"] == key and r["arm"] == arm) for rows in (native, effect, local_rows))
            combined.append(dict(key=key, schema=row["schema"], reverse=row["reverse"], arm=arm,
                native_pass=n["passes"], effect_pass=e["passes"], local_pass=loc["passes"], paired_pass=row["passes"],
                passes=bool(n["passes"] and e["passes"] and loc["passes"] and row["passes"])))
            rr = next(r for r in nonlinear if r["key"] == key and r["arm"] == arm)
            random_fails = sum(not r["passes"] for r in nonlinear if r["key"] == key and r["arm"] in ("G0", "G1", "G2"))
            restoration.append(dict(key=key, schema=row["schema"], reverse=row["reverse"], arm=arm,
                local_pass=loc["nonlinear_pass"], downstream_pass=rr["passes"], random_failures=random_fails,
                passes=bool(loc["nonlinear_pass"] and rr["passes"] and random_fails >= 2)))
    for name, rows in dict(native=native, effect=effect, nonlinear=nonlinear, local=local_rows, paired=pairs,
        combined=combined, restoration=restoration, ledger=ledger, derivatives=derivatives, raw=raw, heads=heads).items():
        table(run / (name + ".csv"), rows)
    def counts(rows):
        return {a: sum(r["passes"] for r in rows if r["arm"] == a) for a in EVAL}
    def strata(rows, arm):
        return {f"{s}/{rev}": sum(r["passes"] for r in rows if r["schema"] == s and r["reverse"] == rev and r["arm"] == arm)
                for s in ("generic", "descriptive") for rev in (0, 1)}
    cc, rc = counts(combined), counts(restoration)
    primary = cc["N100"] >= 24 and min(strata(combined, "N100").values()) >= 6
    restoring = rc["N100"] >= 24 and min(strata(restoration, "N100").values()) >= 6
    candidates = [a for a in EVAL if a.startswith("N") and a != "N111" and cc[a] >= 24 and min(strata(combined, a).values()) >= 6]
    save_json(run / "audit.json", dict(status="passed", experiment=278, retrospective=True,
        design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"), source_sha256=sha(__file__),
        numerical_checks=len(checks), numerical_max_error=max(r["error"] for r in checks), primitive_derivative_directions=len(derivatives),
        raw_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001,
            qualification="Exact public policy replays and separate smooth primitive derivatives required; CPU64 intermediate raw failures retained. Only fully finite whole-MLP write bridges required<=1e-4."),
        native_counts=counts(native), effect_counts=counts(effect), local_counts=counts(local_rows),
        paired_counts=counts(pairs), combined_counts=cc, restoration_counts=rc,
        N100_strata=strata(combined, "N100"), primary_passes=primary,
        N100_restoration_strata=strata(restoration, "N100"), normalization_restoration_passes=restoring,
        normalization_mechanistic_account_passes=primary and restoring,
        prespecified_secondary_candidates=candidates,
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        boundary="Old single-source-MLP primitive factorial, native/carrier source branch and finite downstream computation. Signed factorial is context/operating-point dependent, not a learned feature/circuit, fresh forecast, efficiency result or general Q B solution."))
    print(dict(native=counts(native), combined=cc, restoration=rc, primary=primary, candidates=candidates), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-276", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.reference_276.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / "failure-audit.json").exists():
            save_json(args.run / "failure-audit.json", dict(error=repr(exc), time=stamp()))
        raise
