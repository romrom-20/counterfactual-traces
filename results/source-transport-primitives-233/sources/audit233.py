"""Separate Torch CPU64 AD/FD qualification and fixed retrospective scoring for 233."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .audit229 import checked, sha

BITS = tuple(f"{i:04b}" for i in range(16))


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def functions(weights, data, start):
    """Build actual hybrid functions; their derivative is not passed to this function."""
    w = {k:tensor(v) for k, v in weights.items()}
    x, s = tensor(data["x"]), tensor(data["source"])
    width, groups = int(weights["width"]), int(weights["groups"])
    cos, sin = tensor(data["rope_cos"][0]), tensor(data["rope_sin"][0])
    mask = tensor(data["mask"][0])

    def normalized(r):
        return r*torch.rsqrt(r.square().mean(-1, keepdim=True)+w["attn_eps"])*w["attn_norm"]

    def project(z, name, first=0):
        value = F.linear(z, w[name+"_weight"], w.get(name+"_bias"))
        value = value.reshape(len(z), -1, width).transpose(0, 1)
        if name != "v":
            a, b = value.chunk(2, -1)
            value = value*cos[first:]+torch.cat((-b, a), -1)*sin[first:]
        return value.repeat_interleave(groups, 0) if name != "q" else value

    def output(context):
        return F.linear(context.transpose(0, 1).reshape(context.shape[1], -1), w["o_weight"])

    def anchor(r):
        z = normalized(r)
        q, k, v = (project(z, n) for n in ("q", "k", "v"))
        p = torch.softmax(q@k.transpose(-1, -2)*w["scaling"]+mask, -1)
        return dict(q=q, k=k, v=v, p=p, mid=r+output(p@v)+w.get("o_bias", 0))

    old, source = anchor(x), anchor(s)
    def make(bits):
        k, p, v, pv = (source[name] if bit == "1" else old[name]
                        for name, bit in zip(("k", "p", "v", "p"), bits))
        q0 = old["q"][:, start:]
        base_logits = q0@k.transpose(-1, -2)*w["scaling"]
        pb, pe, r0 = p[:, start:], pv[:, start:], x[start:]

        def hybrid(receiving):
            z = normalized(receiving)
            q, kr, vr = (project(z, n, start) for n in ("q", "k", "v"))
            keys = torch.cat((k[:, :start], kr), 1)
            logits_delta = q@keys.transpose(-1, -2)*w["scaling"]-base_logits
            dp = pb*(logits_delta-(pb*logits_delta).sum(-1, keepdim=True))
            dv = torch.cat((torch.zeros_like(old["v"][:, :start]), vr-old["v"][:, start:]), 1)
            return receiving-r0+output(dp@v+pe@dv)

        def native(receiving):
            prefix = s[:start] if bits == "1111" else x[:start]
            return anchor(torch.cat((prefix, receiving), 0))["mid"][start:]

        return hybrid, native

    return make, old, source


def metrics(prediction, target, false):
    delta, actual = prediction-false, target-false
    den = float(np.linalg.norm(actual))
    relative = float(np.linalg.norm(delta-actual)/den)
    cosine = float(np.vdot(delta, actual)/(np.linalg.norm(delta)*den))
    return relative, cosine, bool(den >= .001 and relative <= .025 and cosine >= .995)


def main(run):
    assert not (run/"audit.json").exists() and not (run/"qualification-status.json").exists()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    design = json.loads((run/"design.json").read_text())
    status = json.loads((run/"status.json").read_text())
    assert status["status"] == "complete" and len(status["cases"]) == 32
    previous = Path(design["input_root"])
    for name, h in design["source_hashes"].items():
        assert sha(root/name) == sha(run/"sources"/Path(name).name) == h
    for name, h in design["inputs"].items():
        assert sha(previous/name) == h
    weights = {k:v.astype(float) for k, v in dict(np.load(previous/"weights.npz")).items()}
    (run/"qualification").mkdir()
    checks, rows, components, qualifications = [], [], [], []
    for index, entry in enumerate(status["cases"], 1):
        key = entry["key"]
        path = run/"cases"/(key+".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == entry["arrays_sha256"] == meta["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert all(c["error"] <= c["bound"] for c in meta["checks"])
        data = {k:v.astype(float) for k, v in dict(np.load(previous/"forecasts"/path.name)).items()}
        values = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        start, group = meta["start"], meta["group"]
        anchor = tensor(data["x"][start:])
        direction = tensor((data["receiver"]-data["x"])[start:])
        witnesses = {}
        make, old, source = functions(weights, data, start)
        for bits in BITS:
            hybrid, native = make(bits)
            for name, function in (("hybrid", hybrid), *(([("native", native)]) if bits in ("0000", "1111") else [])):
                primal, tangent = torch.func.jvp(function, (anchor,), (direction,))
                plus, minus = (function(anchor+sign*1e-5*direction).detach().numpy() for sign in (1, -1))
                tangent, primal = tangent.detach().numpy(), primal.detach().numpy()
                checked(checks, name+"_analytic_ad", tangent, values["T_"+bits], 1e-9)
                checked(checks, name+"_ad_fd", tangent, (plus-minus)/2e-5)
                if name == "hybrid":
                    checked(checks, "hybrid_center", primal, np.zeros_like(primal), 1e-9)
                else:
                    label = "F" if bits == "0000" else "S"
                    checked(checks, "native_smooth_primal", primal, data[label+"_primal"], 1e-9)
                for suffix, value in (("tangent", tangent), ("plus", plus), ("minus", minus)):
                    witnesses[f"{bits}_{name}_{suffix}"] = value
            if bits == "0000":
                checked(checks, "torch_false_public", old["mid"][start:].numpy(), data["false_mid"])
                checked(checks, "torch_source_public", source["mid"][start:].numpy(), data["source_mid"])
        witness = run/"qualification"/path.name
        np.savez_compressed(witness, **witnesses)
        qualifications.append(dict(key=key, arrays_sha256=sha(witness)))
        total = values["T_1111"]-values["T_0000"]
        checked(checks, "factorial_closure", sum(values["M_"+b] for b in BITS[1:]), total, 1e-9)
        for scope in ("all", "verdict"):
            select = slice(None) if scope == "all" else -1
            t = total[select]
            denominator = float(np.linalg.norm(t))
            for bits in BITS:
                increment = values["T_"+bits]-values["T_0000"]
                forecast = values["baseline"]+increment
                rel, cos, accuracy = metrics(forecast[select], values["native_mid"][select], values["false_mid"][select])
                random_rel = []
                for control in range(3) if bits != "0000" else []:
                    control_value = values[f"random_{bits}_{control}"]
                    seed = int(hashlib.sha256((key+bits+str(control)+"233").encode()).hexdigest()[:16], 16)
                    expected = np.random.default_rng(seed).standard_normal(increment.shape)
                    expected *= np.linalg.norm(increment, axis=-1, keepdims=True)/np.linalg.norm(expected, axis=-1, keepdims=True)
                    checked(checks, "seeded_control", control_value, expected, 1e-9)
                    checked(checks, "row_matched_control", np.linalg.norm(control_value, axis=-1),
                            np.linalg.norm(increment, axis=-1), 1e-9)
                    r, _, _ = metrics((values["baseline"]+control_value)[select], values["native_mid"][select], values["false_mid"][select])
                    random_rel.append(r)
                baseline, _, _ = metrics(values["baseline"][select], values["native_mid"][select], values["false_mid"][select])
                transport_relative = float(np.linalg.norm(increment[select]-t)/denominator)
                rows.append(dict(key=key, **group, scope=scope, bits=bits, relative=rel, cosine=cos,
                    accuracy=accuracy, controlled_pass=bool(accuracy and rel < baseline and all(rel < r for r in random_rel)),
                    transport_norm=denominator, transport_relative=transport_relative,
                    transport_sufficient=bool(denominator >= .001 and transport_relative <= .25),
                    **{f"random_{i}_relative":r for i, r in enumerate(random_rel)}))
        for bits in BITS[1:]:
            for position in ("all", "first", "verdict"):
                sub = slice(None) if position == "all" else 0 if position == "first" else -1
                term, target = values["M_"+bits][sub], total[sub]
                norm2 = float(np.vdot(target, target))
                components.append(dict(key=key, **group, position=position, bits=bits,
                    norm=float(np.linalg.norm(term)), signed_share=float(np.vdot(term, target)/norm2)))
        assert len(witnesses) == 54
        print("qualified", index, "/32; all operator graphs and original anchors", flush=True)
    for name, records in (("operators", rows), ("components", components)):
        keys = list(dict.fromkeys(k for r in records for k in r))
        with (run/(name+".csv")).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    summaries = {}
    for scope in ("all", "verdict"):
        summaries[scope] = {}
        for bits in BITS:
            subset = [r for r in rows if r["scope"] == scope and r["bits"] == bits]
            counts = {name:sum(r[name] for r in subset) for name in ("accuracy", "controlled_pass", "transport_sufficient")}
            strata = {f"{s}_{o}":{name:sum(r[name] for r in subset if r["schema"] == s and r["reverse"] == o)
                for name in counts} for s in ("generic", "descriptive") for o in (0, 1)}
            summaries[scope][bits] = dict(counts=counts, strata=strata,
                whole={name:counts[name] >= 24 and all(g[name] >= 6 for g in strata.values()) for name in counts},
                medians={name:float(np.median([r[name] for r in subset])) for name in
                    ("relative", "cosine", "transport_relative", "transport_norm")})
    assert len(rows) == 1024 and len(components) == 1440 and len(qualifications) == 32
    assert sum(len(dict(np.load(run/"qualification"/(q["key"]+".npz"))))//3 for q in qualifications) == 576
    audit = dict(status="passed", experiment=233, retrospective=True, ordinary_models=0,
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device="cpu", dtype="float64"),
        checks=checks, summaries=summaries, source_hashes=design["source_hashes"],
        counts=design["expected"], design_sha256=sha(run/"design.json"),
        component_medians={position:{bits:{name:float(np.median([r[name] for r in components if r["position"] == position and r["bits"] == bits]))
            for name in ("norm", "signed_share")} for bits in BITS[1:]} for position in ("all", "first", "verdict")})
    (run/"qualification-status.json").write_text(json.dumps(dict(status="complete", cases=qualifications), indent=2)+"\n")
    (run/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:v for k, v in audit.items() if k not in ("checks", "source_hashes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
