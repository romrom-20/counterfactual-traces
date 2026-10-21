"""Separate local polynomial-query audit and sealed native forecast scoring."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha, softmax
from .audit242b import decision, geometry, head
from .checkpoint238 import locate
from .core225b import write_json
from .curvature255 import jet

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)



def local_input(base, point, anchor, gamma, eps, arm, key, layer, checks):
    d = point-base
    scale = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+eps)
    a = np.mean(base*d, axis=-1, keepdims=True)/scale**2
    b = np.mean(d*d, axis=-1, keepdims=True)/scale**2
    slope = gamma*(d/scale-base*np.mean(base*d, axis=-1, keepdims=True)/scale**3)
    quadratic = gamma/scale*((1.5*a*a-.5*b)*base-a*d)
    cubic = gamma/scale*((1.5*a*b-2.5*a**3)*base+(1.5*a*a-.5*b)*d)
    for label, actual, expected in zip(("linear", "quadratic", "cubic"), (slope, quadratic, cubic), jet(base, d, gamma, eps), strict=True):
        checked(checks, "independent_polynomial_jet_"+label, actual, expected, 1e-10)
    correction = np.zeros_like(slope)
    if arm in ("Q", "C"):
        correction = quadratic if arm == "Q" else quadratic+cubic
    if arm == "R":
        seed = int(hashlib.sha256(f"{key}:{layer}:local-query-quadratic-control".encode()).hexdigest()[:16], 16)
        direction = np.random.default_rng(seed).normal(size=quadratic.shape)
        correction = direction*(np.linalg.norm(quadratic, axis=-1, keepdims=True)/np.linalg.norm(direction, axis=-1, keepdims=True))
        checked(checks, "independent_random_row_norm", np.linalg.norm(correction, axis=-1), np.linalg.norm(quadratic, axis=-1), 1e-10)
    return (anchor+slope+correction).astype(np.float32), correction, slope, quadratic


def witness_checks(run, case, phase, entry, data, weights, query_weights, checks, raw_checks):
    firsts = {}
    for c in read(run/"sources/cases256.json"):
        firsts.setdefault((c["group"]["schema"], c["group"]["reverse"]), c["key"])
    assert len(firsts) == 4
    if case["key"] not in firsts.values():
        assert entry["witness_sha256"] is None
        return 0
    wp = run/("witnesses-forecast" if phase == "forecast" else "witnesses-native")/(case["key"]+".npz")
    assert sha(wp) == entry["witness_sha256"]
    count = 0
    with np.load(wp) as w:
        for arm in (("F", "Z", "L", "Q", "C", "R") if phase == "forecast" else ("T", "O")):
            for layer in (1, 14, 28):
                prefix = f"{arm}_{layer}_"
                q, k, v, raw, mask, installed, output, current_k = (w[prefix+n].astype(float) for n in ("q", "k", "v", "raw", "mask", "installed_p", "y", "current_k"))
                x, finite_z, used_z, linear, cos, sin = (w[prefix+n].astype(float) for n in ("x", "original_z", "installed_z", "q_linear", "cos", "sin"))
                gamma, eps = weights["input_norm"][layer-1], float(weights["input_eps"])
                normalized = x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)*gamma
                checked(checks, "CPU64_native_normalized_input", normalized, finite_z, 1e-4)
                expected_z = finite_z.copy()
                if arm in ("Z", "L", "Q", "C", "R"):
                    base, anchor = data[f"F_r{layer}"].astype(float), data[f"F_z{layer}"].astype(float)
                    direction = x-base
                    scale = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+eps)
                    derivative = gamma*(direction/scale-base*np.mean(base*direction, axis=-1, keepdims=True)/scale**3)
                    expected_z, correction, _, quadratic = local_input(base, x, anchor, gamma, eps, arm, case["key"], layer, [])
                    checked(checks, "actual_local_curvature_used", w[prefix+"correction"].astype(float), correction, 0.)
                    def smooth(value):
                        return value/np.sqrt(np.mean(value*value, axis=-1, keepdims=True)+eps)*gamma
                    for step in (1e-3, 1e-4):
                        fd = (smooth(base+step*direction)-smooth(base-step*direction))/(2*step)
                        bound = 1e-5 if step == 1e-4 else 1e-3
                        checked(checks, f"CPU64_smooth_RMS_tangent_FD_{step}", fd, derivative, bound)
                    step = 1e-3
                    fd2 = (smooth(base+step*direction)-2*smooth(base)+smooth(base-step*direction))/(2*step**2)
                    checked(checks, "CPU64_smooth_RMS_quadratic_FD", fd2, quadratic, 1e-5)
                checked(checks, "independent_query_input_installation", used_z, expected_z, 0.)
                qw = query_weights[layer]
                projected = expected_z@qw["weight"].T+qw["bias"]
                head_width, groups, scaling = q.shape[-1], int(w[prefix+"groups"]), float(w[prefix+"scaling"])
                unrotated = projected.reshape(len(projected), -1, head_width).transpose(1, 0, 2)
                half = head_width//2
                rotated = unrotated*cos[None]+np.concatenate((-unrotated[..., half:], unrotated[..., :half]), axis=-1)*sin[None]
                raw_cpu = q@np.repeat(k, groups, axis=0).swapaxes(-1, -2)*scaling
                for component, actual, expected in (("query_projection", linear, projected), ("rotary_query", q, rotated), ("scaled_QK", raw, raw_cpu)):
                    error = float(np.max(abs(actual-expected)))
                    assert np.isfinite(error)
                    raw_checks.append(dict(key=case["key"], arm=arm, block=layer, component=component, error=error, original_bound=1e-4, passes=error <= 1e-4, maximum_magnitude=float(np.max(abs(actual)))))
                probability = softmax(raw_cpu+mask)
                checked(checks, "CPU64_composed_probability", probability, installed, 1e-4)
                checked(checks, "CPU64_composed_output", probability@np.repeat(v, groups, axis=0), output, 1e-4)
                independent_raw = rotated@np.repeat(k, groups, axis=0).swapaxes(-1, -2)*scaling
                independent_p = softmax(independent_raw+mask)
                checked(checks, "CPU64_full_query_path_probability", independent_p, installed, 1e-4)
                checked(checks, "CPU64_full_query_path_output", independent_p@np.repeat(v, groups, axis=0), output, 1e-4)
                checked(checks, "CPU64_runtime_softmax", softmax(raw+mask), installed, 1e-4)
                checked(checks, "CPU64_runtime_PV", installed@np.repeat(v, groups, axis=0), output, 1e-4)
                expected_k = current_k.copy()
                checked(checks, "independent_key_installation", k, expected_k, 0.)
                count += 1
    return count


def main(run, phase):
    destination = run/("pre-audit.json" if phase == "pre" else "audit.json")
    assert not destination.exists()
    design, status = read(run/"design.json"), read(run/"forecast-status.json")
    assert design["experiment"] == 256 and design["primary"] == "C" and status["status"] == "complete" and status["ordinary_true"] == 0
    assert len(status["cases"]) == 32 and status["counts"] == dict(full_forwards=192, CPU64_tangent_maps=4480, CPU64_quadratic_maps=2688, CPU64_cubic_maps=896, extra_full_QK_softmax_PV_each=896, witness_extra_QK=72)
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(run/"sources"/Path(p).name) == h
    assert sha(run/"weights.npz") == design["weights_sha256"]
    weights = dict(np.load(run/"weights.npz"))
    checkpoint, identity = locate(ROOT)
    assert identity == design["checkpoint"]
    query_weights = {}
    with safe_open(checkpoint/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in (1, 14, 28):
            prefix = f"model.layers.{layer-1}.self_attn.q_proj."
            query_weights[layer] = {n:handle.get_tensor(prefix+n).to(torch.float32).numpy().astype(float) for n in ("weight", "bias")}
        for i in range(28):
            norm = handle.get_tensor(f"model.layers.{i}.input_layernorm.weight").float().numpy().astype(float)
            checked([], "checkpoint_norm_weight", norm, weights["input_norm"][i], 0.)
    catalog = set(read(run/"sources/historical-input-hashes-256.json")["hashes"])
    seen, strata, checks, rows, layers, interactions = set(), Counter(), [], [], [], []
    witness_count, raw_checks = 0, []
    native = None
    if phase == "pre":
        assert not list((run/"observations").glob("*")) and not (run/"status.json").exists()
    else:
        native, seal, proof, final = (read(run/n) for n in ("status.json", "forecast-seal.json", "measurement-proof.json", "seal.json"))
        assert native["status"] == "complete" and native["ordinary_true"] == 32 and len(native["cases"]) == 32
        assert seal["status"] == "sealed" and seal["status_sha256"] == sha(run/"forecast-status.json")
        assert seal["pre_audit_sha256"] == sha(run/"pre-audit.json") and read(run/"pre-audit.json")["status"] == "passed"
        assert seal["design_sha256"] == sha(run/"design.json") and native["forecast_seal_sha256"] == proof["seal_sha256"] == sha(run/"forecast-seal.json")
        assert native["proof_sha256"] == sha(run/"measurement-proof.json") and proof["accepted_utc"] > seal["created_utc"]
        assert proof["mode"] == ("local_replication" if design["replication"] else "public_remote")
        if not design["replication"]:
            assert proof["commit"] == proof["remote_head"]
        assert final["status_sha256"] == sha(run/"status.json") and final["forecast_seal_sha256"] == sha(run/"forecast-seal.json") and final["cases"] == native["cases"]
        assert native["counts"] == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=24)
    for number, (case, entry) in enumerate(zip(read(run/"sources/cases256.json"), status["cases"], strict=True)):
        key, start = case["key"], case["position"]
        assert key == entry["key"]
        a, b = case["false_ids"], case["true_ids"]
        assert len(a) == len(b) and [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y] == [start]
        assert a[start] == 895 and b[start] == 830
        hashes = [hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() for ids in (a, b)]
        assert hashes == case["input_hashes"] and not set(hashes) & (seen | catalog)
        seen.update(hashes)
        strata[(case["group"]["schema"], case["group"]["reverse"])] += 1
        path = run/"forecasts"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        assert sha(path) == entry["arrays_sha256"] == meta["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert meta["ordinary_true"] == 0 and len(meta["false_prefix_hashes"]) == 28 and meta["witness_sha256"] == entry["witness_sha256"]
        for check in meta["checks"]:
            assert np.isfinite(check["error"]) and check["error"] <= check["bound"]
        with np.load(path) as f:
            for arm in ("F", "Z", "L", "Q", "C", "R"):
                assert np.isfinite(f[arm+"_trajectory"]).all() and f[arm+"_trajectory"].shape == f["F_trajectory"].shape
                checked(checks, "CPU64_forecast_finite_head", head(weights, f[arm+"_trajectory"][-1, -1]), f[arm+"_scores"], .005)
            checked(checks, "False_zero_trajectory", f["F_trajectory"], f["Z_trajectory"], 0.)
            checked(checks, "False_zero_scores", f["F_scores"], f["Z_scores"], 0.)
            for layer in range(1, 29):
                x, z, k = f[f"F_r{layer}"], f[f"F_z{layer}"], f[f"F_k{layer}"]
                assert x.shape == z.shape == (len(a)-start-1, 1536) and k.ndim == 3 and k.shape[-2] == len(a)-start-1
                assert np.isfinite(x).all() and np.isfinite(z).all() and np.isfinite(k).all()
            for layer in range(1, 29):
                base, anchor = f[f"F_r{layer}"].astype(float), f[f"F_z{layer}"].astype(float)
                gamma, eps = weights["input_norm"][layer-1], float(weights["input_eps"])
                for arm in ("Z", "L", "Q", "C", "R"):
                    point = base if layer == 1 else f[arm+"_trajectory"][layer-2, 1:].astype(float)
                    expected, _, _, _ = local_input(base, point, anchor, gamma, eps, arm, key, layer, checks)
                    checked(checks, "every_layer_actual_query_input", f[f"{arm}_z{layer}"], expected, 0.)
            witness_count += witness_checks(run, case, "forecast", entry, f, weights, query_weights, checks, raw_checks)
            if phase == "pre":
                continue
            ent = native["cases"][number]
            assert ent["key"] == key
            npth = run/"observations"/path.name
            nm = read(npth.with_suffix(".json"))
            assert sha(npth) == ent["arrays_sha256"] == nm["arrays_sha256"] and sha(npth.with_suffix(".json")) == ent["record_sha256"]
            assert nm["forecast_arrays_sha256"] == entry["arrays_sha256"] and nm["forecast_record_sha256"] == entry["record_sha256"]
            assert nm["native_started_utc"] > proof["accepted_utc"] and nm["witness_sha256"] == ent["witness_sha256"]
            for check in nm["checks"]:
                assert np.isfinite(check["error"]) and check["error"] <= check["bound"]
            with np.load(npth) as t:
                witness_count += witness_checks(run, case, "native", ent, t, weights, query_weights, checks, raw_checks)
                for arm in ("T", "O"):
                    checked(checks, "CPU64_native_finite_head", head(weights, t[arm+"_trajectory"][-1, -1]), t[arm+"_scores"], .005)
                checked(checks, "oracle_True_probability_trajectory", t["O_trajectory"], t["T_trajectory"], 0.)
                checked(checks, "oracle_True_probability_scores", t["O_scores"], t["T_scores"], 0.)
                for arm in ("L", "Q", "C", "R"):
                    checked(checks, "known_edit_source_exact", f[arm+"_trajectory"][:, 0], t["T_trajectory"][:, 0], 0.)
                    checked(checks, "native_first_block_exact", f[arm+"_trajectory"][0], t["T_trajectory"][0], 0.)
                truth = t["T_trajectory"].astype(float)-f["F_trajectory"].astype(float)
                mt, mf = (float(np.diff(v.astype(float))[0]) for v in (t["T_scores"], f["F_scores"]))
                for arm in ("L", "Q", "C", "R"):
                    pred = f[arm+"_trajectory"].astype(float)-f["F_trajectory"].astype(float)
                    margin = float(np.diff(f[arm+"_scores"].astype(float))[0])
                    row = dict(key=key, **case["group"], arm=arm, passes=False, margin_error=abs(margin-mt),
                        native_margin_change=mt-mf, predicted_margin_change=margin-mf,
                        source_final_error=float(np.linalg.norm(f[arm+"_trajectory"][-1, 0].astype(float)-t["T_trajectory"][-1, 0].astype(float))))
                    valid = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        target, value = truth[-1, pick], pred[-1, pick]
                        rel, cosine = geometry(value, target)
                        row.update({scope+"_relative":rel, scope+"_cosine":cosine, scope+"_norm":float(np.linalg.norm(target)),
                            scope+"_signed_share":float(np.vdot(value, target)/max(np.vdot(target, target), 1e-30))})
                        valid.append(np.linalg.norm(target) >= .001 and rel <= .05 and cosine >= .99)
                        for i in range(28):
                            rel_i, cos_i = geometry(pred[i, pick], truth[i, pick])
                            layers.append(dict(key=key, **case["group"], arm=arm, layer=i+1, scope=scope, relative_error=rel_i, cosine=cos_i))
                    row["passes"] = bool(all(valid) and row["margin_error"] <= .10)
                    rows.append(row)
                increments = dict(quadratic=f["Q_trajectory"].astype(float)-f["L_trajectory"].astype(float),
                    cubic=f["C_trajectory"].astype(float)-f["Q_trajectory"].astype(float),
                    random=f["R_trajectory"].astype(float)-f["L_trajectory"].astype(float))
                for term, increment in increments.items():
                    for i in range(28):
                        for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                            target, value = truth[i, pick], increment[i, pick]
                            interactions.append(dict(key=key, **case["group"], term=term, layer=i+1, scope=scope,
                                relative_norm=float(np.linalg.norm(value)/max(np.linalg.norm(target), 1e-30)),
                                signed_share=float(np.vdot(value, target)/max(np.vdot(target, target), 1e-30))))
        print("local polynomial audit", phase, key[:8], flush=True)
    assert len(seen) == 64 and len(strata) == 4 and set(strata.values()) == {8}
    assert witness_count == (72 if phase == "pre" else 96)
    raw_name = "raw-logit-checks-"+phase+".csv"
    table(run/raw_name, raw_checks)
    result = dict(numerical_boundary="Composed probabilities/outputs; raw-logit absolute criterion is separately reported", raw_logit_checks_sha256=sha(run/raw_name), raw_logit_checks=dict(total=len(raw_checks), failed=sum(not c["passes"] for c in raw_checks), maximum_error=max(c["error"] for c in raw_checks), original_bound=1e-4), status="passed", phase=phase, checks=checks, ordinary_true=0 if phase == "pre" else 32,
        scorer_sha256=sha(__file__), design_sha256=sha(run/"design.json"), forecast_status_sha256=sha(run/"forecast-status.json"),
        witness_coverage=dict(contexts=4, layers=[1, 14, 28], kernels=witness_count, full_forward_reconstruction=False))
    if phase == "final":
        for name, data in (("outcomes", rows), ("layer-errors", layers), ("polynomial-increments", interactions)):
            table(run/(name+".csv"), data)
        decisions = {a:decision([r for r in rows if r["arm"] == a]) for a in ("L", "Q", "C", "R")}
        result.update(primary=decisions["C"], decisions=decisions, seal_sha256=sha(run/"seal.json"), forecast_seal_sha256=sha(run/"forecast-seal.json"),
            csv_hashes={n:sha(run/(n+".csv")) for n in ("outcomes", "layer-errors", "polynomial-increments")})
        print(json.dumps(decisions, indent=2))
    write_json(destination, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.phase)
    except Exception as exc:
        write_json(args.run/("failure-audit-"+args.phase+".json"), dict(error=repr(exc)))
        raise
