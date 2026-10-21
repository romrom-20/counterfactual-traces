"""Explicit matched-shape Float32 qualification; raw CPU64 primal failures retained."""

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
from .audit256 import local_input
from .checkpoint238 import locate
from .core225b import write_json
from .silu258 import sigmoid

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)



def witness_checks(run, case, phase, entry, data, weights, query_weights, checks, raw_checks):
    firsts = {}
    for c in read(run/"sources/cases258.json"):
        firsts.setdefault((c["group"]["schema"], c["group"]["reverse"]), c["key"])
    assert len(firsts) == 4
    if case["key"] not in firsts.values():
        assert entry["witness_sha256"] is None
        return 0
    wp = run/("witnesses-forecast" if phase == "forecast" else "witnesses-native")/(case["key"]+".npz")
    assert sha(wp) == entry["witness_sha256"]
    count = 0
    with np.load(wp) as w:
        for arm in (("F", "Z", "C", "Q", "S", "R") if phase == "forecast" else ("T", "O")):
            for layer in (1, 14, 28):
                prefix = f"{arm}_{layer}_"
                q, k, v, raw, mask, installed, output, current_k = (w[prefix+n].astype(float) for n in ("q", "k", "v", "raw", "mask", "installed_p", "y", "current_k"))
                x, finite_z, used_z, linear, cos, sin = (w[prefix+n].astype(float) for n in ("x", "original_z", "installed_z", "q_linear", "cos", "sin"))
                gamma, eps = weights["input_norm"][layer-1], float(weights["input_eps"])
                normalized = x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)*gamma
                checked(checks, "CPU64_native_normalized_input", normalized, finite_z, 1e-4)
                expected_z = finite_z.copy()
                if arm in ("Z", "C", "Q", "S", "R"):
                    base, anchor = data[f"F_r{layer}"].astype(float), data[f"F_z{layer}"].astype(float)
                    direction = x-base
                    scale = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+eps)
                    derivative = gamma*(direction/scale-base*np.mean(base*direction, axis=-1, keepdims=True)/scale**3)
                    expected_z, correction, _, quadratic = local_input(base, x, anchor, gamma, eps, "C", case["key"], layer, [])
                    checked(checks, "actual_local_curvature_used", w[prefix+"correction"].astype(float), correction, 0.)
                    def smooth(value):
                        return value/np.sqrt(np.mean(value*value, axis=-1, keepdims=True)+eps)*gamma
                    divisor = max(1., float(np.max(abs(direction))))
                    direction = direction/divisor
                    derivative = derivative/divisor
                    quadratic = quadratic/divisor**2
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



def mlp_witnesses(run, case, phase, entry, data, down, replay_down, checks, raw_down):
    if not entry["witness_sha256"]:
        return
    wp = run/("witnesses-forecast" if phase == "forecast" else "witnesses-native")/(case["key"]+".npz")
    with np.load(wp) as w:
        for arm in (("F", "Z", "C", "Q", "S", "R") if phase == "forecast" else ("T", "O")):
            for layer in (1, 14, 28):
                prefix = f"{arm}_{layer}_mlp_"
                gate, phi, used, up, product, write = (w[prefix+n].astype(float) for n in ("gate", "finite_phi", "used_phi", "up", "product", "write"))
                checked(checks, "independent_public_SiLU", gate*sigmoid(gate), phi, 1e-4)
                expected = phi
                if arm in ("Z", "C", "Q", "S", "R"):
                    base, anchor = w[prefix+"base"].astype(float), w[prefix+"anchor"].astype(float)
                    checked(checks, "False_gate_anchor", base, w[f"F_{layer}_mlp_gate"].astype(float), 0.)
                    checked(checks, "False_phi_anchor", anchor, w[f"F_{layer}_mlp_finite_phi"].astype(float), 0.)
                    tb = torch.tensor(base, dtype=torch.float64, requires_grad=True)
                    value = tb*torch.sigmoid(tb)
                    first = torch.autograd.grad(value.sum(), tb, create_graph=True)[0]
                    second = torch.autograd.grad(first.sum(), tb)[0].detach().numpy()
                    first = first.detach().numpy()
                    sigma = sigmoid(base)
                    f1 = sigma*(1+base*(1-sigma))
                    f2 = sigma*(1-sigma)*(2+base*(1-2*sigma))
                    checked(checks, "SiLU_first_CPU64_AD", first, f1, 1e-9)
                    checked(checks, "SiLU_second_CPU64_AD", second, f2, 1e-9)
                    delta = gate-base
                    quadratic = anchor+f1*delta+.5*f2*delta*delta
                    checked(checks, "independent_quadratic", quadratic, w[prefix+"quadratic"], 1e-10)
                    low, high = np.minimum(base, gate), np.maximum(base, gate)
                    closest = np.where(low > 0, low, np.where(high < 0, high, 0.))
                    exponential = np.exp(-abs(closest))
                    bound = exponential/(1+exponential)**2*(3+np.maximum(abs(base), abs(gate)))*abs(delta)**3/6
                    checked(checks, "independent_remainder_bound", bound, w[prefix+"bound"], 1e-8)
                    matrix = down[layer]
                    columns = np.sqrt(np.sum(matrix*matrix, axis=0))
                    checked(checks, "public_down_columns_witness", columns, w[prefix+"columns"], 1e-10)
                    score = bound*abs(up)*columns
                    rank = np.zeros(score.size, dtype=bool)
                    rank[np.argsort(score.ravel(), kind="stable")[-int(.01*score.size):]] = True
                    rank = rank.reshape(score.shape)
                    mask = rank.copy()
                    if arm == "R":
                        mask[:] = False
                        for row in range(len(mask)):
                            seed = int(hashlib.sha256(f"{case['key']}:{layer}:{row}:receiver-silu-control".encode()).hexdigest()[:16], 16)
                            chosen = np.random.default_rng(seed).choice(mask.shape[1], int(rank[row].sum()), replace=False)
                            mask[row, chosen] = True
                    checked(checks, "independent_rank_mask", rank.astype(int), w[prefix+"rank"].astype(int), 0.)
                    checked(checks, "independent_selected_mask", mask.astype(int), w[prefix+"mask"].astype(int), 0.)
                    assert mask.sum() == int(.01*mask.size)
                    checked(checks, "mask_row_counts_archive", mask.sum(axis=-1), data[f"{arm}_mask_counts{layer}"], 0.)
                    assert hashlib.sha256(mask.tobytes()).digest() == data[f"{arm}_mask_hash{layer}"].astype(np.uint8).tobytes()
                    direction = delta/max(1., float(np.max(abs(delta))))
                    step = 1e-5
                    fd1 = ((base+step*direction)*sigmoid(base+step*direction)-(base-step*direction)*sigmoid(base-step*direction))/(2*step)
                    checked(checks, "independent_SiLU_first_FD", fd1, f1*direction, 1e-5)
                    step = 1e-3
                    fd2 = ((base+step*direction)*sigmoid(base+step*direction)-2*base*sigmoid(base)+(base-step*direction)*sigmoid(base-step*direction))/(2*step**2)
                    checked(checks, "independent_SiLU_second_FD", fd2, .5*f2*direction**2, 1e-5)
                    if arm != "C":
                        expected = quadratic.astype(np.float32)
                        if arm != "Q":
                            expected = np.where(mask, phi.astype(np.float32), expected)
                        expected = expected.astype(float)
                    if arm != "Z":
                        tail = phi-quadratic
                        restored = (np.where(mask, tail, 0.)*up).astype(np.float32).astype(float)@matrix.T
                        remaining = (np.where(mask, 0., tail)*up).astype(np.float32).astype(float)@matrix.T
                        finite_product = (phi.astype(np.float32)*up.astype(np.float32)).astype(float)
                        quad_product = (quadratic.astype(np.float32)*up.astype(np.float32)).astype(float)
                        fw, qw = finite_product@matrix.T, quad_product@matrix.T
                        checked(checks, "independent_restored_write", restored, data[f"{arm}_restore{layer}"], 1e-4)
                        checked(checks, "independent_remaining_write", remaining, data[f"{arm}_remain{layer}"], 1e-4)
                        checked(checks, "independent_finite_write", fw, w[prefix+"finite_write"], 1e-4)
                        checked(checks, "independent_quadratic_write", qw, w[prefix+"quadratic_write"], 1e-4)
                        checked(checks, "independent_signed_gap", fw-qw, data[f"{arm}_gap{layer}"], 1e-4)
                checked(checks, "independent_SiLU_installation", used, expected, 0.)
                expected_product = (used.astype(np.float32)*up.astype(np.float32)).astype(float)
                checked(checks, "independent_product_installation", product, expected_product, 0.)
                raw_error = float(np.max(abs(product@down[layer].T-write)))
                raw_down.append(dict(key=case["key"], arm=arm, block=layer, error=raw_error, original_bound=1e-4, passes=raw_error <= 1e-4, maximum_write=float(np.max(abs(write)))))
                with torch.no_grad():
                    full = torch.zeros((1, len(case["false_ids"]), product.shape[-1]), device="mps", dtype=torch.float32)
                    full[:, case["position"]+1:] = torch.tensor(product.astype(np.float32)[None], device="mps")
                    replayed = torch.nn.functional.linear(full, replay_down[layer], None)[0, case["position"]+1:].cpu().numpy()
                checked(checks, "public_full_shape_Float32_down_replay", replayed.astype(float), write, 0.)


def main(run, phase):
    destination = run/("pre-audit.json" if phase == "pre" else "audit.json")
    assert not destination.exists()
    design, status = read(run/"design.json"), read(run/"forecast-status.json")
    qualification = read(run/"qualification258b.json")
    assert qualification["design_sha256"] == sha(run/"design.json") and qualification["forecast_status_sha256"] == sha(run/"forecast-status.json")
    for path, expected in qualification["source_hashes"].items():
        assert sha(ROOT/path) == expected
    assert qualification["original_scorer_sha256"] == sha(ROOT/"research/boolean_coupling/audit258.py")
    assert design["experiment"] == 258 and design["primary"] == "S" and status["status"] == "complete" and status["ordinary_true"] == 0
    assert len(status["cases"]) == 32 and status["counts"] == dict(full_forwards=192, CPU64_tangent_maps=4480, CPU64_quadratic_maps=4480, CPU64_cubic_maps=4480, CPU64_SiLU_selections=4480, CPU64_SiLU_FD_checks=8960, False_mlp_anchors=896, extra_receiving_down_projections=14336, extra_full_QK_softmax_PV_each=896, witness_extra_QK=72)
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(run/"sources"/Path(p).name) == h
    assert sha(run/"weights.npz") == design["weights_sha256"]
    weights = dict(np.load(run/"weights.npz"))
    checkpoint, identity = locate(ROOT)
    assert identity == design["checkpoint"]
    query_weights, down_weights = {}, {}
    with safe_open(checkpoint/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in (1, 14, 28):
            prefix = f"model.layers.{layer-1}.self_attn.q_proj."
            query_weights[layer] = {n:handle.get_tensor(prefix+n).to(torch.float32).numpy().astype(float) for n in ("weight", "bias")}
            down_weights[layer] = handle.get_tensor(f"model.layers.{layer-1}.mlp.down_proj.weight").float().numpy().astype(float)
        for i in range(28):
            norm = handle.get_tensor(f"model.layers.{i}.input_layernorm.weight").float().numpy().astype(float)
            checked([], "checkpoint_norm_weight", norm, weights["input_norm"][i], 0.)
            matrix = handle.get_tensor(f"model.layers.{i}.mlp.down_proj.weight").float().numpy().astype(float)
            checked([], "public_down_column_norm", np.sqrt(np.sum(matrix*matrix, axis=0)), weights["down_columns"][i], 1e-10)
    catalog = set(read(run/"sources/historical-input-hashes-258.json")["hashes"])
    seen, strata, checks, rows, layers, interactions, vectors = set(), Counter(), [], [], [], [], []
    replay_down = {layer:torch.tensor(matrix.astype(np.float32), device="mps") for layer, matrix in down_weights.items()}
    witness_count, raw_checks, raw_down = 0, [], []
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
    for number, (case, entry) in enumerate(zip(read(run/"sources/cases258.json"), status["cases"], strict=True)):
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
            for arm in ("F", "Z", "C", "Q", "S", "R"):
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
                for arm in ("Z", "C", "Q", "S", "R"):
                    point = base if layer == 1 else f[arm+"_trajectory"][layer-2, 1:].astype(float)
                    expected, _, _, _ = local_input(base, point, anchor, gamma, eps, "C", key, layer, checks)
                    assert hashlib.sha256(expected.tobytes()).digest() == f[f"{arm}_z_hash{layer}"].astype(np.uint8).tobytes()
            witness_count += witness_checks(run, case, "forecast", entry, f, weights, query_weights, checks, raw_checks)
            mlp_witnesses(run, case, "forecast", entry, f, down_weights, replay_down, checks, raw_down)
            for arm in ("C", "Q", "S", "R"):
                for layer in range(1, 29):
                    restored, remaining, gap = (f[f"{arm}_{term}{layer}"].astype(float) for term in ("restore", "remain", "gap"))
                    checked(checks, "all_saved_signed_vector_closure", restored+remaining, gap, 1e-4)
                    for scope, pick in (("all", slice(None)), ("verdict", -1)):
                        target = gap[pick]
                        norm = float(np.linalg.norm(target))
                        row = dict(key=key, **case["group"], arm=arm, layer=layer, scope=scope, eligible=norm >= 1e-8, gap_norm=norm)
                        for term, value in (("restore", restored[pick]), ("remain", remaining[pick])):
                            row[term+"_norm"] = float(np.linalg.norm(value))
                            row[term+"_signed_share"] = float(np.vdot(value, target)/max(np.vdot(target, target), 1e-30))
                            row[term+"_relative_gap_error"] = float(np.linalg.norm(value-target))/max(norm, 1e-30)
                        row["restore_remain_cosine"] = geometry(restored[pick], remaining[pick])[1]
                        vectors.append(row)
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
                mlp_witnesses(run, case, "native", ent, t, down_weights, replay_down, checks, raw_down)
                for arm in ("T", "O"):
                    checked(checks, "CPU64_native_finite_head", head(weights, t[arm+"_trajectory"][-1, -1]), t[arm+"_scores"], .005)
                checked(checks, "oracle_True_probability_trajectory", t["O_trajectory"], t["T_trajectory"], 0.)
                checked(checks, "oracle_True_probability_scores", t["O_scores"], t["T_scores"], 0.)
                for arm in ("C", "Q", "S", "R"):
                    checked(checks, "known_edit_source_exact", f[arm+"_trajectory"][:, 0], t["T_trajectory"][:, 0], 0.)
                    if arm == "C":
                        checked(checks, "native_first_block_exact", f[arm+"_trajectory"][0], t["T_trajectory"][0], 0.)
                truth = t["T_trajectory"].astype(float)-f["F_trajectory"].astype(float)
                mt, mf = (float(np.diff(v.astype(float))[0]) for v in (t["T_scores"], f["F_scores"]))
                for arm in ("C", "Q", "S", "R"):
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
                increments = dict(ranked=f["S_trajectory"].astype(float)-f["Q_trajectory"].astype(float),
                    random=f["R_trajectory"].astype(float)-f["Q_trajectory"].astype(float),
                    full=f["C_trajectory"].astype(float)-f["Q_trajectory"].astype(float))
                for term, increment in increments.items():
                    for i in range(28):
                        for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                            target, value = truth[i, pick], increment[i, pick]
                            interactions.append(dict(key=key, **case["group"], term=term, layer=i+1, scope=scope,
                                relative_norm=float(np.linalg.norm(value)/max(np.linalg.norm(target), 1e-30)),
                                signed_share=float(np.vdot(value, target)/max(np.vdot(target, target), 1e-30))))
        print("receiver MLP audit", phase, key[:8], flush=True)
    assert len(seen) == 64 and len(strata) == 4 and set(strata.values()) == {8}
    assert witness_count == (72 if phase == "pre" else 96)
    raw_name = "raw-logit-checks-"+phase+".csv"
    table(run/raw_name, raw_checks)
    down_name = "raw-down-checks-"+phase+".csv"
    table(run/down_name, raw_down)
    result = dict(numerical_boundary="Composed probabilities/outputs; raw-logit absolute criterion is separately reported", raw_logit_checks_sha256=sha(run/raw_name), raw_logit_checks=dict(total=len(raw_checks), failed=sum(not c["passes"] for c in raw_checks), maximum_error=max(c["error"] for c in raw_checks), original_bound=1e-4), status="passed", phase=phase, checks=checks, ordinary_true=0 if phase == "pre" else 32,
        scorer_sha256=sha(__file__), design_sha256=sha(run/"design.json"), forecast_status_sha256=sha(run/"forecast-status.json"),
        witness_coverage=dict(contexts=4, layers=[1, 14, 28], kernels=witness_count, full_forward_reconstruction=False))
    result.update(qualification_sha256=sha(run/"qualification258b.json"), original_scorer_sha256=qualification["original_scorer_sha256"],
        whole_primal_boundary="Exact matched-shape public Float32 replay; CPU64 absolute errors retained failed",
        raw_down_checks_sha256=sha(run/down_name), raw_down_checks=dict(total=len(raw_down), failed=sum(not r["passes"] for r in raw_down),
            maximum_error=max(r["error"] for r in raw_down), original_bound=1e-4), extra_full_shape_down_replays=len(raw_down))
    if phase == "final":
        for name, data in (("outcomes", rows), ("layer-errors", layers), ("policy-increments", interactions), ("signed-local-vectors", vectors)):
            table(run/(name+".csv"), data)
        decisions = {a:decision([r for r in rows if r["arm"] == a]) for a in ("C", "Q", "S", "R")}
        result.update(primary=decisions["S"], decisions=decisions, seal_sha256=sha(run/"seal.json"), forecast_seal_sha256=sha(run/"forecast-seal.json"),
            csv_hashes={n:sha(run/(n+".csv")) for n in ("outcomes", "layer-errors", "policy-increments", "signed-local-vectors")})
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
        write_json(args.run/("failure-audit-b-"+args.phase+".json"), dict(error=repr(exc)))
        raise
