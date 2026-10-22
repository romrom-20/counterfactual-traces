"""Separate smooth-derivative, gate-vector and native-outcome audit for261d: retain failed bridges and bound their effect on local decisions."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha
from .checkpoint238 import locate, read_head
from .core225b import write_json
from .curvature255 import jet
from .silu258 import predict

ROOT = Path(__file__).resolve().parents[2]


def table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    return json.loads(path.read_text())


def norm(x, gamma, eps):
    return x*gamma/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)


def expansion(base, point, anchor, gamma, eps, checks, counts):
    d = point-base
    s = np.sqrt(np.mean(base*base, axis=-1, keepdims=True)+eps)
    linear = gamma*(d/s-base*np.mean(base*d, axis=-1, keepdims=True)/s**3)
    a, b = (np.mean(value, axis=-1, keepdims=True)/s**2 for value in (base*d, d*d))
    q = gamma/s*((1.5*a*a-.5*b)*base-a*d)
    c = gamma/s*((1.5*a*b-2.5*a**3)*base+(1.5*a*a-.5*b)*d)
    for expected, actual in zip(jet(base, d, gamma, eps), (linear, q, c), strict=True):
        checked(checks, "independent_normalization_polynomial_jet", actual, expected, 1e-10)
    base_t, direction_t, gamma_t = (torch.tensor(v, dtype=torch.float64) for v in (base, d, gamma))

    def smooth(x):
        return gamma_t*x/torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True)+eps)

    _, ad = torch.func.jvp(smooth, (base_t,), (direction_t,))
    checked(checks, "smooth_RMS_derivative_AD", ad.numpy(), linear, 1e-9)
    direction = d/max(1., float(np.max(abs(d))))
    tangent = linear/max(1., float(np.max(abs(d))))
    for step, tolerance in ((1e-3, 1e-3), (1e-4, 1e-5)):
        finite_difference = (norm(base+step*direction, gamma, eps)-norm(base-step*direction, gamma, eps))/(2*step)
        checked(checks, "smooth_RMS_derivative_FD", finite_difference, tangent, tolerance)
    counts["normalization_AD"] += 1
    counts["normalization_FD"] += 2
    counts["independent_polynomial_jets"] += 1
    return anchor+linear, q+c, linear


def random_rows(value, key, layer):
    rows = []
    for row in range(len(value)):
        digest = hashlib.sha256(f"{key}:{layer}:{row}:mlp-input-curvature-control".encode()).hexdigest()
        vector = np.random.default_rng(int(digest[:16], 16)).normal(size=value.shape[-1])
        rows.append(vector/np.linalg.norm(vector)*np.linalg.norm(value[row]))
    return np.stack(rows)


def metrics(value, target):
    vn, tn = float(np.linalg.norm(value)), float(np.linalg.norm(target))
    return dict(relative=float(np.linalg.norm(value-target))/max(tn, 1e-30),
        cosine=float(np.vdot(value, target))/max(vn*tn, 1e-30), norm=tn,
        signed_share=float(np.vdot(value, target))/max(tn*tn, 1e-30))


def main(run, replay, out):
    assert not out.exists()
    out.mkdir(parents=True)
    investigation = read(replay/"investigation.json")
    assert investigation["source_sha256"] == sha(ROOT/"research/boolean_coupling/replay261c.py")
    assert investigation["protocol_sha256"] == sha(ROOT/"docs/experiments/261c-projection-replay.md")
    assert investigation["csv_sha256"] == sha(replay/"projection-replays.csv")
    assert investigation["comparisons"] == investigation["exact_replays"] == 1568
    assert investigation["maximum_full_shape_error"] == 0.
    assert investigation["input_design_sha256"] == sha(run/"design.json")
    assert investigation["input_status_sha256"] == sha(run/"status.json")
    with (replay/"projection-replays.csv").open() as stream:
        raw_rows = list(csv.DictReader(stream))
    raw_lookup = {(r["key"], r["arm"], int(r["layer"]), r["branch"]):r for r in raw_rows}
    assert len(raw_lookup) == 1568
    raw_confirmed, raw_displacements = [], []
    write_json(out/"qualification.json", dict(retrospective=True, original_failure_sha256=sha(run/"failure-audit.json"),
        original_status_sha256=sha(run/"status.json"), replay_sha256=sha(replay/"investigation.json"),
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/261d-rounding-bounds.md"),
        previous_261c_failure_sha256=sha(run.parent/"mlp-input-qualified-261c/failure-audit.json"),
        note="Original absolute displacement bridge also failed; retain all errors and bound public local classification conservatively. No original failed check is relabeled passed."))
    design, status = read(run/"design.json"), read(run/"status.json")
    assert design["experiment"] == "261b" and design["retrospective"] and status["status"] == "complete"
    assert len(design["cases"]) == len(status["cases"]) == 4
    for name, digest in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == digest
    assert sha(run/"norm-weights.npz") == status["norm_weights_sha256"]
    folder, checkpoint = locate(ROOT)
    head = read_head(folder)
    with np.load(run/"norm-weights.npz") as weights:
        norms = {key:weights[key].astype(float) for key in weights.files}
    cases = {}
    checks, rows, outcomes, counts = [], [], [], Counter()
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        key = case["key"]
        assert key == entry["key"]
        path = run/"cases"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        assert sha(path) == meta["arrays_sha256"] == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        for record in meta["checks"]:
            assert np.isfinite(record["error"]) and record["error"] <= record["bound"]
        cases[key] = (case, np.load(path), meta)
    vectors = {key:{} for key in cases}
    eps = float(norms["eps"])
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            def tensor(name):
                return handle.get_tensor(f"model.layers.{layer-1}."+name).float().numpy().astype(float)
            gamma = tensor("post_attention_layernorm.weight")
            qgamma = tensor("input_layernorm.weight")
            checked(checks, "checkpoint_MLP_norm_weight", gamma, norms["mlp_norm"][layer-1], 0.)
            checked(checks, "checkpoint_query_norm_weight", qgamma, norms["input_norm"][layer-1], 0.)
            matrices = {name:tensor("mlp."+name+"_proj.weight") for name in ("gate", "up")}
            for key, (case, values, meta) in cases.items():
                prefix0 = f"F_{layer}_"
                x0, z0 = (values[prefix0+k].astype(float) for k in ("x", "original_z"))
                qx0, qz0 = (values[prefix0+k].astype(float) for k in ("qx", "qoriginal"))
                for arm in ("F", "Z", "B", "L", "C", "R", "T"):
                    prefix = f"{arm}_{layer}_"
                    if prefix+"write" not in values:
                        assert meta["statuses"][arm]["status"] == "guard_failed"
                        continue
                    x = values[prefix+"x"].astype(float)
                    original_z, installed_z = (values[prefix+k].astype(float) for k in ("original_z", "installed_z"))
                    checked(checks, "public_MLP_normalization_primal", norm(x, gamma, eps), original_z, 1e-4)
                    linear, curvature, j = expansion(x0, x, z0, gamma, eps, checks, counts)
                    expected = original_z
                    if arm in ("Z", "L", "C", "R"):
                        expected = linear if arm in ("Z", "L") else linear+(curvature if arm == "C" else random_rows(curvature, key, layer))
                        expected = expected.astype(np.float32).astype(float)
                    checked(checks, "exact_MLP_normalization_installation", installed_z, expected, 0.)
                    qx = values[prefix+"qx"].astype(float)
                    qo, qi = (values[prefix+k].astype(float) for k in ("qoriginal", "qinstalled"))
                    checked(checks, "public_query_normalization_primal", norm(qx, qgamma, eps), qo, 1e-4)
                    qlinear, qcurvature, _ = expansion(qx0, qx, qz0, qgamma, eps, checks, counts)
                    qexpected = (qlinear+qcurvature).astype(np.float32).astype(float) if arm in ("Z", "B", "L", "C", "R") else qo
                    checked(checks, "exact_cubic_query_installation", qi, qexpected, 0.)
                    for branch in ("gate", "up"):
                        error = float(np.max(abs(installed_z@matrices[branch].T-values[prefix+branch])))
                        raw_record = raw_lookup[(key, arm, layer, branch)]
                        checked(checks, "retained_raw_projection_error", np.array(error), np.array(float(raw_record["CPU64_error"])), 1e-12)
                        assert float(raw_record["original_bound"]) == 1e-4 and float(raw_record["full_shape_error"]) == 0.
                        raw_confirmed.append(dict(key=key, arm=arm, layer=layer, branch=branch, error=error, original_bound=1e-4, passes=error <= 1e-4))
                        counts["CPU64_gate_up_projections"] += 1
                    if arm in ("Z", "B", "L", "C", "R"):
                        quadratic, mask, _, bound, _, _ = predict(values[prefix0+"gate"].astype(float), values[prefix0+"phi"].astype(float),
                            values[prefix+"gate"].astype(float), values[prefix+"up"].astype(float), norms["columns"][layer-1], key, layer)
                        checked(checks, "original_ranked_mask", mask.astype(int), values[prefix+"mask"].astype(int), 0.)
                        checked(checks, "original_quadratic_SiLU", quadratic, values[prefix+"quadratic"], 0.)
                        checked(checks, "original_remainder_bound", bound, values[prefix+"bound"], 0.)
                        used = np.where(mask, values[prefix+"phi"], quadratic.astype(np.float32)).astype(np.float32)
                        checked(checks, "exact_sparse_SiLU", used, values[prefix+"used_phi"], 0.)
                    checked(checks, "exact_neuron_product", values[prefix+"used_phi"]*values[prefix+"up"], values[prefix+"product"], 0.)
                    if arm not in ("B", "T"):
                        continue
                    mask = predict(values[prefix0+"gate"].astype(float), values[prefix0+"phi"].astype(float),
                        values[prefix+"gate"].astype(float), values[prefix+"up"].astype(float), norms["columns"][layer-1], key, layer)[1]
                    smooth_delta = norm(x, gamma, eps)-norm(x0, gamma, eps)
                    for branch, matrix in matrices.items():
                        projected_linear, projected_curvature, target = (v@matrix.T for v in (j, smooth_delta-j, smooth_delta))
                        public_target = values[prefix+branch].astype(float)-values[prefix0+branch].astype(float)
                        checked(checks, "gate_up_displacement_algebra", projected_linear+projected_curvature, target, 1e-9)
                        arithmetic = public_target-target
                        raw_error = float(np.max(abs(arithmetic)))
                        raw_displacements.append(dict(key=key, arm=arm, layer=layer, branch=branch,
                            error=raw_error, original_bound=1e-4, passes=raw_error <= 1e-4))
                        checked(checks, "explicit_smooth_public_difference_identity", projected_linear+projected_curvature+arithmetic, public_target, 1e-9)
                        counts["CPU64_gate_up_projections"] += 3
                        for name, vector in (("linear", projected_linear), ("curvature", projected_curvature), ("total", target), ("public", public_target), ("arithmetic", arithmetic)):
                            vectors[key][f"{arm}_{layer}_{branch}_{name}"] = vector
                        for scope, pick in (("all", slice(None)), ("verdict", -1), ("selected", mask)):
                            m = metrics(projected_linear[pick], target[pick])
                            cur = metrics(projected_curvature[pick], target[pick])
                            public_norm = float(np.linalg.norm(public_target[pick]))
                            curvature_norm = float(np.linalg.norm(projected_curvature[pick]))
                            arithmetic_norm = float(np.linalg.norm(arithmetic[pick]))
                            lower = max(0., curvature_norm-arithmetic_norm)/max(public_norm, 1e-30)
                            upper = (curvature_norm+arithmetic_norm)/max(public_norm, 1e-30)
                            actual_public = metrics(projected_linear[pick], public_target[pick])
                            checked(checks, "public_error_triangle_lower", np.array(max(0., lower-actual_public["relative"])), np.array(0.), 1e-9)
                            checked(checks, "public_error_triangle_upper", np.array(max(0., actual_public["relative"]-upper)), np.array(0.), 1e-9)
                            rows.append(dict(key=key, arm=arm, layer=layer, branch=branch, scope=scope,
                                eligible=m["norm"] >= 1e-8, linear_relative=m["relative"], linear_cosine=m["cosine"],
                                linear_signed_share=m["signed_share"], target_norm=m["norm"], curvature_norm=float(np.linalg.norm(projected_curvature[pick])),
                                curvature_signed_share=cur["signed_share"], passes=m["norm"] >= 1e-8 and m["relative"] <= .05,
                                public_norm=public_norm, public_linear_relative=actual_public["relative"],
                                arithmetic_norm=arithmetic_norm, arithmetic_relative=arithmetic_norm/max(public_norm, 1e-30),
                                public_error_lower=lower, public_error_upper=upper,
                                robust_public_pass=public_norm >= 1e-8 and upper <= .05,
                                robust_public_reject=public_norm >= 1e-8 and lower > .05,
                                public_ambiguous=public_norm >= 1e-8 and lower <= .05 < upper))
            print("MLP input derivative/vector audit", layer, "/28", flush=True)
    for key, (case, values, meta) in cases.items():
        np.savez_compressed(out/("vectors-"+key+".npz"), **vectors[key])
        truth = values["T_trajectory"].astype(float)-values["F_trajectory"].astype(float)
        margin_true = float(np.diff(values["T_scores"].astype(float))[0])
        for arm in ("B", "L", "C", "R"):
            row = dict(key=key, **case["group"], arm=arm, status=meta["statuses"][arm]["status"], passes=False)
            if row["status"] == "complete":
                trajectory = values[arm+"_trajectory"]
                last = trajectory[-1, -1].astype(float)
                z = last/np.sqrt(np.mean(last*last)+head["final_eps"])*head["final_norm"]
                checked(checks, "finite_two_score_head", z@head["final_head"].T, values[arm+"_scores"], .005)
                pred = trajectory.astype(float)-values["F_trajectory"].astype(float)
                passing = []
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    m = metrics(pred[-1, pick], truth[-1, pick])
                    row.update({scope+"_"+name:value for name, value in m.items()})
                    passing.append(m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99)
                row["margin_error"] = abs(float(np.diff(values[arm+"_scores"].astype(float))[0])-margin_true)
                row["passes"] = bool(all(passing) and row["margin_error"] <= .1)
            outcomes.append(row)
        values.close()
    table(out/"gate-curvature.csv", rows)
    table(out/"outcomes.csv", outcomes)
    local = {}
    for arm in ("B", "T"):
        for scope in ("all", "selected"):
            group = [r for r in rows if r["arm"] == arm and r["branch"] == "gate" and r["scope"] == scope and r["eligible"]]
            local[arm+"_"+scope] = dict(eligible=len(group), passes=sum(r["passes"] for r in group),
                median_linear_relative=float(np.median([r["linear_relative"] for r in group])),
                meets_90=bool(group and sum(r["passes"] for r in group) >= .9*len(group)),
                robust_public_passes=sum(r["robust_public_pass"] for r in group),
                robust_public_rejects=sum(r["robust_public_reject"] for r in group),
                public_ambiguous=sum(r["public_ambiguous"] for r in group),
                maximum_arithmetic_relative=max(r["arithmetic_relative"] for r in group))
    global_results = {arm:dict(total=4, passes=sum(r["passes"] for r in outcomes if r["arm"] == arm),
        guard_failed=sum(r["status"] == "guard_failed" for r in outcomes if r["arm"] == arm)) for arm in ("B", "L", "C", "R")}
    table(out/"raw-projection-checks.csv", raw_confirmed)
    table(out/"raw-displacement-checks.csv", raw_displacements)
    assert len(raw_confirmed) == 1568
    write_json(out/"audit.json", dict(status="passed", experiment="261d", retrospective=True,
        qualification_sha256=sha(out/"qualification.json"), replay_sha256=sha(replay/"investigation.json"),
        raw_projection_checks=dict(total=len(raw_confirmed), failed=sum(not r["passes"] for r in raw_confirmed), original_bound=1e-4),
        raw_projection_csv_sha256=sha(out/"raw-projection-checks.csv"),
        raw_displacement_checks=dict(total=len(raw_displacements), failed=sum(not r["passes"] for r in raw_displacements), original_bound=1e-4),
        raw_displacement_csv_sha256=sha(out/"raw-displacement-checks.csv"),
        numerical_boundary="Exact source-bound publicFloat32 primals; original CPU64 primal/displacement failures retained; public5% local classification bounded by measured arithmetic difference, not a new predicted correction.", source_sha256=sha(__file__),
        design_sha256=sha(run/"design.json"), status_sha256=sha(run/"status.json"), checkpoint=checkpoint,
        checks=checks, extra_work=dict(counts), local_gate=local, global_results=global_results,
        csv_sha256={name:sha(out/name) for name in ("gate-curvature.csv", "outcomes.csv")},
        vectors_sha256={p.name:sha(p) for p in out.glob("vectors-*.npz")},
        limitation="Four old fixed contexts, own-state primitive policies, smooth derivatives only; not a learned circuit, new-template confirmation or speedup."))
    print(json.dumps(dict(local=local, global_results=global_results), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve(), args.replay.resolve(), args.out.resolve())
    except Exception as exc:
        write_json(args.out/"failure-audit.json", dict(error=repr(exc)))
        raise
