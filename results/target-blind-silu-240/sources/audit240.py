"""Separate numerical reconstruction and frozen outcome scoring for240."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .audit229 import checked, sha
from .core225b import write_json


def geometry(pred, target):
    a, b = np.linalg.norm(pred), np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / b) if b else 0., float(np.vdot(pred, target) / (a * b)) if a * b else 0.


def masks_independent(p, key):
    g0, g1 = p["g0"], p["g1"]
    low, high = np.minimum(g0, g1), np.maximum(g0, g1)
    point = np.maximum(low, np.minimum(high, 0.))
    sigma = torch.sigmoid(torch.from_numpy(point)).numpy()
    bound = sigma * (1 - sigma) * (3 + np.maximum(abs(g0), abs(g1))) * abs(g1 - g0)**3 / 6
    score = bound * abs(p["u1"]) * p["down_norm"]
    k = int(np.ceil(score.size / 100))
    selected = sorted(range(score.size), key=lambda i: (-score.flat[i], i))[:k]
    mask = np.zeros(score.shape, bool)
    mask.flat[selected] = True
    masks = {"mask": mask}
    for j in range(3):
        seed = int(hashlib.sha256((key + "240" + str(j)).encode()).hexdigest()[:16], 16)
        random = np.zeros_like(mask)
        rng = np.random.default_rng(seed)
        for i, row in enumerate(mask):
            random[i, rng.choice(mask.shape[1], int(sum(row)), replace=False)] = True
        masks["random" + str(j)] = random
    return bound, score, masks


def rule(rows, flag):
    strata = {f"{s}_{rev}": sum(bool(r[flag]) for r in rows if r["schema"] == s and r["reverse"] == rev)
        for s in ("generic", "descriptive") for rev in (0, 1)}
    count = sum(bool(r[flag]) for r in rows)
    return dict(passes=count, strata=strata, meets_rule=count >= 24 and all(n >= 6 for n in strata.values()))


def main(run):
    assert not (run / "audit.json").exists()
    root = Path(__file__).resolve().parents[2]
    design, seal, status = (json.loads((run / n).read_text()) for n in ("design.json", "mask-seal.json", "status.json"))
    assert status["status"] == "complete" and status["mask_seal_sha256"] == sha(run / "mask-seal.json")
    for name, h in design["source_hashes"].items():
        assert sha(root / name) == sha(run / "sources" / Path(name).name) == h
    assert not seal["native_inputs_opened"] and not seal["intervention_outcomes_measured"]
    assert seal["counts"] == dict(ordinary_false=32, source_blocks=32, mask_mlps=32)
    assert status["counts"] == dict(intervention_mlps=384, suffix_blocks=4608, heads=384)
    assert sha(run / "weights.npz") == seal["weights_sha256"]
    w = dict(np.load(run / "weights.npz"))
    previous, old = Path(design["input239"]), Path(design["input228"])
    checks, raw, local, final, global_rows, masks_rows = [], [], [], [], [], []
    for ent, pent in zip(status["cases"], seal["cases"], strict=True):
        key = ent["key"]
        path = run / "cases" / (key + ".npz")
        meta = json.loads(path.with_suffix(".json").read_text())
        pp = run / "masks" / path.name
        pm = json.loads(pp.with_suffix(".json").read_text())
        assert sha(path) == ent["arrays_sha256"] == meta["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == ent["record_sha256"]
        assert sha(pp) == pent["arrays_sha256"] == meta["mask_arrays_sha256"] == pm["arrays_sha256"]
        assert sha(pp.with_suffix(".json")) == pent["record_sha256"]
        assert all(c["error"] <= c["bound"] for c in meta["checks"] + pm["checks"])
        d, p = dict(np.load(path)), dict(np.load(pp))
        assert sha(old / "observations" / path.name) == meta["observation228_sha256"]
        assert sha(old / "forecasts" / path.name) == pm["forecast228_sha256"]
        assert sha(Path(design["input238"]) / "cases" / path.name) == pm["upstream238_sha256"]
        group = meta["group"]
        bound, score, masks = masks_independent(p, key)
        checked(checks, "bound", bound, p["bound"], 1e-9)
        checked(checks, "score", score, p["score"], 1e-9)
        checked(checks, "down_column_norm", np.linalg.norm(w["down"].astype(float), axis=0), p["down_norm"], 1e-9)
        for name, mask in masks.items():
            checked(checks, "mask_exact", mask.astype(int), p[name].astype(int), 0.)
        t = torch.from_numpy(p["g0"])
        ones = torch.ones_like(t)
        phi, slope = torch.func.jvp(F.silu, (t,), (ones,))
        def first(v):
            return torch.func.jvp(F.silu, (v,), (ones,))[1]
        _, curve = torch.func.jvp(first, (t,), (ones,))
        s = 1 / (1 + np.exp(-p["g0"]))
        checked(checks, "SiLU_slope", slope.numpy(), s * (1 + p["g0"] * (1 - s)), 1e-10)
        checked(checks, "SiLU_curve", curve.numpy(), s * (1 - s) * (2 + p["g0"] * (1 - 2 * s)), 1e-10)
        checked(checks, "public_anchor_rounding", phi.numpy(), p["phi0"], 1e-5)
        for state in ("native", "own"):
            g = torch.from_numpy(d[state + "_gate"].astype(float))
            quadratic = (torch.from_numpy(p["phi0"]) + slope * (g - t) + .5 * curve * (g - t).square()).numpy().astype(np.float32)
            full_phi = d[state + "_full_phi"]
            checked(checks, "public_SiLU_primal", F.silu(g).numpy(), full_phi, 1e-5)
            for arm in ("full", "quadratic", "mask", "random0", "random1", "random2"):
                label = state + "_" + arm
                installed = full_phi if arm == "full" else quadratic if arm == "quadratic" else np.where(masks[arm], full_phi, quadratic)
                checked(checks, "installed_activation", installed, d[label + "_phi"], 0.)
                # Same public FP32 product, separate NumPy down projection/reduction.
                product = (installed * d[state + "_up"]).astype(np.float32)
                single = d[state + "_input"] + product @ w["down"].T
                checked(checks, "independent_Float32_local", single, d[label + "_local"])
                double = d[state + "_input"].astype(float) + product.astype(float) @ w["down"].astype(float).T
                error = float(np.max(abs(double - d[label + "_local"])))
                raw.append(dict(key=key, state=state, arm=arm, error=error, bound=1e-4, passed=error <= 1e-4))
                value = d[label + "_final"][-1].astype(float)
                z = value / np.sqrt(np.mean(value**2) + w["eps"]) * w["norm"]
                checked(checks, "finite_final_head", z @ w["head"].astype(float).T, d[label + "_scores"], .005)
        target = (d["native_full_local"].astype(float) - d["native_quadratic_local"]
            - d["own_full_local"].astype(float) + d["own_quadratic_local"])
        restored = (d["native_mask_local"].astype(float) - d["native_quadratic_local"]
            - d["own_mask_local"].astype(float) + d["own_quadratic_local"])
        for scope, pick in (("all", slice(None)), ("verdict", -1)):
            rel, cos = geometry(restored[pick], target[pick])
            local.append(dict(key=key, **group, scope=scope, target_norm=float(np.linalg.norm(target[pick])),
                relative_error=rel, cosine=cos, passes=bool(np.linalg.norm(target[pick]) >= .001 and rel <= .25 and cos >= .95)))
        prior = dict(np.load(previous / "cases" / path.name))
        actual = prior["n31_target_native"] - prior["n31_target_own"]
        pred = sum(prior["n31_native_finite_" + c] - prior["n31_own_finite_" + c] for c in ("norm", "silu", "product")) - prior["n31_cast"]
        # Public activation anchors introduce explicit small rounding differences.
        checked(checks, "239_missing_tail_public_bridge", actual - pred, target)
        for scope, pick in (("all", slice(None)), ("verdict", -1)):
            r, c = geometry(pred[pick] + restored[pick], actual[pick])
            masks_rows.append(dict(key=key, **group, scope=scope, selected=int(p["mask"].sum()),
                total=int(p["mask"].size), row_counts=json.dumps(p["mask"].sum(-1).tolist()),
                original239_relative=float(np.linalg.norm((actual - pred)[pick]) / np.linalg.norm(actual[pick])),
                repaired239_relative=r, repaired239_cosine=c))
        observation = dict(np.load(old / "observations" / path.name))
        false = dict(np.load(old / "forecasts" / path.name))
        for state in ("native", "own"):
            reference, scores = d[state + "_full_final"].astype(float), d[state + "_full_scores"].astype(float)
            gaps = {}
            for arm in ("quadratic", "mask", "random0", "random1", "random2"):
                value = d[state + "_" + arm + "_final"].astype(float)
                score = d[state + "_" + arm + "_scores"].astype(float)
                gaps[arm] = dict(whole=float(np.linalg.norm(value - reference)), vector=float(np.linalg.norm(value[-1] - reference[-1])),
                    margin=abs(float(np.diff(score)[0] - np.diff(scores)[0])))
            base, repair = gaps["quadratic"], gaps["mask"]
            eligible = base["margin"] >= .10 and base["vector"] >= .001
            passes = eligible and all(repair[k] <= .5 * base[k] and all(repair[k] < gaps["random" + str(j)][k] for j in range(3)) for k in ("margin", "vector"))
            final.append(dict(key=key, **group, state=state, eligible=eligible, passes=bool(passes),
                **{a + "_" + k: v for a, values in gaps.items() for k, v in values.items()}))
            true_delta = observation["true_final"].astype(float) - false["false_final"].astype(float)
            for arm in ("full", "quadratic", "mask"):
                value = d[state + "_" + arm + "_final"].astype(float)
                score = d[state + "_" + arm + "_scores"].astype(float)
                rel, cos = geometry(value - false["false_final"], true_delta)
                global_rows.append(dict(key=key, **group, state=state, arm=arm, whole_relative=rel, whole_cosine=cos,
                    verdict_error=float(np.linalg.norm(value[-1] - observation["true_final"][-1])),
                    margin_error=abs(float(np.diff(score)[0] - np.diff(observation["true_scores"].astype(float))[0]))))
        print("primitive audit", key[:8], flush=True)
    for name, rows in (("local", local), ("final", final), ("native-errors", global_rows), ("masks", masks_rows)):
        with (run / (name + ".csv")).open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(status="passed" if all(r["passed"] for r in raw) else "passed_with_retained_Float64_bridge_failures",
        retrospective=True, source_hashes=design["source_hashes"], checks=checks, raw_float64_bridges=raw,
        local={scope: rule([r for r in local if r["scope"] == scope], "passes") for scope in ("all", "verdict")},
        final={state: dict(**rule([r for r in final if r["state"] == state], "passes"),
            eligible=sum(r["eligible"] for r in final if r["state"] == state)) for state in ("native", "own")},
        local_medians={scope: {k: float(np.median([r[k] for r in local if r["scope"] == scope])) for k in ("relative_error", "cosine")}
            for scope in ("all", "verdict")},
        final_medians={state: {k: float(np.median([r[k] for r in final if r["state"] == state])) for k in final[0]
            if k.startswith(("quadratic_", "mask_", "random0_", "random1_", "random2_"))} for state in ("native", "own")})
    write_json(run / "audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("checks", "raw_float64_bridges", "source_hashes")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve())
    except Exception as exc:
        write_json(args.run / "failure-audit.json", dict(error=repr(exc)))
        raise
