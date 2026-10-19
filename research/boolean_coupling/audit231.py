"""Pre-observation NumPy qualification/seal and frozen local forecast scoring."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .audit229 import build, checked, derivative, sha
from .audit230 import corner

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ("L", "N", "NV", "R")


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def forecast(run):
    assert not (run/"forecast-seal.json").exists() and not (run/"forecast-audit.json").exists()
    design = json.loads((run/"design.json").read_text())
    manifest = json.loads((run/"forecast-manifest.json").read_text())
    assert manifest["counts"] == design["expected_forecast"]
    assert manifest["ordinary_true_before_seal"] == manifest["native_both_attention2_before_seal"] == 0
    assert not any((run/"observations").iterdir())
    for name, digest_value in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == digest_value
    weights = {k:v.astype(float) for k, v in dict(np.load(run/"weights.npz")).items()}
    cases = json.loads((HERE/"cases231.json").read_text())
    catalog = set(json.loads((HERE/"historical-input-hashes-231.json").read_text())["hashes"])
    fresh, checks = set(), []
    assert len(cases) == len(manifest["entries"]) == 16
    derivative_count = 0
    for case, entry in zip(cases, manifest["entries"]):
        assert case["key"] == entry["key"]
        for name in ("false", "true"):
            h = digest(case[name+"_ids"])
            assert h not in catalog and h not in fresh
            fresh.add(h)
        assert len(case["false_ids"]) == len(case["true_ids"])
        changed = [i for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"])) if a != b]
        assert changed == [case["position"]]
        assert case["false_ids"][changed[0]] == 895 and case["true_ids"][changed[0]] == 830
        path = run/"forecasts"/(case["key"]+".npz")
        record = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == record["arrays_sha256"] == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["record_sha256"]
        assert all(c["error"] <= c["bound"] for c in record["checks"])
        data = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        start = record["start"]
        original = build(weights, data, data["x"])
        checked(checks, "false_primal", (data["x"]+original["write"])[start:], data["false_mid"])
        source = build(weights, data, data["source"])
        checked(checks, "source_only_primal", (data["source"]+source["write"])[start:], data["source_mid"])
        direction = data["receiver"]-data["x"]
        tangent = derivative(weights, data, data["x"], direction, original)[start:]
        checked(checks, "numpy_Jd", tangent, data["tangent"], 1e-9)
        checked(checks, "used_FD", (data["plus"]-data["minus"])/2e-5, tangent)
        values = {b:corner(weights, data, original, b)[start:] for b in ("0000", "1000", "1001")}
        n, nv = values["1000"]-values["0000"], values["1001"]-values["0000"]
        checked(checks, "N_correction", n, data["N_correction"], 1e-9)
        checked(checks, "NV_correction", nv, data["NV_correction"], 1e-9)
        seed = int(hashlib.sha256((case["key"]+"231").encode()).hexdigest()[:16], 16)
        random = np.random.default_rng(seed).standard_normal(n.shape)
        random *= np.linalg.norm(nv-n)/np.linalg.norm(random)
        checked(checks, "random_control", random, data["random"], 1e-9)
        base = data["source_mid"]+tangent
        expected = dict(L=base, N=base+n, NV=base+nv, R=base+n+random)
        for arm, value in expected.items():
            checked(checks, "prediction_"+arm, value, data[arm+"_prediction"], 1e-9)
        checked(checks, "known_first_prefix", data["known_first"][:start-1], data["x"][:start-1], 0.)
        for d in record["derivatives"]:
            assert d["ad_error"] <= 1e-9 and d["fd_error"] <= 1e-4
            key = d["witness_key"]
            checked(checks, "affine_node_FD", data[key+"_tangent"], (data[key+"_plus"]-data[key+"_minus"])/2e-5)
        derivative_count += len(record["derivatives"])
    assert derivative_count == manifest["local_affine_derivatives"]
    audit = dict(status="passed", model_calls=0, targets_observed=0, checks=checks,
        local_affine_derivatives=derivative_count, fresh_hashes=len(fresh), design_sha256=sha(run/"design.json"))
    (run/"forecast-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    paths = [run/n for n in ("design.json", "runtime.json", "weights.npz", "forecast-manifest.json", "forecast-status.json", "forecast-audit.json")]
    paths += list((run/"forecasts").iterdir())+list((run/"sources").iterdir())
    seal = dict(status="qualified", sealed_monotonic_ns=time.monotonic_ns(),
        files={str(p.relative_to(run)):sha(p) for p in sorted(paths)})
    (run/"forecast-seal.json").write_text(json.dumps(seal, indent=2)+"\n")
    print(f"Qualified and globally sealed 64 local forecasts; {len(checks)} checks, zero targets observed.")


def score(run):
    assert not (run/"audit.json").exists()
    seal = json.loads((run/"forecast-seal.json").read_text())
    assert all(sha(run/p) == h for p, h in seal["files"].items())
    begin = json.loads((run/"observation-start.json").read_text())
    assert begin["started_monotonic_ns"] > seal["sealed_monotonic_ns"]
    assert begin["seal_sha256"] == sha(run/"forecast-seal.json")
    cases, checks = [], []
    statuses = json.loads((run/"status.json").read_text())
    assert statuses["status"] == "complete" and statuses["ordinary_true"] == 16
    assert len(statuses["observations"]) == 16
    for name, h in json.loads((run/"design.json").read_text())["source_hashes"].items():
        assert sha(ROOT/name) == h
    for case, entry in zip(json.loads((HERE/"cases231.json").read_text()), statuses["observations"]):
        assert case["key"] == entry["key"]
        fp = run/"forecasts"/(case["key"]+".npz")
        op = run/"observations"/(case["key"]+".npz")
        meta = json.loads(op.with_suffix(".json").read_text())
        assert sha(op) == meta["arrays_sha256"] == entry["arrays_sha256"]
        assert sha(op.with_suffix(".json")) == entry["record_sha256"]
        assert sha(fp) == meta["forecast_sha256"] and meta["seal_sha256"] == sha(run/"forecast-seal.json")
        assert meta["observed_monotonic_ns"] > begin["started_monotonic_ns"]
        assert all(c["error"] <= c["bound"] for c in meta["checks"])
        f, obs = dict(np.load(fp)), dict(np.load(op))
        checked(checks, "known_first_native_replay", obs["native_first"], f["known_first"], 0.)
        target = obs["native_mid"].astype(float)-f["false_mid"].astype(float)
        row = dict(key=case["key"], **case["group"])
        for scope in ("verdict", "all"):
            t = target[-1] if scope == "verdict" else target
            den = float(np.linalg.norm(t))
            row[scope+"_native_norm"] = den
            for arm in ARMS:
                value = f[arm+"_prediction"]-f["false_mid"].astype(float)
                value = value[-1] if scope == "verdict" else value
                rel = float(np.linalg.norm(value-t)/den)
                cos = float(np.vdot(value, t)/(np.linalg.norm(value)*den))
                row[scope+"_"+arm+"_relative"] = rel
                row[scope+"_"+arm+"_cosine"] = cos
            row[scope+"_NV_pass"] = bool(den >= .001 and row[scope+"_NV_relative"] <= .025
                and row[scope+"_NV_cosine"] >= .995
                and all(row[scope+"_NV_relative"] < row[scope+"_"+arm+"_relative"] for arm in ("L", "N", "R")))
        cases.append(row)
    with (run/"cases.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    strata = {f"{s}_{o}":{scope:sum(r[scope+"_NV_pass"] for r in cases if r["schema"] == s and r["reverse"] == o)
        for scope in ("verdict", "all")} for s in ("generic", "descriptive") for o in (0, 1)}
    passes = {scope:sum(r[scope+"_NV_pass"] for r in cases) for scope in ("verdict", "all")}
    audit = dict(status="passed", experiment=231, scope="Prospective local attention2 prediction only.", checks=checks,
        cases=16, ordinary_true=16, passes=passes, strata=strata,
        whole={scope:passes[scope] >= 12 and all(s[scope] >= 3 for s in strata.values()) for scope in ("verdict", "all")},
        medians={k:float(np.median([r[k] for r in cases])) for k in cases[0]
                 if k not in ("key", "block", "schema", "reverse", "assignment", "target")},
        forecast_seal_sha256=sha(run/"forecast-seal.json"), source_hashes=json.loads((run/"design.json").read_text())["source_hashes"])
    (run/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "score"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    {"forecast":forecast, "score":score}[args.phase](args.out.resolve())
