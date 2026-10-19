"""Separate NumPy and Torch primitive qualification, seal and fixed source-K/V scoring."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit229 import build, checked, derivative, sha
from .audit230 import corner
from .audit233 import functions, tensor
from .transport233 import operator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ("FNV", "K", "V", "A", "C", "R0", "R1", "R2")


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
    cases = json.loads((HERE/"cases234.json").read_text())
    catalog = set(json.loads((HERE/"historical-input-hashes-234.json").read_text())["hashes"])
    fresh, checks = set(), []
    assert len(cases) == len(manifest["entries"]) == 32
    derivative_count = 0
    torch.set_num_threads(1)
    (run/"qualification").mkdir()
    qualification = []
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
        values, tangents, primals = {}, {}, {}
        frames = dict(F=data)
        for label, frame in frames.items():
            anchor = original if label == "F" else source
            direction = frame["receiver"]-frame["x"]
            tangent = derivative(weights, frame, frame["x"], direction, anchor)[start:]
            checked(checks, label+"_numpy_Jd", tangent, data[label+"_tangent"], 1e-9)
            checked(checks, label+"_used_FD", (data[label+"_plus"]-data[label+"_minus"])/2e-5, tangent)
            primals[label] = (frame["x"]+anchor["write"])[start:]
            checked(checks, label+"_smooth_primal", primals[label], data[label+"_primal"], 1e-9)
            values[label] = {bits:corner(weights, frame, anchor, bits)[start:] for bits in ("0000", "1001")}
            for bits, value in values[label].items():
                checked(checks, label+"_corner_"+bits, value, data[label+"_"+bits], 1e-9)
            tangents[label] = tangent
            for d in record["derivatives"][label]:
                assert d["ad_error"] <= 1e-9 and d["fd_error"] <= 1e-4
                key = label+"_"+d["witness_key"]
                checked(checks, label+"_affine_node_FD", data[key+"_tangent"], (data[key+"_plus"]-data[key+"_minus"])/2e-5)
            derivative_count += len(record["derivatives"][label])
        fnv = data["source_mid"]+tangents["F"]+(values["F"]["1001"]-values["F"]["0000"])
        transport = {bits:operator(weights, data, original, source, bits)[start:] for bits in ("0000", "1000", "0010", "1010")}
        components = dict(key=transport["1000"]-transport["0000"],
            value=transport["0010"]-transport["0000"],
            mixed=transport["1010"]-transport["1000"]-transport["0010"]+transport["0000"])
        for name, component in components.items():
            checked(checks, "numpy_term_"+name, component, data["term_"+name], 1e-9)
        expected = dict(FNV=fnv, K=fnv+components["key"], V=fnv+components["value"],
            A=fnv+components["key"]+components["value"], C=fnv+sum(components.values()))
        increment = expected["C"]-fnv
        for control in range(3):
            seed = int(hashlib.sha256((case["key"]+str(control)+"234").encode()).hexdigest()[:16], 16)
            random = np.random.default_rng(seed).standard_normal(increment.shape)
            random *= np.linalg.norm(increment, axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
            checked(checks, "row_random_control", random, data["random_"+str(control)], 1e-9)
            expected["R"+str(control)] = fnv+random
        for arm, value in expected.items():
            checked(checks, "prediction_"+arm, value, data[arm+"_prediction"], 1e-9)
        make, _, _ = functions(weights, data, start)
        anchor = tensor(data["x"][start:])
        direction = tensor((data["receiver"]-data["x"])[start:])
        witnesses = {}
        for bits in transport:
            hybrid, native = make(bits)
            for name, function in (("hybrid", hybrid), *(([("native", native)]) if bits == "0000" else [])):
                _, tangent = torch.func.jvp(function, (anchor,), (direction,))
                plus, minus = (function(anchor+sign*1e-5*direction).detach().numpy() for sign in (1, -1))
                tangent = tangent.detach().numpy()
                checked(checks, name+"_analytic_ad", tangent, transport[bits], 1e-9)
                checked(checks, name+"_ad_fd", tangent, (plus-minus)/2e-5)
                for suffix, value in (("tangent", tangent), ("plus", plus), ("minus", minus)):
                    witnesses[f"{bits}_{name}_{suffix}"] = value
        witness = run/"qualification"/(case["key"]+".npz")
        np.savez_compressed(witness, **witnesses)
        qualification.append(dict(key=case["key"], arrays_sha256=sha(witness)))
        checked(checks, "known_first_prefix", data["known_first"][:start-1], data["x"][:start-1], 0.)
    assert derivative_count == manifest["local_affine_derivatives"]
    assert len(qualification) == 32
    (run/"qualification-status.json").write_text(json.dumps(dict(status="complete", cases=qualification), indent=2)+"\n")
    audit = dict(status="passed", model_calls=0, targets_observed=0, checks=checks,
        hybrid_ad_fd=128, native_ad_fd=32, torch_fd_endpoints=320,
        local_affine_derivatives=derivative_count, fresh_hashes=len(fresh), design_sha256=sha(run/"design.json"))
    (run/"forecast-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    paths = [run/n for n in ("design.json", "runtime.json", "weights.npz", "forecast-manifest.json", "forecast-status.json", "forecast-audit.json", "qualification-status.json")]
    paths += list((run/"forecasts").iterdir())+list((run/"sources").iterdir())+list((run/"qualification").iterdir())
    seal = dict(status="qualified", sealed_monotonic_ns=time.monotonic_ns(),
        files={str(p.relative_to(run)):sha(p) for p in sorted(paths)})
    (run/"forecast-seal.json").write_text(json.dumps(seal, indent=2)+"\n")
    print(f"Qualified and globally sealed 256 local forecasts; {len(checks)} checks, zero targets observed.")


def score(run):
    assert not (run/"audit.json").exists()
    seal = json.loads((run/"forecast-seal.json").read_text())
    assert all(sha(run/p) == h for p, h in seal["files"].items())
    begin = json.loads((run/"observation-start.json").read_text())
    assert begin["started_monotonic_ns"] > seal["sealed_monotonic_ns"]
    assert begin["seal_sha256"] == sha(run/"forecast-seal.json")
    cases, checks = [], []
    statuses = json.loads((run/"status.json").read_text())
    assert statuses["status"] == "complete" and statuses["ordinary_true"] == 32
    assert len(statuses["observations"]) == 32
    for name, h in json.loads((run/"design.json").read_text())["source_hashes"].items():
        assert sha(ROOT/name) == h
    for case, entry in zip(json.loads((HERE/"cases234.json").read_text()), statuses["observations"]):
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
            for arm in ARMS:
                row[scope+"_"+arm+"_accuracy"] = bool(den >= .001 and row[scope+"_"+arm+"_relative"] <= .025
                    and row[scope+"_"+arm+"_cosine"] >= .995)
            missing = (obs["native_mid"].astype(float)-f["FNV_prediction"])
            mixed = f["term_mixed"] if scope == "all" else f["term_mixed"][-1]
            missing = missing if scope == "all" else missing[-1]
            row[scope+"_mixed_signed_projection"] = float(np.vdot(mixed, missing)/np.vdot(missing, missing))
            row[scope+"_mixed_removal_improves"] = row[scope+"_A_relative"] < row[scope+"_C_relative"]
            controls = all(row[scope+"_C_relative"] < row[scope+"_"+arm+"_relative"] for arm in ("FNV", "K", "V", "A", "R0", "R1", "R2"))
            row[scope+"_C_pass"] = bool(row[scope+"_C_accuracy"] and (controls if scope == "all" else True))
        cases.append(row)
    with (run/"cases.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    strata = {f"{s}_{o}":{scope:sum(r[scope+"_C_pass"] for r in cases if r["schema"] == s and r["reverse"] == o)
        for scope in ("verdict", "all")} for s in ("generic", "descriptive") for o in (0, 1)}
    passes = {scope:sum(r[scope+"_C_pass"] for r in cases) for scope in ("verdict", "all")}
    audit = dict(status="passed", experiment=234, scope="Primary: source K/V mixed update, all receiving rows after attention2; secondary: verdict accuracy preservation.", checks=checks,
        cases=32, ordinary_true=32, passes=passes, strata=strata,
        whole={scope:passes[scope] >= 24 and all(s[scope] >= 6 for s in strata.values()) for scope in ("verdict", "all")},
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
