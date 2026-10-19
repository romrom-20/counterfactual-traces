"""Retrospective four-operand source-tangent interventions, explicit NumPy arithmetic."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .audit229 import build, checked, derivative, sha

ROOT = Path(__file__).resolve().parents[2]
BITS = tuple(f"{i:04b}" for i in range(16))
SOURCES = [ROOT/p for p in (
    "research/boolean_coupling/transport233.py", "research/boolean_coupling/audit233.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/cases232.json",
    "docs/experiments/233-source-transport-primitives.md")]


def projected_direction(weights, data, original):
    x, direction = data["x"], data["receiver"]-data["x"]
    rms = original["rms"]
    dz = weights["attn_norm"]*(direction/rms-x*np.mean(x*direction, -1, keepdims=True)/rms**3)
    width, groups = int(weights["width"]), int(weights["groups"])

    def project(name):
        value = (dz@weights[name+"_weight"].T).reshape(len(x), -1, width).transpose(1, 0, 2)
        if name != "v":
            a, b = np.split(value, 2, -1)
            value = value*data["rope_cos"][0]+np.concatenate((-b, a), -1)*data["rope_sin"][0]
        return np.repeat(value, groups, 0) if name != "q" else value

    return direction, *(project(n) for n in ("q", "k", "v"))


def operator(weights, data, original, source, bits):
    direction, dq, dk, dv = projected_direction(weights, data, original)
    k, p, v, pv = (source[name] if bit == "1" else original[name]
                    for name, bit in zip(("k", "p", "v", "p"), bits))
    dl = (dq@k.transpose(0, 2, 1)+original["q"]@dk.transpose(0, 2, 1))*weights["scaling"]
    dp = p*(dl-np.sum(p*dl, -1, keepdims=True))
    context = dp@v+pv@dv
    return direction+context.transpose(1, 0, 2).reshape(direction.shape)@weights["o_weight"].T


def factorial(values):
    result = {}
    for number in range(1, 16):
        term = np.zeros_like(values["0000"])
        for subset in range(16):
            if subset & ~number == 0:
                term += (-1)**(number.bit_count()-subset.bit_count())*values[f"{subset:04b}"]
        result[f"{number:04b}"] = term
    return result


def random_rows(key, bits, value, index):
    seed = int(hashlib.sha256((key+bits+str(index)+"233").encode()).hexdigest()[:16], 16)
    control = np.random.default_rng(seed).standard_normal(value.shape)
    return control*np.linalg.norm(value, axis=-1, keepdims=True)/np.linalg.norm(control, axis=-1, keepdims=True)


def main(previous, out):
    assert not out.exists(), "Never overwrite evidence."
    out.mkdir(parents=True)
    (out/"sources").mkdir()
    (out/"cases").mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    manifest = json.loads((previous/"forecast-manifest.json").read_text())
    status = json.loads((previous/"status.json").read_text())
    assert len(status["observations"]) == len(manifest["entries"]) == 32
    inputs = {str(p.relative_to(previous)):sha(p) for p in sorted(previous.rglob("*.json"))}
    inputs["weights.npz"] = sha(previous/"weights.npz")
    design = dict(experiment=233, retrospective=True, source_hashes=sources, inputs=inputs,
        input_root=str(previous), expected=dict(numpy_anchors=64, operators=512, factorial_terms=480,
            hybrid_ad_fd=512, native_ad_fd=64, fd_endpoints=1152, ordinary_models=0),
        numerical_bounds=dict(analytic_ad=1e-9, central_fd=1e-4, public_primal=1e-4),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, device="cpu", dtype="float64"))
    (out/"design.json").write_text(json.dumps(design, indent=2)+"\n")
    weights = {k:v.astype(float) for k, v in dict(np.load(previous/"weights.npz")).items()}
    entries = []
    for index, (f, o) in enumerate(zip(manifest["entries"], status["observations"]), 1):
        key = f["key"]
        assert key == o["key"]
        fp, op = (previous/phase/(key+".npz") for phase in ("forecasts", "observations"))
        assert sha(fp) == f["arrays_sha256"] and sha(op) == o["arrays_sha256"]
        data = {k:v.astype(float) for k, v in dict(np.load(fp)).items()}
        obs = dict(np.load(op))
        meta = json.loads(fp.with_suffix(".json").read_text())
        checks, start = [], meta["start"]
        original, source = (build(weights, data, data[name]) for name in ("x", "source"))
        checked(checks, "public_false_anchor", (data["x"]+original["write"])[start:], data["false_mid"])
        checked(checks, "public_source_anchor", (data["source"]+source["write"])[start:], data["source_mid"])
        checked(checks, "protected_prefix", data["source"][:start-1], data["x"][:start-1], 0.)
        checked(checks, "direction_prefix", data["receiver"][:start], data["x"][:start], 0.)
        checked(checks, "receiving_state_unchanged", data["source"][start:], data["x"][start:], 0.)
        checked(checks, "receiving_direction_unchanged", (data["known_first"]-data["source"])[start:],
                (data["receiver"]-data["x"])[start:], 0.)
        values = {bits:operator(weights, data, original, source, bits)[start:] for bits in BITS}
        for bits, label in (("0000", "F"), ("1111", "S")):
            frame = data if label == "F" else {**data, "x":data["source"], "receiver":data["known_first"]}
            anchor = original if label == "F" else source
            reference = derivative(weights, frame, frame["x"], frame["receiver"]-frame["x"], anchor)[start:]
            checked(checks, label+"_full_derivative", values[bits], reference, 1e-9)
            checked(checks, label+"_saved_derivative", values[bits], data[label+"_tangent"], 1e-9)
        terms = factorial(values)
        checked(checks, "factorial_closure", sum(terms.values()), values["1111"]-values["0000"], 1e-9)
        array_data = {"T_"+bits:value for bits, value in values.items()}
        array_data.update({"M_"+bits:value for bits, value in terms.items()})
        array_data.update(false_mid=data["false_mid"], baseline=data["FNV_prediction"],
            native_mid=obs["native_mid"], original_SNV=data["SNV_prediction"])
        for bits in BITS[1:]:
            change = values[bits]-values["0000"]
            for control in range(3):
                array_data[f"random_{bits}_{control}"] = random_rows(key, bits, change, control)
        path = out/"cases"/(key+".npz")
        np.savez_compressed(path, **array_data)
        record = dict(key=key, group=meta["group"], start=start, arrays_sha256=sha(path),
            input_forecast_sha256=sha(fp), input_observation_sha256=sha(op), checks=checks)
        path.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        print("diagnostic", index, "/32; existing targets only", flush=True)
    (out/"status.json").write_text(json.dumps(dict(status="complete", cases=entries,
        ordinary_models=0, source_hashes=sources), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-232", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_232.resolve(), args.out.resolve())
