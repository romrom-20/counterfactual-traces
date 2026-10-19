"""Qualified retrospective four-primitive factorial for attention receiver curvature."""

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from .alignment228 import stamp
from .attention229 import graph
from .core225b import HERE, array, check, load_model, sha, write_json
from .curvature230 import BITS, corner, norm_direction, normalized_paths
from .smooth228 import norm, promote
from .tail226b import stored

ROOT = HERE.parents[1]
SOURCES = [HERE/n for n in ("run230.py", "curvature230.py", "audit230.py", "attention229.py",
    "core225b.py", "smooth228.py", "audit229.py", "cases228.json", "settings.json")]
SOURCES += [ROOT/"docs/experiments/230-receiver-curvature-primitives.md"]


def main(args):
    out, previous = args.out.resolve(), args.input.resolve()
    assert not out.exists(), "Never overwrite evidence."
    assert json.loads((previous/"audit.json").read_text())["status"] == "passed"
    cases = json.loads((HERE/"cases228.json").read_text())
    out.mkdir(parents=True)
    (out/"cases").mkdir()
    (out/"sources").mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    inputs = [previous/n for n in ("design.json", "status.json", "audit.json", "weights.npz")]
    inputs += [previous/"cases"/(c["key"]+suffix) for c in cases for suffix in (".json", ".npz")]
    write_json(out/"design.json", dict(experiment=230, retrospective=True, created_utc=stamp(),
        source_hashes=sources, input_root=str(previous),
        inputs={str(p.relative_to(previous)):sha(p) for p in inputs},
        bits="N,K,S,V; finite=1, False-affine=0", thresholds=dict(relative=.25, cosine=.95, guard=.001,
            total=24, each_stratum=6, coordinate=1e-4, identity=1e-9, fd=1e-4),
        expected=dict(ordinary_models=0, public_attention=64, false_public_graphs=32,
            false_smooth_graphs=32, public_intervention_graphs=608, smooth_intervention_graphs=608)))
    model = load_model(args.offline)
    layer = model.model.layers[1]
    write_json(out/"runtime.json", dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        smooth="cpu_float64", revision=model.config._commit_hash))
    weights = dict(np.load(previous/"weights.npz"))
    for name in ("q", "k", "v", "o"):
        module = getattr(layer.self_attn, name+"_proj")
        assert np.array_equal(array(module.weight), weights[name+"_weight"])
        if module.bias is not None:
            assert np.array_equal(array(module.bias), weights[name+"_bias"])
    assert np.array_equal(array(layer.input_layernorm.weight), weights["attn_norm"])
    assert layer.input_layernorm.variance_epsilon == float(weights["attn_eps"])
    assert layer.self_attn.scaling == float(weights["scaling"])
    assert layer.self_attn.head_dim == int(weights["width"])
    assert layer.self_attn.num_key_value_groups == int(weights["groups"])
    entries, all_derivatives = [], 0
    counts = Counter({k:0 for k in json.loads((out/"design.json").read_text())["expected"]})
    for number, case in enumerate(cases, 1):
        old = stored(previous/"cases"/(case["key"]+".npz"))
        meta = json.loads((previous/"cases"/(case["key"]+".json")).read_text())
        start = meta["start"]
        context = dict(mask=torch.tensor(old["mask"], device="mps"),
            rope=tuple(torch.tensor(old[name], device="mps") for name in ("rope_cos", "rope_sin")))
        false = torch.tensor(old["x"][None], device="mps")
        receiving = torch.tensor(old["receiver"][None], device="mps")
        checks, derivatives, cache, data = [], [], {"start":start}, {}
        with torch.no_grad():
            anchors = {kind:graph(layer, false, context, 0, kind == "smooth") for kind in ("public", "smooth")}
            counts["false_public_graphs"] += 1
            counts["false_smooth_graphs"] += 1
            for name, operand in (("zero", false), ("receiver", receiving)):
                write = layer.self_attn(layer.input_layernorm(operand), attention_mask=context["mask"],
                    position_embeddings=context["rope"])[0]
                counts["public_attention"] += 1
                check(checks, "public_"+name+"_replay", np.max(abs(array((operand+write)[0, start:])-old[name+"_mid"])))
            for kind in ("public", "smooth"):
                double = kind == "smooth"
                anchor = anchors[kind]
                reference = (promote(false) if double else false)+anchor["write"]
                for bits in BITS:
                    value, _ = corner(layer, false, receiving, context, start, anchor, bits, double, derivatives, cache)
                    counts[kind+"_intervention_graphs"] += 1
                    check(checks, kind+"_prefix", (value[:, :start]-reference[:, :start]).abs().max(), 0.)
                    data[kind+"_"+bits] = array(value[0, start:])
                groups = dict(first=[start], middle=list(range(start+1, false.shape[1]-1)), verdict=[false.shape[1]-1])
                for name, rows in groups.items():
                    value, _ = corner(layer, false, receiving, context, start, anchor, "1000", double,
                                      derivatives, cache, norm_rows=rows)
                    counts[kind+"_intervention_graphs"] += 1
                    data[kind+"_N_"+name] = array(value[0, start:])
                    check(checks, kind+"_row_prefix", (value[:, :start]-reference[:, :start]).abs().max(), 0.)
                data[kind+"_false"] = array(reference[0, start:])
                check(checks, kind+"_finite_replay", np.max(abs(data[kind+"_1111"]-old["receiver_mid"])))
            for suffix in (*BITS, "N_first", "N_middle", "N_verdict"):
                check(checks, "smooth_public_endpoint", np.max(abs(data["smooth_"+suffix]-data["public_"+suffix].astype(float))))
                check(checks, "smooth_public_increment", np.max(abs((data["smooth_"+suffix]-data["smooth_0000"])
                      -(data["public_"+suffix].astype(float)-data["public_0000"].astype(float)))))
            check(checks, "smooth_affine_Jd", np.max(abs(data["smooth_0000"]-data["smooth_false"]-old["used_tangent"])), 1e-9)
            check(checks, "public_affine_Jd", np.max(abs(data["public_0000"].astype(float)-old["zero_mid"].astype(float)-old["used_tangent"])))
            data["old_R"] = old["R"]
            data["rounding_R"] = old["R"]-(data["smooth_1111"]-data["smooth_0000"])
            check(checks, "native_R_rounding", np.max(abs(data["rounding_R"])))
            x, d = promote(false), promote(receiving)-promote(false)
            z0 = anchors["smooth"]["norm"]
            dz = norm_direction(x, d, promote(layer.input_layernorm.weight), layer.input_layernorm.variance_epsilon)
            eta = norm(promote(receiving), promote(layer.input_layernorm.weight), layer.input_layernorm.variance_epsilon)-z0-dz
            data.update(norm_curvature=array(eta[0, start:]), input_delta=array(d[0, start:]), false_input=array(x[0, start:]))
            first = torch.zeros_like(eta)
            first[:, start] = eta[:, start]
            paths = normalized_paths(layer, anchors["smooth"], first, context)
            for name, value in paths.items():
                data["first_"+name+"_path"] = array(value[0, start:])
            check(checks, "first_norm_paths", np.max(abs(sum(data["first_"+k+"_path"] for k in paths)
                  -(data["smooth_N_first"]-data["smooth_0000"]))), 1e-9)
            check(checks, "verdict_first_query_zero", np.max(abs(data["first_query_path"][-1])), 0.)
            check(checks, "row_norm_partition", np.max(abs(sum(data["smooth_N_"+k]-data["smooth_0000"] for k in groups)
                  -(data["smooth_1000"]-data["smooth_0000"]))), 1e-9)
        for key, witness in cache.items():
            if key != "start":
                for name, value in witness.items():
                    data[key+"_"+name] = array(value)
        all_derivatives += len(derivatives)
        path = out/"cases"/(case["key"]+".npz")
        np.savez_compressed(path, **data)
        record = dict(key=case["key"], group=case["group"], start=start, checks=checks,
            derivatives=derivatives, arrays_sha256=sha(path), recorded_utc=stamp())
        write_json(path.with_suffix(".json"), record)
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        print(f"case {number}/32 complete; {len(derivatives)} distinct affine directions", flush=True)
    assert all(sha(ROOT/p) == h for p, h in sources.items())
    assert dict(counts) == json.loads((out/"design.json").read_text())["expected"], counts
    write_json(out/"status.json", dict(status="complete", completed_utc=stamp(), cases=entries,
        smooth_derivatives=all_derivatives, fd_endpoints=2*all_derivatives,
        counts=dict(counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    main(parser.parse_args())
