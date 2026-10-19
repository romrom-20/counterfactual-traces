"""Globally sealed source-key/value local forecasts with direct mixed-product control."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from .alignment228 import mids, stamp
from .attention229 import graph, output
from .core225b import HERE, array, check, load_model, observe, sha, write_json
from .curvature230 import corner, norm_direction
from .smooth228 import attention_map, norm, promote, qualified_jvp, rotate
from .tail226b import block

ROOT = HERE.parents[1]
ARMS = ("FNV", "K", "V", "A", "C", "R0", "R1", "R2")
SOURCES = [HERE/n for n in ("local234.py", "audit234.py", "fixtures234.py", "cases234.json",
    "historical-input-hashes-234.json", "curvature230.py", "attention229.py", "smooth228.py", "audit233.py", "transport233.py",
    "core225b.py", "alignment228.py", "tail226b.py", "audit230.py", "audit229.py", "settings.json")]
SOURCES += [ROOT/"docs/experiments/234-source-key-value-forecast.md"]
EXPECTED = dict(ordinary_false=32, ordinary_true=0, edited_first_blocks=32,
    source_only_attention=32, smooth_anchor_graphs=32, smooth_intervention_graphs=64,
    receiving_jvps=32, receiving_fd_endpoints=64, source_row_kv=32)


def source_terms(layer, x, known, context, start, anchor):
    """False-background key/value/mixed transmission; no source probability derivative."""
    attn = layer.self_attn
    weight, eps = promote(layer.input_layernorm.weight), layer.input_layernorm.variance_epsilon
    direction = torch.zeros_like(promote(x))
    direction[:, start:] = promote(known[:, start:])-promote(x[:, start:])
    dz = norm_direction(promote(x), direction, weight, eps)
    ds = norm(promote(known[:, start-1:start]), weight, eps)-anchor["norm"][:, start-1:start]
    cosine, sine = (promote(v).unsqueeze(1) for v in context["rope"])

    def project(z, name, first=0):
        value = torch.nn.functional.linear(z, promote(getattr(attn, name+"_proj").weight))
        value = value.view(*value.shape[:-1], -1, attn.head_dim).transpose(1, 2)
        if name != "v":
            value = rotate(value, cosine[:, :, first:first+z.shape[1]], sine[:, :, first:first+z.shape[1]])
        return value.repeat_interleave(attn.num_key_value_groups, 1) if name != "q" else value

    dq, dk = (project(dz, name) for name in ("q", "k"))
    dkey, dvalue = torch.zeros_like(anchor["key"]), torch.zeros_like(anchor["value"])
    dkey[:, :, start-1:start] = project(ds, "k", start-1)
    dvalue[:, :, start-1:start] = project(ds, "v", start-1)
    p, v = anchor["probabilities"], anchor["value"]
    dl = (dq@anchor["key"].transpose(-1, -2)+anchor["query"]@dk.transpose(-1, -2))*attn.scaling
    b = dq@dkey.transpose(-1, -2)*attn.scaling

    def response(logits):
        return p*(logits-(p*logits).sum(-1, keepdim=True))

    contexts = dict(key=response(b)@v, value=response(dl)@dvalue, mixed=response(b)@dvalue)
    return {name:output(layer, value, True)[:, start:] for name, value in contexts.items()}


def save(folder, case, data, extra):
    path = folder/(case["key"]+".npz")
    assert not path.exists()
    np.savez_compressed(path, **data)
    record = dict(key=case["key"], group=case["group"], arrays_sha256=sha(path), recorded_utc=stamp(), **extra)
    write_json(path.with_suffix(".json"), record)
    return dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")))


def forecast(args):
    out = args.out
    assert not out.exists(), "Never overwrite evidence."
    out.mkdir(parents=True)
    for name in ("forecasts", "observations", "sources"):
        (out/name).mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"design.json", dict(experiment=234, prospective_local_only=True, created_utc=stamp(),
        source_hashes=sources, expected_forecast=EXPECTED, thresholds=dict(relative=.025, cosine=.995,
            guard=.001, total=24, each_stratum=6), primary="Source K/V correction, all receiving rows after attention 2",
        known_edit_budget="One exact edited block 1 and one public source-only attention 2 per case; "
                          "False control graphs and derivatives counted separately; all transport uses False H and P."))
    model = load_model(args.offline)
    layer = model.model.layers[1]
    write_json(out/"runtime.json", dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        smooth="cpu_float64", revision=model.config._commit_hash))
    weights = dict(attn_norm=array(layer.input_layernorm.weight), attn_eps=np.array(layer.input_layernorm.variance_epsilon),
        scaling=np.array(layer.self_attn.scaling), groups=np.array(layer.self_attn.num_key_value_groups), width=np.array(layer.self_attn.head_dim))
    for name in ("q", "k", "v", "o"):
        module = getattr(layer.self_attn, name+"_proj")
        weights[name+"_weight"] = array(module.weight)
        if module.bias is not None:
            weights[name+"_bias"] = array(module.bias)
    np.savez_compressed(out/"weights.npz", **weights)
    counts, entries, derivative_count = Counter({k:0 for k in EXPECTED}), [], 0
    for number, case in enumerate(json.loads((HERE/"cases234.json").read_text()), 1):
        start = case["position"]+1
        with mids(model) as middle:
            states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        counts["ordinary_false"] += 1
        checks, derivative_records, caches, data = [], {}, {}, {}
        with torch.no_grad():
            embedding = model.model.embed_tokens(torch.tensor([case["true_ids"]], device="mps"))
            known = block(model, 0, embedding, context)
            counts["edited_first_blocks"] += 1
            x = states[0]
            check(checks, "known_first_prefix", (known[:, :start-1]-x[:, :start-1]).abs().max(), 0.)
            source, receiver = x.clone(), x.clone()
            source[:, start-1] = known[:, start-1]
            receiver[:, start:] = known[:, start:]
            check(checks, "source_known_prefix", (source[:, :start]-known[:, :start]).abs().max(), 0.)
            source_write = layer.self_attn(layer.input_layernorm(source), attention_mask=context["mask"],
                position_embeddings=context["rope"])[0]
            source_mid = source+source_write
            counts["source_only_attention"] += 1
            values, tangents, primals = {}, {}, {}
            for label, anchor_input, parent in (("F", x, receiver),):
                # Separate caches: a direction hash does not encode the different parents.
                cache, derivatives = {"start":start}, []
                caches[label], derivative_records[label] = cache, derivatives
                check(checks, label+"_direction_prefix", (parent[:, :start]-anchor_input[:, :start]).abs().max(), 0.)
                anchor = graph(layer, anchor_input, context, 0, True)
                counts["smooth_anchor_graphs"] += 1
                primal_mid = promote(anchor_input[:, start:])+anchor["write"][:, start:]
                target_primal = middle[1] if label == "F" else source_mid
                check(checks, label+"_smooth_primal", (primal_mid-promote(target_primal[:, start:])).abs().max())
                values[label] = {}
                for bits in ("0000", "1001"):
                    value, _ = corner(layer, anchor_input, parent, context, start, anchor, bits, True, derivatives, cache)
                    values[label][bits] = value[:, start:]
                    data[label+"_"+bits] = array(value[0, start:])
                    counts["smooth_intervention_graphs"] += 1
                function = attention_map(layer, anchor_input, context, start)
                primal, tangent, plus, minus, error = qualified_jvp(function, promote(anchor_input[:, start:]),
                    promote(parent[:, start:])-promote(anchor_input[:, start:]))
                counts["receiving_jvps"] += 1
                counts["receiving_fd_endpoints"] += 2
                check(checks, label+"_receiving_fd", error)
                check(checks, label+"_all_affine_Jd", (values[label]["0000"]-primal-tangent).abs().max(), 1e-9)
                tangents[label], primals[label] = tangent, primal_mid
                for name, value in (("tangent", tangent), ("plus", plus), ("minus", minus), ("primal", primal_mid)):
                    data[label+"_"+name] = array(value[0])
            actual_source = promote(source_mid[:, start:])
            fnv = actual_source+tangents["F"]+(values["F"]["1001"]-values["F"]["0000"])
            terms = source_terms(layer, x, known, context, start, anchor)
            counts["source_row_kv"] += 1
            predictions = dict(FNV=fnv, K=fnv+terms["key"], V=fnv+terms["value"],
                A=fnv+terms["key"]+terms["value"], C=fnv+sum(terms.values()))
            increment = array((predictions["C"]-fnv)[0])
            random_vectors = {}
            for control in range(3):
                seed = int(hashlib.sha256((case["key"]+str(control)+"234").encode()).hexdigest()[:16], 16)
                random = np.random.default_rng(seed).standard_normal(increment.shape)
                random *= np.linalg.norm(increment, axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
                check(checks, "row_random_norm", np.max(abs(np.linalg.norm(random, axis=-1)-np.linalg.norm(increment, axis=-1))), 1e-9)
                predictions["R"+str(control)] = fnv+torch.tensor(random[None], dtype=torch.float64)
                random_vectors["random_"+str(control)] = random
            data.update({"term_"+name:array(value[0]) for name, value in terms.items()})
            data.update(random_vectors)
        data.update(x=array(x[0]), known_first=array(known[0]), receiver=array(receiver[0]), source=array(source[0]),
            source_mid=array(source_mid[0, start:]), false_mid=array(middle[1][0, start:]),
            mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
        data.update({arm+"_prediction":array(value[0]) for arm, value in predictions.items()})
        for label, cache in caches.items():
            for key, witness in cache.items():
                if key != "start":
                    for name, value in witness.items():
                        data[label+"_"+key+"_"+name] = array(value)
            derivative_count += len(derivative_records[label])
        entries.append(save(out/"forecasts", case, data, dict(checks=checks, derivatives=derivative_records, start=start)))
        print("forecast", number, "/32; targets not observed", flush=True)
    assert dict(counts) == EXPECTED, counts
    assert not any((out/"observations").iterdir())
    write_json(out/"forecast-manifest.json", dict(entries=entries, source_hashes=sources,
        completed_monotonic_ns=time.monotonic_ns(), ordinary_true_before_seal=0,
        native_both_attention2_before_seal=0, counts=dict(counts), local_affine_derivatives=derivative_count))
    write_json(out/"forecast-status.json", dict(status="complete", counts=dict(counts)))


def observation(args):
    out = args.out
    design = json.loads((out/"design.json").read_text())
    seal = json.loads((out/"forecast-seal.json").read_text())
    assert all(sha(ROOT/p) == h for p, h in design["source_hashes"].items())
    assert all(sha(out/p) == h for p, h in seal["files"].items())
    assert not any((out/"observations").iterdir()), "Never repeat target observation."
    begin = time.monotonic_ns()
    assert begin > seal["sealed_monotonic_ns"]
    write_json(out/"observation-start.json", dict(started_monotonic_ns=begin, seal_sha256=sha(out/"forecast-seal.json")))
    model = load_model(args.offline)
    entries = []
    for number, case in enumerate(json.loads((HERE/"cases234.json").read_text()), 1):
        path = out/"forecasts"/(case["key"]+".npz")
        saved = dict(np.load(path))
        meta = json.loads(path.with_suffix(".json").read_text())
        assert sha(path) == meta["arrays_sha256"]
        with mids(model) as middle:
            states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        checks = []
        check(checks, "known_first_native_replay", np.max(abs(array(states[0][0])-saved["known_first"])), 0.)
        start = case["position"]+1
        entries.append(save(out/"observations", case,
            dict(native_first=array(states[0][0]), native_mid=array(middle[1][0, start:]),
                 native_final=array(states[-1][0, start:]), scores=array(scores)),
            dict(checks=checks, observed_monotonic_ns=time.monotonic_ns(), forecast_sha256=sha(path),
                seal_sha256=sha(out/"forecast-seal.json"))))
        print("observe", number, "/32", flush=True)
    assert all(sha(out/p) == h for p, h in seal["files"].items())
    write_json(out/"status.json", dict(status="complete", observations=entries, ordinary_true=32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "observe"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    {"forecast":forecast, "observe":observation}[args.phase](args)
