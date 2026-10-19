"""Globally sealed local attention forecasts with a declared known-edit budget."""

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
from .attention229 import graph
from .core225b import HERE, array, check, load_model, observe, sha, write_json
from .curvature230 import corner
from .smooth228 import attention_map, promote, qualified_jvp
from .tail226b import block

ROOT = HERE.parents[1]
ARMS = ("L", "N", "NV", "R")
SOURCES = [HERE/n for n in ("local231.py", "audit231.py", "fixtures231.py", "cases231.json",
    "historical-input-hashes-231.json", "curvature230.py", "attention229.py", "smooth228.py",
    "core225b.py", "alignment228.py", "tail226b.py", "audit230.py", "audit229.py", "settings.json")]
SOURCES += [ROOT/"docs/experiments/231-local-normalization-product-forecast.md"]
EXPECTED = dict(ordinary_false=16, ordinary_true=0, edited_first_blocks=16,
    source_only_attention=16, smooth_false_graphs=16, smooth_intervention_graphs=48,
    receiving_jvps=16, receiving_fd_endpoints=32)


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
    write_json(out/"design.json", dict(experiment=231, prospective_local_only=True, created_utc=stamp(),
        source_hashes=sources, expected_forecast=EXPECTED, thresholds=dict(relative=.025, cosine=.995,
            guard=.001, total=12, each_stratum=3),
        known_edit_budget="One finite public block 1 plus one source-only attention 2, per case; "
                          "one ordinary False computation. No native both-input attention-2 evaluation before seal."))
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
    for number, case in enumerate(json.loads((HERE/"cases231.json").read_text()), 1):
        start = case["position"]+1
        with mids(model) as middle:
            states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        counts["ordinary_false"] += 1
        checks, derivatives, cache = [], [], {"start":start}
        with torch.no_grad():
            embedding = model.model.embed_tokens(torch.tensor([case["true_ids"]], device="mps"))
            known = block(model, 0, embedding, context)
            counts["edited_first_blocks"] += 1
            x = states[0]
            check(checks, "known_first_prefix", (known[:, :start-1]-x[:, :start-1]).abs().max(), 0.)
            source, receiver = x.clone(), x.clone()
            source[:, start-1] = known[:, start-1]
            receiver[:, start:] = known[:, start:]
            source_write = layer.self_attn(layer.input_layernorm(source), attention_mask=context["mask"],
                position_embeddings=context["rope"])[0]
            source_mid = source+source_write
            counts["source_only_attention"] += 1
            anchor = graph(layer, x, context, 0, True)
            counts["smooth_false_graphs"] += 1
            check(checks, "false_smooth_primal", (promote(x[:, start:])+anchor["write"][:, start:]-promote(middle[1][:, start:])).abs().max())
            local = {}
            for bits in ("0000", "1000", "1001"):
                local[bits], _ = corner(layer, x, receiver, context, start, anchor, bits, True, derivatives, cache)
                counts["smooth_intervention_graphs"] += 1
            function = attention_map(layer, x, context, start)
            primal, tangent, plus, minus, error = qualified_jvp(function, promote(x[:, start:]),
                promote(receiver[:, start:])-promote(x[:, start:]))
            counts["receiving_jvps"] += 1
            counts["receiving_fd_endpoints"] += 2
            check(checks, "receiving_fd", error)
            check(checks, "all_affine_Jd", (local["0000"][:, start:]-primal-tangent).abs().max(), 1e-9)
            base = promote(source_mid[:, start:])+tangent
            n = local["1000"][:, start:]-local["0000"][:, start:]
            nv = local["1001"][:, start:]-local["0000"][:, start:]
            extra = array((nv-n)[0])
            seed = int(hashlib.sha256((case["key"]+"231").encode()).hexdigest()[:16], 16)
            random = np.random.default_rng(seed).standard_normal(extra.shape)
            random *= np.linalg.norm(extra)/np.linalg.norm(random)
            check(checks, "random_norm", abs(np.linalg.norm(random)-np.linalg.norm(extra)), 1e-9)
            predictions = dict(L=base, N=base+n, NV=base+nv, R=base+n+torch.tensor(random[None], dtype=torch.float64))
        data = dict(x=array(x[0]), known_first=array(known[0]), receiver=array(receiver[0]), source=array(source[0]),
            source_mid=array(source_mid[0, start:]), false_mid=array(middle[1][0, start:]),
            mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]),
            tangent=array(tangent[0]), plus=array(plus[0]), minus=array(minus[0]), random=random,
            N_correction=array(n[0]), NV_correction=array(nv[0]))
        data.update({arm+"_prediction":array(value[0]) for arm, value in predictions.items()})
        for key, witness in cache.items():
            if key != "start":
                for name, value in witness.items():
                    data[key+"_"+name] = array(value)
        derivative_count += len(derivatives)
        entries.append(save(out/"forecasts", case, data, dict(checks=checks, derivatives=derivatives, start=start)))
        print("forecast", number, "/16; targets not observed", flush=True)
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
    for number, case in enumerate(json.loads((HERE/"cases231.json").read_text()), 1):
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
        print("observe", number, "/16", flush=True)
    assert all(sha(out/p) == h for p, h in seal["files"].items())
    write_json(out/"status.json", dict(status="complete", observations=entries, ordinary_true=16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "observe"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    {"forecast":forecast, "observe":observation}[args.phase](args)
