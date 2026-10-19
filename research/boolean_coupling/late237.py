"""Frozen late-attention QK/softmax factorial with finite MLPs and final readout."""

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from .alignment228 import mids
from .attention229 import graph
from .core225b import array, check, load_model, observe, sha
from .curvature230 import corner
from .finite235 import capture_weights
from .smooth228 import attention_map, promote, qualified_jvp
from .tail226b import block

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("1001", "1101", "1011", "1111")
MODES = ("legacy", *POLICIES, "public")
ARMS = tuple(f"{regime}_{mode}" for regime in ("forecast", "oracle") for mode in MODES)
EXPECTED = dict(ordinary_false=32, ordinary_true=0, source_attention=160, false_graphs=128,
    source_corners=512, receiving_corners=1792, legacy_jvps=256, legacy_fd_endpoints=512,
    public_attention=512, finite_mlps=1920, suffix_blocks=1536, heads=384)
SOURCES = [ROOT/p for p in ("research/boolean_coupling/late237.py", "research/boolean_coupling/audit237.py",
    "research/boolean_coupling/curvature230.py", "research/boolean_coupling/attention229.py",
    "research/boolean_coupling/smooth228.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/alignment228.py", "research/boolean_coupling/finite235.py",
    "research/boolean_coupling/ledger235.py", "research/boolean_coupling/tail226b.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit230.py",
    "research/boolean_coupling/audit235.py", "research/boolean_coupling/audit236.py",
    "research/boolean_coupling/cases228.json", "research/boolean_coupling/settings.json",
    "docs/experiments/237-late-attention-primitives.md")]


def main(previous, out):
    assert not out.exists(), "Never overwrite evidence."
    assert json.loads((previous/"audit.json").read_text())["status"] == "passed"
    prior = json.loads((previous/"design.json").read_text())
    old, weights235 = Path(prior["input228"]), Path(prior["input235"])/"weights.npz"
    for name, h in prior["input_hashes"].items():
        assert sha(old/name) == h
    assert sha(weights235) == prior["weights235_sha256"]
    out.mkdir(parents=True)
    (out/"cases").mkdir()
    (out/"sources").mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    (out/"design.json").write_text(json.dumps(dict(experiment=237, retrospective=True, expected=EXPECTED,
        source_hashes=sources, input236=str(previous), audit236_sha256=sha(previous/"audit.json"),
        input228=str(old), input_hashes=prior["input_hashes"], weights235=str(weights235),
        weights235_sha256=sha(weights235)), indent=2)+"\n")
    model = load_model(True)
    np.savez_compressed(out/"weights.npz", **capture_weights(model, (40, 42, 44, 46)))
    (out/"runtime.json").write_text(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        primitive_dtype="cpu_float64", revision=model.config._commit_hash), indent=2)+"\n")
    counts, entries, derivative_total = Counter({k:0 for k in EXPECTED}), [], 0
    with torch.no_grad():
        for number, case in enumerate(json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text()), 1):
            key, start = case["key"], case["position"]+1
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            old236 = dict(np.load(previous/"cases"/(key+".npz")))
            checks, data, derivatives, source_mids, source_inputs = [], {}, [], {}, {}
            with mids(model) as middle:
                false_states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            embedding = model.model.embed_tokens(torch.tensor([case["false_ids"]], device="mps"))
            boundaries = [embedding, *false_states]
            data.update(mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
            check(checks, "false_replay", np.max(abs(array(torch.stack(boundaries)[:, 0, start:])-f["false_boundaries"])), 0.)
            for i in range(19, 24):
                operand = boundaries[i].clone()
                operand[:, start-1] = torch.tensor(f["source_rows"][i-1][None], device="mps")
                layer = model.model.layers[i]
                source_inputs[i] = operand
                source_mids[i] = operand+layer.self_attn(layer.input_layernorm(operand), attention_mask=context["mask"],
                    position_embeddings=context["rope"])[0]
                counts["source_attention"] += 1
                check(checks, "source_mid_replay", np.max(abs(array(source_mids[i][0, start:]).astype(float)
                    -f["false_mids"][i].astype(float)-f["source_mid"][i])), 1e-4)
            anchors, caches, sources_output, legacy_maps = {}, {}, {}, {}
            for i in range(20, 24):
                layer = model.model.layers[i]
                anchors[i] = graph(layer, boundaries[i], context, 0, True)
                caches[i] = {"start":start}
                counts["false_graphs"] += 1
                data[f"a{i}_x"] = array(boundaries[i][0])
                data[f"a{i}_source"] = array(source_inputs[i][0])
                data[f"a{i}_source_mid"] = array(source_mids[i][0, start:])
                for bits in POLICIES:
                    result, _ = corner(layer, boundaries[i], promote(source_inputs[i]), context, start, anchors[i],
                        bits, True, derivatives, caches[i])
                    sources_output[i, bits] = result
                    data[f"a{i}_source_{bits}"] = array(result[0, start:])
                    counts["source_corners"] += 1
                check(checks, "smooth_full_source_bridge", np.max(abs(data[f"a{i}_source_1111"]-data[f"a{i}_source_mid"])))
                legacy_maps[i] = attention_map(layer, boundaries[i], context, start)
                check(checks, "legacy_false_primal", (legacy_maps[i](promote(boundaries[i][:, start:]))-promote(middle[i][:, start:])).abs().max())

            def finite_mlp(i, proposed, label):
                operand = source_mids[i].clone()
                operand[:, start:] = proposed.to("mps", torch.float32)
                layer = model.model.layers[i]
                value = operand+layer.mlp(layer.post_attention_layernorm(operand))
                data[label+"_parent"], data[label+"_output"] = array(operand[0, start:]), array(value[0, start:])
                counts["finite_mlps"] += 1
                return promote(value[:, start:]), value

            def public_attention(i, proposed):
                operand = source_inputs[i].clone()
                operand[:, start:] = proposed.to("mps", torch.float32)
                layer = model.model.layers[i]
                value = operand+layer.self_attn(layer.input_layernorm(operand), attention_mask=context["mask"],
                    position_embeddings=context["rope"])[0]
                counts["public_attention"] += 1
                return promote(value[:, start:])

            def policy(i, proposed, bits, label):
                operand = promote(source_inputs[i]).clone()
                operand[:, start:] = proposed
                value, _ = corner(model.model.layers[i], boundaries[i], operand, context, start, anchors[i],
                    bits, True, derivatives, caches[i])
                counts["receiving_corners"] += 1
                result = promote(source_mids[i][:, start:])+value[:, start:]-sources_output[i, bits][:, start:]
                data[label+"_input"] = array(proposed[0])
                data[label+"_corner"] = array(value[0, start:])
                data[label+"_output"] = array(result[0])
                return result

            for regime in ("forecast", "oracle"):
                for mode in MODES:
                    arm = regime+"_"+mode
                    initial = f["F_pred_mid"][19] if regime == "forecast" else o["true_mids"][19]
                    proposed = torch.tensor(initial[None], dtype=torch.float64)
                    data[arm+"_initial"] = array(proposed[0])
                    proposed, _ = finite_mlp(19, proposed, arm+"_m19")
                    for i in range(20, 24):
                        label = f"{arm}_a{i}"
                        incoming = proposed
                        if mode == "legacy":
                            direction = proposed-promote(boundaries[i][:, start:])
                            _, tangent, plus, minus, error = qualified_jvp(legacy_maps[i], promote(boundaries[i][:, start:]), direction)
                            check(checks, "legacy_used_derivative", error)
                            for name, value in (("direction", direction), ("tangent", tangent), ("plus", plus), ("minus", minus)):
                                data[label+"_"+name] = array(value[0])
                            counts["legacy_jvps"] += 1
                            counts["legacy_fd_endpoints"] += 2
                            proposed = promote(source_mids[i][:, start:])+tangent
                            data[label+"_input"], data[label+"_output"] = array(incoming[0]), array(proposed[0])
                        elif mode == "public":
                            proposed = public_attention(i, proposed)
                            data[label+"_input"], data[label+"_output"] = array(incoming[0]), array(proposed[0])
                        else:
                            proposed = policy(i, proposed, mode, label)
                            if mode == "1111":
                                bridge = public_attention(i, incoming)
                                data[label+"_public"] = array(bridge[0])
                                check(checks, "smooth_full_receiving_bridge", (proposed-bridge).abs().max())
                            if mode == "1001":
                                for bits in POLICIES[1:]:
                                    policy(i, incoming, bits, f"{regime}_fixed{i}_{bits}")
                        data[f"{arm}_m{i}_input"] = array(proposed[0])
                        if i < 23:
                            proposed, _ = finite_mlp(i, proposed, f"{arm}_m{i}")
                        else:
                            parent = source_mids[i].clone()
                            parent[:, start:] = proposed.to("mps", torch.float32)
                            data[arm+"_parent"] = array(parent[0, start:])
                            proposed, parent = finite_mlp(i, proposed, f"{arm}_m{i}")
                            for j in range(24, 28):
                                parent = block(model, j, parent, context)
                                counts["suffix_blocks"] += 1
                            scores = torch.nn.functional.linear(model.model.norm(parent)[0, -1], model.lm_head.weight[[895, 830]])
                            data[arm+"_final"], data[arm+"_scores"] = array(parent[0, start:]), array(scores)
                            counts["heads"] += 1
                for suffix, bound in (("parent", 1e-4), ("final", 1e-4), ("scores", .005)):
                    check(checks, "legacy_236_"+suffix, np.max(abs(data[regime+"_legacy_"+suffix]-old236[regime+"_11_"+suffix])), bound)
                    check(checks, "full_graph_public_"+suffix, np.max(abs(data[regime+"_1111_"+suffix]-data[regime+"_public_"+suffix])), bound)
            check(checks, "native_finite_replay", np.max(abs(data["oracle_public_final"]-o["true_final"])))
            check(checks, "native_finite_scores", np.max(abs(data["oracle_public_scores"]-o["true_scores"])), .005)
            # A witness cache is per layer: identical byte directions in different layers never alias.
            for i, cache in caches.items():
                for digest, witness in cache.items():
                    if digest != "start":
                        for name, value in witness.items():
                            data[f"w{i}_{digest}_{name}"] = array(value)
            # Each derivative record occurs on its first evaluation at one layer.
            # Find its cache membership explicitly rather than relying on a global direction hash.
            for record in derivatives:
                matches = [i for i, cache in caches.items() if record["witness_key"] in cache]
                assert len(matches) == 1, "Ambiguous cross-layer derivative direction."
                record["layer_index"] = matches[0]
            derivative_total += len(derivatives)
            data.update(native_final=o["true_final"], native_scores=o["true_scores"], false_final=f["false_final"])
            path = out/"cases"/(key+".npz")
            np.savez_compressed(path, **data)
            meta = dict(key=key, group=case["group"], start=start, arrays_sha256=sha(path), checks=checks, derivatives=derivatives,
                input236_arrays_sha256=sha(previous/"cases"/path.name))
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
            entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
            print(f"late attention {number}/32; {len(derivatives)} qualified primitive directions", flush=True)
    assert dict(counts) == EXPECTED, counts
    assert all(sha(ROOT/p) == h for p, h in sources.items())
    (out/"status.json").write_text(json.dumps(dict(status="complete", counts=dict(counts), cases=entries,
        weights_sha256=sha(out/"weights.npz"), primitive_derivatives=derivative_total,
        primitive_fd_endpoints=2*derivative_total), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-236", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_236.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            (args.out/"failure.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
