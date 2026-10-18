"""229b: correct only the inherited 228 audit-status gate before any model call."""

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
from .attention229 import PRIMITIVES, graph, mixed, primitive_removed
from .core225b import HERE, array, check, load_model, observe, sha, write_json
from .smooth228 import attention_map, promote, qualified_jvp
from .tail226b import block, stored

ROOT = HERE.parents[1]
ARMS = ("native", "base", "repair_R", "repair_C", "repair_both", "random_R", "random_C",
        *("remove_"+k for k in PRIMITIVES), *("random_"+k for k in PRIMITIVES))
EXPECTED = dict(ordinary_false=32, ordinary_true=0, public_attention=128,
                manual_attention=128, primitive_interventions=128, smooth_corners=128,
                smooth_jvps=96, fd_comparisons=96, fd_endpoints=192,
                continuation_mlps=480, continuation_blocks=12480, heads=480)
SOURCES = [HERE/n for n in ("generator229b.py", "generator229.py", "attention229.py", "audit229.py", "cases228.json",
                           "core225b.py", "smooth228.py", "alignment228.py", "tail226b.py", "settings.json")]
SOURCES += [ROOT/"docs/experiments/229-first-attention-generator.md",
            ROOT/"docs/experiments/229b-bootstrap-correction.md"]


def random_match(key, name, target):
    seed = int(hashlib.sha256((key+"229"+name).encode()).hexdigest()[:16], 16)
    value = np.random.default_rng(seed).standard_normal(target.shape)
    return value * np.linalg.norm(target)/np.linalg.norm(value)


def main(args):
    out, previous = args.out.resolve(), args.input.resolve()
    assert not out.exists(), "Never overwrite evidence."
    cases = json.loads((HERE/"cases228.json").read_text())
    old_audit = json.loads((previous/"audit.json").read_text())
    assert old_audit["status"] == "complete"
    assert old_audit["global_forecast_order_passed"]
    assert all(c["error"] <= c["bound"] for c in old_audit["checks"])
    out.mkdir(parents=True)
    (out/"sources").mkdir()
    (out/"cases").mkdir()
    sources = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    inputs = {str(p.relative_to(previous)): sha(p) for p in previous.glob("*.json")}
    for c in cases:
        for phase in ("forecasts", "observations"):
            for suffix in (".json", ".npz"):
                p = previous/phase/(c["key"]+suffix)
                inputs[str(p.relative_to(previous))] = sha(p)
    write_json(out/"design.json", dict(experiment=229, retrospective=True, created_utc=stamp(),
        source_hashes=sources, inputs=inputs, input_root=str(previous), expected=EXPECTED,
        ordinary_true_calls=0, arms=ARMS, thresholds=dict(coordinate=1e-4, derivative=1e-4,
            identity=1e-9, score=.005, local_ratio=.25, guard=.001, margin_guard=.10,
            downstream_ratio=.5, total=24, each_stratum=6)))
    model = load_model(args.offline)
    write_json(out/"runtime.json", dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        smooth_dtype="cpu_float64", attention="eager", revision=model.config._commit_hash))
    layer = model.model.layers[1]
    weights = dict(norm=array(model.model.norm.weight), head=array(model.lm_head.weight[[895, 830]]),
                   eps=np.array(model.model.norm.variance_epsilon),
                   attn_norm=array(layer.input_layernorm.weight),
                   attn_eps=np.array(layer.input_layernorm.variance_epsilon),
                   scaling=np.array(layer.self_attn.scaling), groups=np.array(layer.self_attn.num_key_value_groups),
                   width=np.array(layer.self_attn.head_dim))
    for name in ("q", "k", "v", "o"):
        module = getattr(layer.self_attn, name+"_proj")
        weights[name+"_weight"] = array(module.weight)
        if module.bias is not None:
            weights[name+"_bias"] = array(module.bias)
    np.savez_compressed(out/"weights.npz", **weights)
    counts = Counter({k: 0 for k in EXPECTED})
    entries = []
    for number, case in enumerate(cases, 1):
        started = time.monotonic_ns()
        f = stored(previous/"forecasts"/(case["key"]+".npz"))
        native = stored(previous/"observations"/(case["key"]+".npz"))
        start = case["position"]+1
        checks, data = [], {}
        with mids(model) as middle:
            states, context, scores = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        counts["ordinary_false"] += 1
        for name, got, want in (("false_boundary", array(states[0][0, start:]), f["false_boundaries"][1]),
                                ("false_mid", array(middle[1][0, start:]), f["false_mids"][1]),
                                ("false_final", array(states[-1][0, start:]), f["false_final"])):
            check(checks, name, np.max(abs(got-want)))
        x = states[0].clone()
        # Stored differences are computed in CPU64, then exact native rows are installed.
        operands = dict(zero=x, source=x.clone(), receiver=x.clone(), both=x.clone())
        operands["source"][:, start-1] = torch.tensor(f["source_rows"][0], device="mps")
        operands["receiver"][:, start:] = torch.tensor(native["true_states"][0][None], device="mps")
        operands["both"][:, start-1] = operands["source"][:, start-1]
        operands["both"][:, start:] = operands["receiver"][:, start:]
        check(checks, "exact_first_delta", np.max(abs(f["L_delta_out"][0]
              -(native["true_states"][0].astype(float)-f["false_boundaries"][1].astype(float)))), 0.)
        data.update(x=array(x[0]), source=array(operands["source"][0]),
                    receiver=array(operands["receiver"][0]), both=array(operands["both"][0]),
                    mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]),
                    false_final=f["false_final"], measured_final=native["true_final"],
                    false_scores=f["false_scores"], measured_scores=native["true_scores"])
        maps = {k: {} for k in ("public", "smooth")}
        public_mids = {}
        with torch.no_grad():
            for name, operand in operands.items():
                public_write = layer.self_attn(layer.input_layernorm(operand), attention_mask=context["mask"],
                                               position_embeddings=context["rope"])[0]
                counts["public_attention"] += 1
                public_mids[name] = operand+public_write
                for double in (False, True):
                    kind = "smooth" if double else "public"
                    result = graph(layer, operand, context, start, double)
                    counts["smooth_corners" if double else "manual_attention"] += 1
                    check(checks, kind+"_graph_"+name, np.max(abs(array(result["write"]).astype(float)
                        -array(public_write).astype(float))))
                    maps[kind][name] = result
                    data[kind+"_"+name+"_write"] = array(result["write"][0, start:])
                data[name+"_mid"] = array(public_mids[name][0, start:])
            check(checks, "native_mid_replay", np.max(abs(data["both_mid"]-native["true_mids"][1])))
            check(checks, "source_mid_replay", np.max(abs(data["source_mid"].astype(float)
                -f["source_mid"][1]-f["false_mids"][1].astype(float))))
        function = attention_map(layer, x, context, start)
        anchor = promote(x[:, start:])
        direction = promote(operands["receiver"][:, start:])-anchor
        for name, tangent_direction in (("used", direction), ("sine", torch.sin(torch.arange(anchor.numel(),
                dtype=torch.float64).reshape_as(anchor)*.013)*.01),
                ("cosine", torch.cos(torch.arange(anchor.numel(),dtype=torch.float64).reshape_as(anchor)*.013)*.01)):
            primal, tangent, plus, minus, error = qualified_jvp(function, anchor, tangent_direction)
            counts["smooth_jvps"] += 1
            counts["fd_comparisons"] += 1
            counts["fd_endpoints"] += 2
            check(checks, "jvp_fd_"+name, error)
            check(checks, "derivative_primal", np.max(abs(array(primal[0])-data["zero_mid"].astype(float))))
            for label, value in (("direction", tangent_direction), ("tangent", tangent),
                                 ("plus", plus), ("minus", minus)):
                data[name+"_"+label] = array(value[0])
        j = data["used_tangent"]
        check(checks, "original_Jd", np.max(abs(j-f["L_attn_jvp"][1])), 1e-9)
        y = {k: v.astype(float) for k, v in data.items() if k in
             ("zero_mid", "source_mid", "receiver_mid", "both_mid")}
        remainder = y["receiver_mid"]-y["zero_mid"]-j
        cross = y["both_mid"]-y["source_mid"]-y["receiver_mid"]+y["zero_mid"]
        base = y["source_mid"]+j
        error_vector = y["both_mid"]-base
        data.update(R=remainder, C=cross, E=error_vector, base_mid=base)
        check(checks, "RC_identity", np.max(abs(error_vector-remainder-cross)), 1e-9)
        terms = {}
        with torch.no_grad():
            for kind, corners in maps.items():
                term = mixed(layer, corners, context, start, kind == "smooth")
                terms[kind] = term
                for key, value in term.items():
                    data[kind+"_"+key] = array(value[0])
                direct_cross = (corners["both"]["write"]-corners["source"]["write"]
                                -corners["receiver"]["write"]+corners["zero"]["write"])[:, start:]
                check(checks, kind+"_primitive_ledger", (direct_cross-sum(term["vector_"+k] for k in PRIMITIVES)).abs().max(),
                      1e-9 if kind == "smooth" else 1e-4)
                check(checks, kind+"_logit_identity", (corners["both"]["logits"]
                      -term["added_logits"]-term["cross_logit"]).abs().max(), 1e-9 if kind == "smooth" else 1e-4)
                check(checks, kind+"_norm_disjoint", (corners["both"]["norm"]-corners["source"]["norm"]
                      -corners["receiver"]["norm"]+corners["zero"]["norm"]).abs().max(), 1e-9 if kind == "smooth" else 1e-4)
                check(checks, kind+"_probability_sum", term["soft_mix"].sum(-1).abs().max(), 1e-9 if kind == "smooth" else 1e-4)
            data["smooth_rounding_remainder"] = cross-sum(data["smooth_vector_"+k] for k in PRIMITIVES)
            check(checks, "smooth_public_C", np.max(abs(data["smooth_rounding_remainder"])))
            replacements = {}
            for k in PRIMITIVES:
                removed = primitive_removed(layer, maps["public"], terms["public"], context, start, k)
                counts["primitive_interventions"] += 1
                write = maps["public"]["both"]["write"][:, start:]
                effect = array(write[0]).astype(float)-array(removed[0]).astype(float)
                data["actual_effect_"+k] = effect
                data["removed_write_"+k] = array(removed[0])
                check(checks, "primitive_effect_"+k, np.max(abs(effect-data["smooth_vector_"+k])))
                # Install this actual primitive graph's output, preserving the native prefix.
                receiving_input = operands["both"][:, start:]
                replacements["remove_"+k] = array((receiving_input+removed)[0])
                data["random_"+k] = random_match(case["key"], k, effect)
            data["random_R"] = random_match(case["key"], "R", remainder)
            data["random_C"] = random_match(case["key"], "C", cross)
            proposals = dict(native=y["both_mid"], base=base,
                repair_R=base+remainder, repair_C=base+cross, repair_both=base+remainder+cross,
                random_R=base+data["random_R"], random_C=base+data["random_C"], **replacements,
                **{"random_"+k:y["both_mid"]-data["random_"+k] for k in PRIMITIVES})
            for arm in ARMS:
                state = public_mids["both"].clone()
                state[:, start:] = torch.tensor(proposals[arm][None], device="mps", dtype=torch.float32)
                data[arm+"_installed"] = array(state[0, start:])
                check(checks, "installed_prefix", (state[:, :start]-public_mids["both"][:, :start]).abs().max(), 0.)
                state = state+layer.mlp(layer.post_attention_layernorm(state))
                counts["continuation_mlps"] += 1
                for following in range(2, 28):
                    state = block(model, following, state, context)
                    counts["continuation_blocks"] += 1
                    if arm == "native":
                        data[f"prefix_{following}"] = array(state[0, :start])
                    else:
                        check(checks, "continuation_prefix", np.max(abs(array(state[0, :start])-data[f"prefix_{following}"])), 0.)
                score = torch.nn.functional.linear(model.model.norm(state)[0, -1], model.lm_head.weight[[895, 830]])
                counts["heads"] += 1
                data[arm+"_final"] = array(state[0, start:])
                data[arm+"_scores"] = array(score)
                if arm in ("native", "repair_both"):
                    check(checks, arm+"_final_replay", np.max(abs(data[arm+"_final"]-native["true_final"])))
                    check(checks, arm+"_score_replay", np.max(abs(data[arm+"_scores"]-native["true_scores"])), .005)
        path = out/"cases"/(case["key"]+".npz")
        np.savez_compressed(path, **data)
        record = dict(key=case["key"], group=case["group"], start=start, checks=checks,
            arrays_sha256=sha(path), recorded_utc=stamp(), elapsed_seconds=(time.monotonic_ns()-started)/1e9)
        write_json(path.with_suffix(".json"), record)
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        print(f"case {number}/32 complete ({record['elapsed_seconds']:.1f}s)", flush=True)
    assert dict(counts) == EXPECTED, dict(counts)
    assert all(sha(ROOT/p) == h for p, h in sources.items())
    write_json(out/"status.json", dict(status="complete", counts=dict(counts), cases=entries,
        weights_sha256=sha(out/"weights.npz"), completed_utc=stamp()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    main(parser.parse_args())
