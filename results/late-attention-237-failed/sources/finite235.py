"""Finite conditional diagnosis and terminal-vector interventions for global ledger235."""

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from .alignment228 import mids
from .core225b import array, check, load_model, observe, sha
from .ledger235 import NODES, ROOT, states
from .tail226b import block

ARMS = ("base", "full", "top1", "top2", "top4", "gen4", "state4", "replay4",
        "random2_0", "random2_1", "random2_2", "random4_0", "random4_1", "random4_2")


def row_random(key, name, value):
    seed = int(hashlib.sha256((key+name+"235").encode()).hexdigest()[:16], 16)
    vector = np.random.default_rng(seed).standard_normal(value.shape)
    return vector*np.linalg.norm(value, axis=-1, keepdims=True)/np.linalg.norm(vector, axis=-1, keepdims=True)


def capture_weights(model, selected):
    weights = dict(final_norm=array(model.model.norm.weight), final_head=array(model.lm_head.weight[[895, 830]]),
        final_eps=np.array(model.model.norm.variance_epsilon))
    for node in selected:
        i, kind = NODES[node]
        layer = model.model.layers[i]
        local = {}
        if kind == "attention":
            attn = layer.self_attn
            local.update(attn_norm=array(layer.input_layernorm.weight), attn_eps=np.array(layer.input_layernorm.variance_epsilon),
                width=np.array(attn.head_dim), groups=np.array(attn.num_key_value_groups), scaling=np.array(attn.scaling))
            modules = [(name, getattr(attn, name+"_proj")) for name in ("q", "k", "v", "o")]
        else:
            local.update(norm=array(layer.post_attention_layernorm.weight), eps=np.array(layer.post_attention_layernorm.variance_epsilon))
            modules = [(name, getattr(layer.mlp, name+"_proj")) for name in ("gate", "up", "down")]
        for name, module in modules:
            local[name+"_weight"] = array(module.weight)
            if module.bias is not None:
                local[name+"_bias"] = array(module.bias)
        weights.update({f"n{node}_"+name:value for name, value in local.items()})
    return weights


def main(run):
    assert not (run/"status.json").exists() and not (run/"weights.npz").exists()
    design = json.loads((run/"design.json").read_text())
    old = Path(design["input_root"])
    for name, h in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == h
    for name, h in design["inputs"].items():
        assert sha(old/name) == h
    selection = json.loads((run/"selection.json").read_text())
    assert sha(run/"selection.json") == json.loads((run/"ledger-status.json").read_text())["selection_sha256"]
    selected = selection["selected"]
    assert len(selected) == 4
    mlps = sum(NODES[n][1] == "mlp" for n in selected)
    expected = dict(ordinary_false=32, ordinary_true=0, source_attention=32*(1+mlps),
        selected_finite=256, intervention_mlps=448, suffix_blocks=1792, heads=448)
    model = load_model(True)
    (run/"runtime.json").write_text(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        revision=model.config._commit_hash, expected=expected), indent=2)+"\n")
    np.savez_compressed(run/"weights.npz", **capture_weights(model, selected))
    counts = Counter({k:0 for k in expected})
    entries = []
    with torch.no_grad():
        for number, case in enumerate(json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text()), 1):
            key, start = case["key"], case["position"]+1
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            budget = dict(np.load(run/"ledger"/(key+".npz")))
            with mids(model) as middle:
                false_states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            embedding = model.model.embed_tokens(torch.tensor([case["false_ids"]], device="mps"))
            true_embedding = model.model.embed_tokens(torch.tensor([case["true_ids"]], device="mps"))
            boundaries = [embedding, *false_states]
            checks, saved, source_cache = [], {}, {}
            check(checks, "false_boundary_replay", np.max(abs(array(torch.stack(boundaries)[:, 0, start:])-f["false_boundaries"])), 0.)
            check(checks, "false_mid_replay", np.max(abs(array(torch.stack(middle)[:, 0, start:])-f["false_mids"])), 0.)

            def source_input(i):
                parent = boundaries[i].clone()
                row = true_embedding[:, start-1] if i == 0 else torch.tensor(f["source_rows"][i-1][None], device="mps")
                parent[:, start-1] = row
                return parent

            def source_mid(i):
                if i not in source_cache:
                    parent = source_input(i)
                    layer = model.model.layers[i]
                    write = layer.self_attn(layer.input_layernorm(parent), attention_mask=context["mask"],
                        position_embeddings=context["rope"])[0]
                    source_cache[i] = parent+write
                    counts["source_attention"] += 1
                    check(checks, "source_mid_reference", np.max(abs(array(source_cache[i][0, start:]).astype(float)
                        -(f["false_mids"][i].astype(float)+f["source_mid"][i]))), 1e-4)
                return source_cache[i]

            generation, state_terms, replay = [], [], []
            for node in selected:
                i, kind = NODES[node]
                layer = model.model.layers[i]
                ni, no, pi, po = states(f, o, i, kind)
                reference = source_input(i) if kind == "attention" else source_mid(i)
                native, predicted = reference.clone(), reference.clone()
                native[:, start:] = torch.tensor(ni[None], device="mps", dtype=torch.float32)
                predicted[:, start:] = torch.tensor(pi[None], device="mps", dtype=torch.float32)

                def finite(parent):
                    if kind == "attention":
                        write = layer.self_attn(layer.input_layernorm(parent), attention_mask=context["mask"],
                            position_embeddings=context["rope"])[0]
                        return parent+write
                    return parent+layer.mlp(layer.post_attention_layernorm(parent))

                yn, yp = (array(finite(parent)[0, start:]).astype(float) for parent in (native, predicted))
                counts["selected_finite"] += 2
                installed_n, installed_p = (array(parent[0, start:]).astype(float) for parent in (native, predicted))
                check(checks, "native_component_replay", np.max(abs(yn-no)), 1e-4)
                ng, pg = yn-installed_n, yp-installed_p
                g = pg-(po-pi)
                st, rounding = ng-pg, (no-ni)-ng
                check(checks, "conditional_split", np.max(abs(g+st+rounding-budget["terms"][node])), 1e-9)
                generation.append(g)
                state_terms.append(st)
                replay.append(rounding)
                prefix = f"n{node}_"
                for label, value in dict(native_input=array(native[0]), forecast_input=array(predicted[0]),
                    native_output=yn, forecast_output=yp, forecast_input64=pi, native_reference=no,
                    forecast_reference=po, generation=g, state=st, replay=rounding,
                    input_cast=installed_p-pi).items():
                    saved[prefix+label] = value
            corrections = dict(base=np.zeros_like(budget["target_error"]), full=budget["terms"].sum(0),
                top1=budget["terms"][selected[:1]].sum(0), top2=budget["terms"][selected[:2]].sum(0),
                top4=budget["terms"][selected].sum(0), gen4=sum(generation), state4=sum(state_terms), replay4=sum(replay))
            check(checks, "top4_split", np.max(abs(corrections["top4"]-corrections["gen4"]-corrections["state4"]-corrections["replay4"])), 1e-9)
            for total in (2, 4):
                for control in range(3):
                    name = f"random{total}_{control}"
                    corrections[name] = row_random(key, name, corrections[f"top{total}"])
            terminal_reference = source_mid(23)
            saved.update(mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]),
                native_final=o["true_final"], native_scores=o["true_scores"], false_final=f["false_final"],
                parent_native=o["true_mids"][-1], parent_base=f["F_target_mid"],
                reference_final=f["F_suffix"][-1], reference_scores=f["F_scores"])
            for arm in ARMS:
                operand = terminal_reference.clone()
                receiving = f["F_target_mid"].astype(float)+corrections[arm]
                operand[:, start:] = torch.tensor(receiving[None], device="mps", dtype=torch.float32)
                saved[arm+"_installed"] = array(operand[0, start:])
                saved[arm+"_correction"] = corrections[arm]
                layer = model.model.layers[23]
                output = operand+layer.mlp(layer.post_attention_layernorm(operand))
                counts["intervention_mlps"] += 1
                for i in range(24, 28):
                    output = block(model, i, output, context)
                    counts["suffix_blocks"] += 1
                scores = torch.nn.functional.linear(model.model.norm(output)[0, -1], model.lm_head.weight[[895, 830]])
                counts["heads"] += 1
                saved[arm+"_final"], saved[arm+"_scores"] = array(output[0, start:]), array(scores)
                if arm in ("base", "full"):
                    final = f["F_suffix"][-1] if arm == "base" else o["true_final"]
                    reference_scores = f["F_scores"] if arm == "base" else o["true_scores"]
                    check(checks, arm+"_final_replay", np.max(abs(saved[arm+"_final"]-final)), 1e-4)
                    check(checks, arm+"_score_replay", np.max(abs(saved[arm+"_scores"]-reference_scores)), .005)
            path = run/"cases"/(key+".npz")
            assert not path.exists()
            np.savez_compressed(path, **saved)
            meta = dict(key=key, group=case["group"], start=start, selected=selected, checks=checks,
                arrays_sha256=sha(path), selection_sha256=sha(run/"selection.json"))
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
            entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
            print("finite global diagnostic", number, "/32", flush=True)
    assert dict(counts) == expected, counts
    (run/"status.json").write_text(json.dumps(dict(status="complete", cases=entries, counts=dict(counts),
        weights_sha256=sha(run/"weights.npz")), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    target = parser.parse_args().run.resolve()
    try:
        main(target)
    except Exception as exc:
        if target.exists():
            (target/"failure-finite.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
