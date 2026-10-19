"""Seal forecast-derived SiLU masks, then intervene at MLP16 on old contexts."""

import argparse
import json
import platform
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers

from .alignment228 import source_block
from .checkpoint238 import locate
from .core225b import array, capture_mlp, check, load_model, observe, sha, write_json
from .tail226b import block
from .tail240 import ARMS, execute, select

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / p for p in (
    "research/boolean_coupling/run240.py", "research/boolean_coupling/tail240.py",
    "research/boolean_coupling/audit240.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/alignment228.py", "research/boolean_coupling/tail226b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/cases228.json",
    "research/boolean_coupling/settings.json", "docs/experiments/240-target-blind-silu-tail.md")]


def save(out, phase, case, data, checks, **extra):
    assert shutil.disk_usage(out).free > 1024**3
    path = out / phase / (case["key"] + ".npz")
    assert not path.exists()
    np.savez_compressed(path, **data)
    write_json(path.with_suffix(".json"), dict(key=case["key"], group=case["group"],
        arrays_sha256=sha(path), checks=checks, **extra))
    return dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")))


def prepare(previous, out):
    assert not out.exists(), "Never overwrite evidence."
    assert json.loads((previous / "audit.json").read_text())["status"] == "passed"
    design239 = json.loads((previous / "design.json").read_text())
    upstream, old = Path(design239["input238"]), Path(design239["input228"])
    _, checkpoint = locate(ROOT)
    assert checkpoint == design239["checkpoint"]
    out.mkdir(parents=True)
    for name in ("masks", "cases", "sources"):
        (out / name).mkdir()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    for p in SOURCES:
        (out / "sources" / p.name).write_bytes(p.read_bytes())
    write_json(out / "design.json", dict(experiment=240, retrospective=True, input239=str(previous),
        input238=str(upstream), input228=str(old), checkpoint=checkpoint, source_hashes=hashes,
        audit239_sha256=sha(previous / "audit.json"), audit238_sha256=sha(upstream / "audit.json"),
        ordinary_true_calls=0, new_derivatives=False))
    model, counts, entries = load_model(True), Counter(), []
    layer = model.model.layers[15]
    write_json(out / "runtime.json", dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        revision=model.config._commit_hash))
    np.savez_compressed(out / "weights.npz", down=array(layer.mlp.down_proj.weight),
        norm=array(model.model.norm.weight), head=array(model.lm_head.weight[[895, 830]]),
        eps=np.array(model.model.norm.variance_epsilon))
    cases = json.loads((ROOT / "research/boolean_coupling/cases228.json").read_text())
    with torch.no_grad():
        for case in cases:
            key, start = case["key"], case["position"] + 1
            f = dict(np.load(old / "forecasts" / (key + ".npz")))
            # This file contains diagnostic arms too, but only the declared own input is read.
            with np.load(upstream / "cases" / (key + ".npz")) as prior:
                own_input = prior["own4_n31_finite_input"]
            checks = []
            with capture_mlp(layer) as captured:
                states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            check(checks, "false_mid_replay", np.max(abs(array(captured["mid"][0, start:]) - f["false_mids"][15])), 0.)
            operand = states[14].clone()
            operand[:, start - 1] = torch.tensor(f["source_rows"][14][None], device="mps")
            _, source_mid = source_block(model, 15, operand, context)
            counts["source_blocks"] += 1
            check(checks, "source_mid_replay", np.max(abs(array(source_mid[0, start:]).astype(float)
                - f["false_mids"][15].astype(float) - f["source_mid"][15])))
            own_state = source_mid.clone()
            own_state[:, start:] = torch.tensor(own_input[None], device="mps")
            with capture_mlp(layer) as own_capture:
                own_result = own_state + layer.mlp(layer.post_attention_layernorm(own_state))
            counts["mask_mlps"] += 1
            with np.load(upstream / "cases" / (key + ".npz")) as prior:
                check(checks, "own_local_replay", np.max(abs(array(own_result[0, start:]) - prior["own4_n31_finite_output"])), 0.)
            g0, g1, u1 = (array(v[0, start:]).astype(float) for v in
                (captured["gate"], own_capture["gate"], own_capture["up"]))
            down_norm = np.linalg.norm(array(layer.mlp.down_proj.weight).astype(float), axis=0)
            bound, score, masks = select(g0, g1, u1, down_norm, key)
            data = dict(g0=g0, phi0=array(captured["phi"][0, start:]).astype(float),
                g1=g1, u1=u1, down_norm=down_norm, bound=bound, score=score, **masks,
                source_mid=array(source_mid[0]), own_input=own_input,
                attention_mask=array(context["mask"]), rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
            entries.append(save(out, "masks", case, data, checks,
                forecast228_sha256=sha(old / "forecasts" / (key + ".npz")),
                upstream238_sha256=sha(upstream / "cases" / (key + ".npz"))))
            print(f"sealed mask {len(entries)}/32", flush=True)
    write_json(out / "mask-seal.json", dict(status="sealed", cases=entries, counts=dict(counts),
        source_hashes=hashes, native_inputs_opened=False, intervention_outcomes_measured=False,
        weights_sha256=sha(out / "weights.npz")))


def intervene(out):
    assert not (out / "status.json").exists()
    design = json.loads((out / "design.json").read_text())
    seal = json.loads((out / "mask-seal.json").read_text())
    assert seal["status"] == "sealed" and not seal["native_inputs_opened"]
    for p, h in design["source_hashes"].items():
        assert sha(ROOT / p) == sha(out / "sources" / Path(p).name) == h
    old, upstream = Path(design["input228"]), Path(design["input238"])
    model, counts, entries = load_model(True), Counter(), []
    layer = model.model.layers[15]
    cases = json.loads((ROOT / "research/boolean_coupling/cases228.json").read_text())
    with torch.no_grad():
        for case, ent in zip(cases, seal["cases"], strict=True):
            key, start = case["key"], case["position"] + 1
            path = out / "masks" / (key + ".npz")
            assert ent["key"] == key and sha(path) == ent["arrays_sha256"]
            prepared = dict(np.load(path))
            o = dict(np.load(old / "observations" / path.name))
            data, checks = {}, []
            context = dict(mask=torch.tensor(prepared["attention_mask"], device="mps"),
                rope=tuple(torch.tensor(prepared[k], device="mps") for k in ("rope_cos", "rope_sin")))
            for state_name, value in (("native", o["true_mids"][15]), ("own", prepared["own_input"])):
                data[state_name + "_input"] = value
                for arm in ARMS:
                    label = state_name + "_" + arm
                    state = torch.tensor(prepared["source_mid"][None], device="mps")
                    state[:, start:] = torch.tensor(value[None], device="mps")
                    state, captured = execute(layer, state, start, prepared, arm)
                    counts["intervention_mlps"] += 1
                    data[label + "_local"] = array(state[0, start:])
                    # Every installed activation is retained for separate auditing.
                    data[label + "_phi"] = captured["phi"]
                    if arm == "full":
                        data[state_name + "_gate"], data[state_name + "_up"] = captured["gate"], captured["up"]
                    for i in range(16, 28):
                        state = block(model, i, state, context)
                        counts["suffix_blocks"] += 1
                    data[label + "_final"] = array(state[0, start:])
                    data[label + "_scores"] = array(torch.nn.functional.linear(model.model.norm(state)[0, -1],
                        model.lm_head.weight[[895, 830]]))
                    counts["heads"] += 1
            with np.load(upstream / "cases" / path.name) as prior:
                check(checks, "own_local_replay", np.max(abs(data["own_full_local"] - prior["own4_n31_finite_output"])), 0.)
            check(checks, "native_local_replay", np.max(abs(data["native_full_local"] - o["true_states"][15])), 0.)
            check(checks, "native_final_replay", np.max(abs(data["native_full_final"] - o["true_final"])))
            check(checks, "native_scores_replay", np.max(abs(data["native_full_scores"] - o["true_scores"])), .005)
            entries.append(save(out, "cases", case, data, checks,
                observation228_sha256=sha(old / "observations" / path.name),
                mask_arrays_sha256=ent["arrays_sha256"]))
            print(f"finite primitive interventions {len(entries)}/32", flush=True)
    assert dict(counts) == dict(intervention_mlps=384, suffix_blocks=4608, heads=384)
    write_json(out / "status.json", dict(status="complete", counts=dict(counts), cases=entries,
        mask_seal_sha256=sha(out / "mask-seal.json"), ordinary_true_calls=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "intervene"))
    parser.add_argument("--input-239", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            prepare(args.input_239.resolve(), args.out.resolve())
        else:
            intervene(args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            write_json(args.out / ("failure-" + args.phase + ".json"), dict(error=repr(exc)))
        raise
