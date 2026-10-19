"""Two upstream MLP switches through the actual F forecast, with native-reset controls."""

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
from .core225b import array, capture_mlp, check, load_model, observe, sha
from .smooth228 import attention_map, mlp_map, promote, qualified_jvp
from .tail226b import block

ROOT = Path(__file__).resolve().parents[2]
MODES = ("00", "10", "01", "11", "R")
ARMS = tuple(f"{incoming}_{mode}" for incoming in ("forecast", "oracle") for mode in MODES)+("finite_oracle",)
SOURCES = [ROOT/p for p in ("research/boolean_coupling/continue236.py", "research/boolean_coupling/audit236.py",
    "research/boolean_coupling/smooth228.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/alignment228.py", "research/boolean_coupling/tail226b.py",
    "research/boolean_coupling/audit235.py", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/cases228.json", "research/boolean_coupling/settings.json",
    "docs/experiments/236-upstream-mlp-factorial.md")]
EXPECTED = dict(ordinary_false=32, ordinary_true=0, source_attention=160, jvps=1664,
    fd_endpoints=3328, finite_mlps=1376, continuation_blocks=1536, heads=352)


def main(input235, out):
    assert not out.exists(), "Never overwrite evidence."
    assert json.loads((input235/"audit.json").read_text())["status"] == "passed"
    old = Path(json.loads((input235/"design.json").read_text())["input_root"])
    out.mkdir(parents=True)
    for name in ("sources", "cases"):
        (out/name).mkdir()
    sources = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    assert set(json.loads((input235/"selection.json").read_text())["selected"]) == {39, 41, 43, 45}
    design235 = json.loads((input235/"design.json").read_text())
    for name, h in design235["inputs"].items():
        assert sha(old/name) == h
    (out/"design.json").write_text(json.dumps(dict(experiment=236, retrospective=True, source_hashes=sources,
        input235=str(input235), input228=str(old), input_hashes=design235["inputs"],
        audit235_sha256=sha(input235/"audit.json"), weights235_sha256=sha(input235/"weights.npz"), expected=EXPECTED), indent=2)+"\n")
    model = load_model(True)
    (out/"runtime.json").write_text(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, transformers=transformers.__version__, device="mps", dtype="float32",
        derivative_dtype="cpu_float64", revision=model.config._commit_hash), indent=2)+"\n")
    counts, entries = Counter({k:0 for k in EXPECTED}), []
    with torch.no_grad():
        for number, case in enumerate(json.loads((ROOT/"research/boolean_coupling/cases228.json").read_text()), 1):
            key, start = case["key"], case["position"]+1
            f, o = (dict(np.load(old/phase/(key+".npz"))) for phase in ("forecasts", "observations"))
            checks, data, source_mids = [], {}, {}
            with mids(model) as middle:
                false_states, context, _ = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            embedding = model.model.embed_tokens(torch.tensor([case["false_ids"]], device="mps"))
            boundaries = [embedding, *false_states]
            check(checks, "false_replay", np.max(abs(array(torch.stack(boundaries)[:, 0, start:])-f["false_boundaries"])), 0.)
            for i in range(19, 24):
                operand = boundaries[i].clone()
                operand[:, start-1] = torch.tensor(f["source_rows"][i-1][None], device="mps")
                layer = model.model.layers[i]
                source_mids[i] = operand+layer.self_attn(layer.input_layernorm(operand), attention_mask=context["mask"],
                    position_embeddings=context["rope"])[0]
                counts["source_attention"] += 1
                check(checks, "source_mid_replay", np.max(abs(array(source_mids[i][0, start:]).astype(float)
                    -f["false_mids"][i].astype(float)-f["source_mid"][i])), 1e-4)
            amap = {i:attention_map(model.model.layers[i], boundaries[i], context, start) for i in range(20, 24)}
            mmap = {i:mlp_map(model.model.layers[i]) for i in (19, 20)}
            for i, function in amap.items():
                check(checks, "false_attention_primal", (function(promote(boundaries[i][:, start:]))-promote(middle[i][:, start:])).abs().max())
            for i, function in mmap.items():
                check(checks, "false_mlp_primal", (function(promote(middle[i][:, start:]))-promote(boundaries[i+1][:, start:])).abs().max())

            def jvp(function, anchor, direction, label):
                _, tangent, plus, minus, error = qualified_jvp(function, anchor, direction)
                check(checks, "used_receiving_derivative", error)
                counts["jvps"] += 1
                counts["fd_endpoints"] += 2
                for name, value in (("direction", direction), ("tangent", tangent), ("plus", plus), ("minus", minus)):
                    data[label+"_"+name] = array(value[0])
                return tangent

            def finite_mlp(i, proposed, label):
                operand = source_mids[i].clone()
                operand[:, start:] = proposed.to(device="mps", dtype=torch.float32)
                layer = model.model.layers[i]
                value = operand+layer.mlp(layer.post_attention_layernorm(operand))
                counts["finite_mlps"] += 1
                data[label+"_finite_parent"] = array(operand[0, start:])
                data[label+"_finite_output"] = array(value[0, start:])
                return promote(value[:, start:])

            def finish(parent, arm):
                data[arm+"_parent"] = array(parent[0, start:])
                layer = model.model.layers[23]
                value = parent+layer.mlp(layer.post_attention_layernorm(parent))
                counts["finite_mlps"] += 1
                for i in range(24, 28):
                    value = block(model, i, value, context)
                    counts["continuation_blocks"] += 1
                scores = torch.nn.functional.linear(model.model.norm(value)[0, -1], model.lm_head.weight[[895, 830]])
                counts["heads"] += 1
                data[arm+"_final"], data[arm+"_scores"] = array(value[0, start:]), array(scores)

            for incoming in ("forecast", "oracle"):
                for mode in MODES:
                    arm = incoming+"_"+mode
                    initial = f["F_pred_mid"][19] if incoming == "forecast" else o["true_mids"][19]
                    proposed = torch.tensor(initial[None], dtype=torch.float64)
                    data[arm+"_initial"] = array(proposed[0])
                    for i in range(19, 24):
                        if i > 19:
                            carry = proposed-promote(boundaries[i][:, start:])
                            tangent = jvp(amap[i], promote(boundaries[i][:, start:]), carry, f"{arm}_a{i}")
                            proposed = promote(source_mids[i][:, start:])+tangent
                        data[f"{arm}_m{i}_input"] = array(proposed[0])
                        if i == 23:
                            parent = source_mids[i].clone()
                            parent[:, start:] = proposed.to(device="mps", dtype=torch.float32)
                            finish(parent, arm)
                            continue
                        use_finite = i >= 21 or mode == "R" or mode[i-19] == "1"
                        need_linear = i < 21 and (not use_finite or mode == "R")
                        linear = None
                        if need_linear:
                            direction = proposed-promote(source_mids[i][:, start:])
                            tangent = jvp(mmap[i], promote(middle[i][:, start:]), direction, f"{arm}_m{i}")
                            linear = torch.tensor((f["false_boundaries"][i+1].astype(float)+f["source_out"][i])[None])+tangent
                            data[f"{arm}_m{i}_linear"] = array(linear[0])
                        finite = finite_mlp(i, proposed, f"{arm}_m{i}") if use_finite else None
                        if mode == "R" and i < 21:
                            correction = array((finite-linear)[0])
                            seed = int(hashlib.sha256((key+arm+str(i)+"236").encode()).hexdigest()[:16], 16)
                            random = np.random.default_rng(seed).standard_normal(correction.shape)
                            random *= np.linalg.norm(correction, axis=-1, keepdims=True)/np.linalg.norm(random, axis=-1, keepdims=True)
                            data[f"{arm}_m{i}_random"] = random
                            data[f"{arm}_m{i}_correction"] = correction
                            proposed = linear+torch.tensor(random[None])
                        else:
                            proposed = finite if use_finite else linear
                        data[f"{arm}_m{i}_output"] = array(proposed[0])
            # Fully finite oracle replay starts at the same native MLP20 input.
            operand = source_mids[19].clone()
            operand[:, start:] = torch.tensor(o["true_mids"][19][None], device="mps")
            value = operand+model.model.layers[19].mlp(model.model.layers[19].post_attention_layernorm(operand))
            counts["finite_mlps"] += 1
            with capture_mlp(model.model.layers[23]) as oracle_cap:
                for i in range(20, 28):
                    value = block(model, i, value, context)
                    counts["continuation_blocks"] += 1
            data["finite_oracle_final"] = array(value[0, start:])
            data["finite_oracle_scores"] = array(torch.nn.functional.linear(model.model.norm(value)[0, -1], model.lm_head.weight[[895, 830]]))
            data["finite_oracle_parent"] = array(oracle_cap["mid"][0, start:])
            counts["heads"] += 1
            data.update(native_final=o["true_final"], native_scores=o["true_scores"], false_final=f["false_final"],
                native_parent=o["true_mids"][-1], false_parent=f["false_mids"][-1])
            check(checks, "original_parent_replay", np.max(abs(data["forecast_00_parent"]-f["F_target_mid"])))
            check(checks, "original_final_replay", np.max(abs(data["forecast_00_final"]-f["F_suffix"][-1])))
            check(checks, "original_score_replay", np.max(abs(data["forecast_00_scores"]-f["F_scores"])), .005)
            check(checks, "native_finite_replay", np.max(abs(data["finite_oracle_final"]-o["true_final"])))
            check(checks, "native_score_replay", np.max(abs(data["finite_oracle_scores"]-o["true_scores"])), .005)
            path = out/"cases"/(key+".npz")
            np.savez_compressed(path, **data)
            meta = dict(key=key, group=case["group"], arrays_sha256=sha(path), checks=checks)
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
            entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
            print("upstream factorial", number, "/32", flush=True)
    assert dict(counts) == EXPECTED, counts
    (out/"status.json").write_text(json.dumps(dict(status="complete", counts=dict(counts), cases=entries), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-235", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_235.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            (args.out/"failure.json").write_text(json.dumps(dict(error=repr(exc)), indent=2)+"\n")
        raise
