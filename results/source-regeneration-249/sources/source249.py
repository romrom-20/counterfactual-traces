"""False-only source regeneration, globally sealed before oracle controls."""

import argparse
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, load_model, observe, write_json
from .reference242b import digest
from .verdict247 import read

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/p for p in (
    "research/boolean_coupling/source249.py", "research/boolean_coupling/audit249.py",
    "research/boolean_coupling/verdict247.py", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/audit242b.py", "research/boolean_coupling/primitive242b.py",
    "research/boolean_coupling/reference242b.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/settings.json",
    "research/boolean_coupling/cases242.json", "docs/experiments/249-source-regeneration.md")]
PROGRESS = {}


def model_for(previous):
    design = read(previous/"design.json")
    _, identity = locate(ROOT)
    assert identity == design["checkpoint"] and sha(Path(qwen.__file__)) == design["qwen_source_sha256"]
    model = load_model(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)
    assert runtime == read(previous/"runtime.json")
    return model, runtime


def finite_block(model, i, state, context):
    out = model.model.layers[i](state, attention_mask=context["mask"], position_embeddings=context["rope"])
    return out[0] if isinstance(out, tuple) else out


def suffix(model, state, context, start, prefix, checks):
    trajectory = []
    for i in range(9, 28):
        state = finite_block(model, i, state, context)
        assert digest(array(state[0, :start])) == prefix[i]
        trajectory.append(array(state[0, start:]))
        assert np.isfinite(trajectory[-1]).all()
    scores = model.lm_head(model.model.norm(state)[:, -1:])[0, 0, [895, 830]]
    return np.stack(trajectory), array(scores)


def forecast(previous, out):
    assert not out.exists()
    out.mkdir(parents=True)
    for name in ("sources", "forecasts", "controls"):
        (out/name).mkdir()
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"design.json", dict(experiment=249, retrospective=True, primary="native_source_regeneration", source_hashes=hashes,
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input242={n:sha(previous/n) for n in ("design.json", "forecast-status.json", "status.json", "weights.npz")}))
    model, runtime = model_for(previous)
    write_json(out/"runtime.json", runtime)
    counts, entries = Counter(), []
    for number, case in enumerate(read(ROOT/"research/boolean_coupling/cases242.json"), 1):
        key, start = case["key"], case["position"]
        path = previous/"forecasts"/(key+".npz")
        meta = read(path.with_suffix(".json"))
        saved, checks = {}, []
        PROGRESS.update(key=key, data=saved, checks=checks)
        with np.load(path) as old, torch.no_grad():
            ids = torch.tensor([case["false_ids"]], device="mps")
            states, context, scores = observe(model, ids)
            counts["ordinary_false"] += 1
            checked(checks, "False_full_native_replay", array(torch.stack(states)[:, 0, start:]), old["false_trajectory"], 0.)
            checked(checks, "False_full_native_score_replay", array(scores), old["false_scores"], 0.)
            embed = model.model.embed_tokens(ids)
            source = model.model.embed_tokens(torch.tensor([[case["true_ids"][start]]], device="mps"))[:, 0]
            rows = []
            for i, baseline in enumerate([embed, *states[:8]]):
                parent = baseline.clone()
                parent[:, start] = source
                outstate = finite_block(model, i, parent, context)
                checked(checks, "source_component_unchanged_prefix", array(outstate[0, :start]), array(states[i][0, :start]), 0.)
                source = outstate[:, start]
                rows.append(array(source[0]))
                counts["source_dense_blocks"] += 1
            parent = states[8].clone()
            saved["false_parent_full"] = array(parent[0])
            saved["mask"] = array(context["mask"])
            saved["cos"], saved["sin"] = (array(v) for v in context["rope"])
            parent[:, start] = source
            saved["S_input"], saved["source_rows"] = array(parent[0, start:]), np.stack(rows)
            checked(checks, "receiving_input_False_only", saved["S_input"][1:], old["false_trajectory"][8, 1:], 0.)
            trajectory, score = suffix(model, parent, context, start, meta["false_prefix_hashes"], checks)
            saved["S_trajectory"], saved["S_scores"] = trajectory, score
            counts["suffix_blocks"] += 19
            counts["finite_heads"] += 1
        dest = out/"forecasts"/(key+".npz")
        np.savez_compressed(dest, **saved)
        write_json(dest.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(dest), prior_forecast_sha256=sha(path),
            checks=checks, native_True_arrays_opened=False))
        entries.append(dict(key=key, arrays_sha256=sha(dest), record_sha256=sha(dest.with_suffix(".json"))))
        PROGRESS.clear()
        write_json(out/"progress.json", dict(phase="forecast", cases=number, counts=dict(counts)))
        print(f"source regeneration forecast {number}/32", flush=True)
    assert counts["ordinary_false"] == 32 and counts["source_dense_blocks"] == 288 and counts["suffix_blocks"] == 608
    write_json(out/"forecast-status.json", dict(status="complete", cases=entries, counts=dict(counts), native_True_arrays_opened=False))
    write_json(out/"forecast-seal.json", dict(status="sealed", created_utc=datetime.now(timezone.utc).isoformat(), cases=entries, design_sha256=sha(out/"design.json"), status_sha256=sha(out/"forecast-status.json"), native_True_arrays_opened=False))


def controls(previous, local, carried, out):
    assert not (out/"status.json").exists()
    seal = read(out/"forecast-seal.json")
    assert seal["status"] == "sealed" and not seal["native_True_arrays_opened"]
    assert sha(out/"forecast-status.json") == seal["status_sha256"]
    assert read(local/"audit.json")["status"] == read(carried/"audit.json")["status"] == "passed"
    for name, h in read(out/"design.json")["source_hashes"].items():
        assert sha(ROOT/name) == h
    write_json(out/"control-design.json", dict(started_utc=datetime.now(timezone.utc).isoformat(), forecast_seal_sha256=sha(out/"forecast-seal.json"),
        input247={n:sha(local/n) for n in ("status.json", "audit.json")}, input248={n:sha(carried/n) for n in ("status.json", "audit.json")}, native_True_inputs_used=True))
    model, runtime = model_for(previous)
    assert runtime == read(out/"runtime.json")
    counts, entries = Counter(), []
    for number, case in enumerate(read(ROOT/"research/boolean_coupling/cases242.json"), 1):
        key, start = case["key"], case["position"]
        lp = local/"cases"/(key+".npz")
        fp = out/"forecasts"/(key+".npz")
        pm = read((previous/"forecasts"/(key+".json")))
        saved, checks = {}, []
        PROGRESS.update(key=key, data=saved, checks=checks)
        with np.load(fp) as f, np.load(lp) as native, np.load(carried/"cases"/(key+".npz")) as ref, torch.no_grad():
            checked(checks, "source_computation_native_replay", f["S_input"][0], native["T_input"][0], 0.)
            checked(checks, "source_future_native_replay", f["S_trajectory"][:, 0], ref["T_trajectory"][:, 0], 0.)
            parent = torch.tensor(f["false_parent_full"][None], device="mps")
            parent[:, start+1:-1] = torch.tensor(native["T_input"][None, 1:-1], device="mps")
            saved["C_input"] = array(parent[0, start:])
            checked(checks, "control_keeps_False_source", saved["C_input"][0], native["F_input"][0], 0.)
            checked(checks, "control_keeps_False_verdict", saved["C_input"][-1], native["F_input"][-1], 0.)
            checked(checks, "control_True_intermediate_rows", saved["C_input"][1:-1], native["T_input"][1:-1], 0.)
            context = dict(mask=torch.tensor(f["mask"], device="mps"), rope=tuple(torch.tensor(f[n], device="mps") for n in ("cos", "sin")))
            trajectory, scores = suffix(model, parent, context, start, pm["false_prefix_hashes"], checks)
            saved["C_trajectory"], saved["C_scores"] = trajectory, scores
            checked(checks, "control_source_stays_False", trajectory[:, 0], ref["F_trajectory"][:, 0], 0.)
            counts["suffix_blocks"] += 19
            counts["finite_heads"] += 1
        path = out/"controls"/(key+".npz")
        np.savez_compressed(path, **saved)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path), forecast_arrays_sha256=sha(fp),
            input247_sha256=sha(lp), input248_sha256=sha(carried/"cases"/(key+".npz")), checks=checks))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        PROGRESS.clear()
        print(f"source regeneration control {number}/32", flush=True)
    write_json(out/"status.json", dict(status="complete", cases=entries, counts=dict(counts), ordinary_true=0, oracle_controls=True))
    write_json(out/"seal.json", dict(status="sealed", cases=entries, status_sha256=sha(out/"status.json"), forecast_seal_sha256=sha(out/"forecast-seal.json"), control_design_sha256=sha(out/"control-design.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "controls"), required=True)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--input-247", type=Path)
    parser.add_argument("--input-248", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            forecast(args.input_242.resolve(), args.out.resolve())
        else:
            assert args.input_247 and args.input_248
            controls(args.input_242.resolve(), args.input_247.resolve(), args.input_248.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/("partial-"+args.phase+".npz"), **PROGRESS["data"])
                write_json(args.out/("partial-"+args.phase+".json"), {k:v for k,v in PROGRESS.items() if k != "data"})
            write_json(args.out/("failure-"+args.phase+".json"), dict(error=repr(exc)))
        raise
