"""Native finite-suffix carry/read factorial at the pre-block10 state."""

import argparse
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, load_model, observe, write_json
from .reference242b import digest
from .verdict247 import ARMS, read

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/p for p in (
    "research/boolean_coupling/suffix248.py", "research/boolean_coupling/audit248.py",
    "research/boolean_coupling/verdict247.py", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/audit242b.py", "research/boolean_coupling/primitive242b.py",
    "research/boolean_coupling/reference242b.py", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/checkpoint238.py", "research/boolean_coupling/settings.json",
    "research/boolean_coupling/cases242.json", "docs/experiments/248-native-carry-read.md")]
PROGRESS = {}


def main(previous, local, out):
    assert not out.exists()
    assert read(local/"audit.json")["status"] == "passed" and read(local/"design.json")["experiment"] == 247
    checkpoint, identity = locate(ROOT)
    assert identity == read(local/"design.json")["checkpoint"]
    assert sha(Path(qwen.__file__)) == read(local/"design.json")["qwen_source_sha256"]
    out.mkdir(parents=True)
    (out/"cases").mkdir()
    (out/"sources").mkdir()
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out/"sources"/p.name).write_bytes(p.read_bytes())
    write_json(out/"design.json", dict(experiment=248, retrospective=True, primary="verdict_state_carry", source_hashes=hashes,
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), checkpoint=identity,
        input247={n:sha(local/n) for n in ("design.json", "status.json", "audit.json")},
        input242={n:sha(previous/n) for n in ("design.json", "forecast-status.json", "status.json", "weights.npz")}))
    model = load_model(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)
    assert runtime == read(local/"runtime.json") == read(previous/"runtime.json")
    write_json(out/"runtime.json", runtime)
    le = {e["key"]:e for e in read(local/"status.json")["cases"]}
    fe = {e["key"]:e for e in read(previous/"forecast-status.json")["cases"]}
    te = {e["key"]:e for e in read(previous/"status.json")["cases"]}
    counts, entries = Counter(), []
    for number, case in enumerate(read(ROOT/"research/boolean_coupling/cases242.json"), 1):
        key, start = case["key"], case["position"]
        lp, fp, tp = local/"cases"/(key+".npz"), previous/"forecasts"/(key+".npz"), previous/"observations"/(key+".npz")
        assert sha(lp) == le[key]["arrays_sha256"] and sha(fp) == fe[key]["arrays_sha256"] and sha(tp) == te[key]["arrays_sha256"]
        meta = read(fp.with_suffix(".json"))
        data, checks = {}, []
        PROGRESS.update(key=key, data=data, checks=checks)
        with np.load(lp) as localdata, np.load(fp) as f, np.load(tp) as t, torch.no_grad():
            states, context, scores = observe(model, torch.tensor([case["false_ids"]], device="mps"))
            counts["ordinary_false"] += 1
            checked(checks, "False_full_replay", array(torch.stack(states)[:, 0, start:]), f["false_trajectory"], 0.)
            checked(checks, "False_full_score_replay", array(scores), f["false_scores"], 0.)
            parent_false = states[8]
            for arm in ARMS:
                parent = parent_false.clone()
                parent[:, start:] = torch.tensor(localdata[arm+"_input"][None], device="mps")
                data[arm+"_input"] = array(parent[0, start:])
                checked(checks, "actual_crossed_state_install", data[arm+"_input"], localdata[arm+"_input"], 0.)
                checked(checks, "unchanged_input_causal_prefix", array(parent[0, :start]), array(parent_false[0, :start]), 0.)
                trajectory = []
                state = parent
                for i in range(9, 28):
                    value = model.model.layers[i](state, attention_mask=context["mask"], position_embeddings=context["rope"])
                    state = value[0] if isinstance(value, tuple) else value
                    trajectory.append(array(state[0, start:]))
                    counts["suffix_blocks"] += 1
                    assert digest(array(state[0, :start])) == meta["false_prefix_hashes"][i]
                    assert np.isfinite(trajectory[-1]).all()
                # Match the public model's full-vocabulary projection shape.
                scores = model.lm_head(model.model.norm(state)[:, -1:])[0, 0, [895, 830]]
                counts["finite_heads"] += 1
                data[arm+"_trajectory"], data[arm+"_scores"] = np.stack(trajectory), array(scores)
                if arm in ("F", "T"):
                    ref = f["false_trajectory"] if arm == "F" else t["true_trajectory"]
                    refscore = f["false_scores"] if arm == "F" else t["true_scores"]
                    checked(checks, "native_suffix_endpoint_replay", data[arm+"_trajectory"], ref[9:], 0.)
                    checked(checks, "native_suffix_score_replay", data[arm+"_scores"], refscore, 0.)
            for a, b in (("F", "R"), ("P", "T")):
                checked(checks, "verdict_cannot_change_earlier_rows", data[a+"_trajectory"][:, :-1], data[b+"_trajectory"][:, :-1], 0.)
        path = out/"cases"/(key+".npz")
        np.savez_compressed(path, **data)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path),
            input247_sha256=sha(lp), prior_forecast_sha256=sha(fp), prior_native_sha256=sha(tp), checks=checks))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        PROGRESS.clear()
        write_json(out/"progress.json", dict(cases=number, counts=dict(counts)))
        print(f"native carry/read suffix {number}/32", flush=True)
    assert counts["ordinary_false"] == 32 and counts["suffix_blocks"] == 2432 and counts["finite_heads"] == 128
    write_json(out/"status.json", dict(status="complete", cases=entries, counts=dict(counts), ordinary_true=0, reused_True_inputs=True))
    write_json(out/"seal.json", dict(status="sealed", cases=entries, design_sha256=sha(out/"design.json"), status_sha256=sha(out/"status.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--input-247", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_242.resolve(), args.input_247.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/"partial.npz", **PROGRESS["data"])
                write_json(args.out/"partial.json", {k:v for k,v in PROGRESS.items() if k != "data"})
            write_json(args.out/"failure.json", dict(error=repr(exc)))
        raise
