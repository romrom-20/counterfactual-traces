"""Sealed source-driven routing forecasts with static or adaptive receiver Q/K."""

import argparse
import json
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate, read_head
from .core225b import array, load_model, observe, write_json
from .reference242b import digest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT/p for p in (
    "research/boolean_coupling/attention252.py", "research/boolean_coupling/audit252.py",
    "research/boolean_coupling/fixtures252.py", "research/boolean_coupling/cases252.json",
    "research/boolean_coupling/historical-input-hashes-252.json", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/audit242b.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/core225b.py", "research/boolean_coupling/reference242b.py",
    "research/boolean_coupling/settings.json", "docs/experiments/252-source-driven-attention.md")]
PROGRESS = {}


def read(path):
    return json.loads(path.read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_check(run):
    design = read(run/"design.json")
    for p, h in design["source_hashes"].items():
        assert sha(ROOT/p) == sha(run/"sources"/Path(p).name) == h
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    return design


def runtime(model):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)


def execute_case(model, case, phase, forecast=None):
    start, key = case["position"], case["key"]
    saved, witnesses, checks, counts, banks = {}, {}, [], Counter(), {}
    original = qwen.eager_attention_forward
    arm = "F" if phase == "forecast" else "T"
    cases = read(ROOT/"research/boolean_coupling/cases252.json")
    firsts = {}
    for c in cases:
        firsts.setdefault((c["group"]["schema"], c["group"]["reverse"]), c["key"])
    assert len(firsts) == 4
    witnessed = key in firsts.values()
    PROGRESS.update(key=key, phase=phase, data=saved, checks=checks)

    def attention(module, query, keys, values, mask, scaling, dropout=0., **kwargs):
        output, prob = original(module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs)
        index = module.layer_idx
        assert dropout == 0 and not module.training
        current_q, current_k = query, keys
        used_q, used_k = query, keys
        if arm in ("F", "T"):
            banks[index] = dict(q=query[:, :, start+1:].detach().clone(), k=keys[:, :, start+1:].detach().clone())
            saved[arm+f"_q{index+1}"] = array(banks[index]["q"][0])
            saved[arm+f"_k{index+1}"] = array(banks[index]["k"][0])
        else:
            used_q, used_k = query.clone(), keys.clone()
            if arm in ("A", "K"):
                used_q[:, :, start+1:] = banks[index]["q"]
                checked(checks, "False_receiver_query_installation", array(used_q[:, :, start+1:]), array(banks[index]["q"]), 0.)
            if arm in ("A", "Q"):
                used_k[:, :, start+1:] = banks[index]["k"]
                checked(checks, "False_receiver_key_installation", array(used_k[:, :, start+1:]), array(banks[index]["k"]), 0.)
            checked(checks, "source_and_prefix_keys_live", array(used_k[:, :, :start+1]), array(keys[:, :, :start+1]), 0.)
            raw_used = (used_q@qwen.repeat_kv(used_k, module.num_key_value_groups).transpose(2, 3))*scaling
            proposed_p = torch.softmax(raw_used+mask[:, :, :, :keys.shape[-2]], dim=-1, dtype=torch.float32).to(query.dtype)
            proposed_y = (proposed_p@qwen.repeat_kv(values, module.num_key_value_groups)).transpose(1, 2).contiguous()
            changed, changed_p = output.clone(), prob.clone()
            changed[:, start+1:] = proposed_y[:, start+1:]
            changed_p[:, :, start+1:] = proposed_p[:, :, start+1:]
            checked(checks, "actual_attention_output_installation", array(changed[:, start+1:]), array(proposed_y[:, start+1:]), 0.)
            checked(checks, "prefix_source_attention_unchanged", array(changed[:, :start+1]), array(output[:, :start+1]), 0.)
            output, prob = changed, changed_p
            counts["extra_full_QK_softmax_PV_each"] += 1
        if witnessed and index+1 in (1, 14, 28):
            raw = (used_q@qwen.repeat_kv(used_k, module.num_key_value_groups).transpose(2, 3))*scaling
            prefix = f"{arm}_{index+1}_"
            data = dict(q=array(used_q[0, :, start+1:]), k=array(used_k[0]), v=array(values[0]), raw=array(raw[0, :, start+1:]),
                mask=array(mask[0, :, start+1:, :keys.shape[-2]]), installed_p=array(prob[0, :, start+1:]),
                y=array(output[0, start+1:]).transpose(1, 0, 2), current_q=array(current_q[0, :, start+1:]), current_k=array(current_k[0]),
                groups=np.array(module.num_key_value_groups), scaling=np.array(scaling))
            witnesses.update({prefix+n:v for n, v in data.items()})
            counts["witness_extra_QK"] += 1
        return output, prob

    qwen.eager_attention_forward = attention
    try:
        prefix = None if phase == "forecast" else forecast["false_prefix_hashes"]
        with torch.no_grad():
            for arm in (("F", "Z", "A", "Q", "K") if phase == "forecast" else ("T", "O")):
                ids = case["false_ids" if arm in ("F", "Z") else "true_ids"]
                states, _, scores = observe(model, torch.tensor([ids], device="mps"))
                counts["full_forwards"] += 1
                tail = np.stack([array(state[0, start:]) for state in states])
                if arm == "F":
                    prefix = [digest(array(state[0, :start])) for state in states]
                for i, state in enumerate(states):
                    assert digest(array(state[0, :start])) == prefix[i]
                    checked(checks, "causal_prefix_exact", np.array(0.), np.array(0.), 0.)
                    if arm in ("A", "Q", "K"):
                        ratio = np.linalg.norm(tail[i].astype(float))/max(np.linalg.norm(saved["F_trajectory"][i].astype(float)), .001)
                        assert np.isfinite(tail[i]).all() and ratio <= 100, (arm, i+1, float(ratio))
                saved[arm+"_trajectory"], saved[arm+"_scores"] = tail, array(scores)
                assert np.isfinite(tail).all() and np.isfinite(array(scores)).all()
                if arm in ("Z", "O"):
                    target = "F" if arm == "Z" else "T"
                    checked(checks, "zero_or_oracle_full_trajectory_replay", tail, saved[target+"_trajectory"], 0.)
                    checked(checks, "zero_or_oracle_score_replay", array(scores), saved[target+"_scores"], 0.)
                if arm == "T":
                    checked(checks, "known_edit_source_computation", tail[:, 0], forecast["A_trajectory"][:, 0], 0.)
    finally:
        qwen.eager_attention_forward = original
    return saved, witnesses, checks, counts, prefix


def forecast(run, replication):
    assert not run.exists(), "Never overwrite evidence"
    run.mkdir(parents=True)
    for name in ("forecasts", "observations", "sources", "witnesses-forecast", "witnesses-native"):
        (run/name).mkdir()
    for p in SOURCES:
        (run/"sources"/p.name).write_bytes(p.read_bytes())
    folder, checkpoint = locate(ROOT)
    np.savez_compressed(run/"weights.npz", **read_head(folder))
    write_json(run/"design.json", dict(experiment=252, primary="A", fresh_literals=True, dependent_templates=True, replication=replication,
        checkpoint=checkpoint, qwen_source_sha256=sha(qwen.__file__), created_utc=stamp(),
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}, weights_sha256=sha(run/"weights.npz")))
    model = load_model(True)
    write_json(run/"runtime.json", runtime(model))
    counts, entries = Counter(), []
    for number, case in enumerate(read(run/"sources/cases252.json"), 1):
        saved, witness, checks, used, prefix = execute_case(model, case, "forecast")
        counts.update(used)
        assert shutil.disk_usage(run).free > 256*1024**2
        path = run/"forecasts"/(case["key"]+".npz")
        np.savez_compressed(path, **saved)
        whash = None
        if witness:
            wp = run/"witnesses-forecast"/path.name
            np.savez_compressed(wp, **witness)
            whash = sha(wp)
        write_json(path.with_suffix(".json"), dict(key=case["key"], group=case["group"], arrays_sha256=sha(path), witness_sha256=whash,
            checks=checks, false_prefix_hashes=prefix, ordinary_true=0))
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")), witness_sha256=whash))
        PROGRESS.clear()
        write_json(run/"progress-forecast.json", dict(cases=number, counts=dict(counts)))
        print(f"source-driven attention forecast {number}/32", flush=True)
    assert counts == dict(full_forwards=160, extra_full_QK_softmax_PV_each=3584, witness_extra_QK=60)
    write_json(run/"forecast-status.json", dict(status="complete", cases=entries, counts=dict(counts), ordinary_true=0))


def seal(run):
    source_check(run)
    assert not (run/"forecast-seal.json").exists() and not list((run/"observations").glob("*"))
    assert read(run/"pre-audit.json")["status"] == "passed"
    status = read(run/"forecast-status.json")
    write_json(run/"forecast-seal.json", dict(status="sealed", cases=status["cases"], ordinary_true=0, created_utc=stamp(),
        design_sha256=sha(run/"design.json"), status_sha256=sha(run/"forecast-status.json"), pre_audit_sha256=sha(run/"pre-audit.json")))


def measure(run, proof_path):
    design = source_check(run)
    assert not (run/"status.json").exists() and not list((run/"observations").glob("*"))
    proof, sealed = read(proof_path), read(run/"forecast-seal.json")
    assert proof["seal_sha256"] == sha(run/"forecast-seal.json") and proof["accepted_utc"] > sealed["created_utc"]
    assert proof["mode"] == ("local_replication" if design["replication"] else "public_remote")
    if not design["replication"]:
        assert proof["commit"] == proof["remote_head"]
        assert subprocess.check_output(["git", "ls-remote", "origin", proof["ref"]], cwd=ROOT, text=True).split()[0] == proof["commit"]
        receipt = proof["receipt_path"]
        assert subprocess.check_output(["git", "show", proof["commit"]+":"+receipt], cwd=ROOT) == (run/"forecast-seal.json").read_bytes()
    write_json(run/"measurement-proof.json", proof)
    model = load_model(True)
    assert runtime(model) == read(run/"runtime.json")
    counts, entries = Counter(), []
    for number, (case, ent) in enumerate(zip(read(run/"sources/cases252.json"), read(run/"forecast-status.json")["cases"], strict=True), 1):
        key = case["key"]
        assert key == ent["key"]
        fp = run/"forecasts"/(key+".npz")
        meta = read(fp.with_suffix(".json"))
        assert sha(fp) == ent["arrays_sha256"] and sha(fp.with_suffix(".json")) == ent["record_sha256"]
        started = stamp()
        assert started > proof["accepted_utc"]
        with np.load(fp) as data:
            payload = dict(data)
        payload["false_prefix_hashes"] = meta["false_prefix_hashes"]
        saved, witness, checks, used, _ = execute_case(model, case, "native", payload)
        counts.update(used)
        path = run/"observations"/fp.name
        np.savez_compressed(path, **saved)
        whash = None
        if witness:
            wp = run/"witnesses-native"/path.name
            np.savez_compressed(wp, **witness)
            whash = sha(wp)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path), witness_sha256=whash,
            checks=checks, native_started_utc=started, forecast_arrays_sha256=sha(fp), forecast_record_sha256=ent["record_sha256"]))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json")), witness_sha256=whash))
        PROGRESS.clear()
        print(f"source-driven attention native/control {number}/32", flush=True)
    assert counts == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=24)
    write_json(run/"status.json", dict(status="complete", cases=entries, counts=dict(counts), ordinary_true=32,
        forecast_seal_sha256=sha(run/"forecast-seal.json"), proof_sha256=sha(run/"measurement-proof.json")))
    write_json(run/"seal.json", dict(status="sealed", cases=entries, status_sha256=sha(run/"status.json"), forecast_seal_sha256=sha(run/"forecast-seal.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "seal", "measure"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    try:
        run = args.out.resolve()
        if args.phase == "forecast":
            forecast(run, args.replication)
        elif args.phase == "seal":
            seal(run)
        else:
            assert args.proof
            measure(run, args.proof.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out/("partial-"+args.phase+".npz"), **PROGRESS["data"])
                write_json(args.out/("partial-"+args.phase+".json"), {k:v for k, v in PROGRESS.items() if k != "data"})
            write_json(args.out/("failure-"+args.phase+".json"), dict(error=repr(exc)))
        raise
