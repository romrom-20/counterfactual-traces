"""Finite primitive diagnosis of the first regional interaction at attention2."""

import argparse
import platform
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, load_model, observe, write_json
from .native244 import ARMS, bank_from, read, region, stamp
from .native244 import SOURCES as PARENTS
from .primitive242b import Taylor
from .reference242b import digest

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = {}
EDITS = ("QK", "SM", "PV", "ALL", "random0", "random1", "random2")
SOURCES = list(dict.fromkeys([*PARENTS, *[ROOT / p for p in (
    "research/boolean_coupling/mixing245.py", "research/boolean_coupling/audit245.py",
    "docs/experiments/245-first-mixing.md")]]))


def random_rows(key, label, value):
    import hashlib
    seed = int(hashlib.sha256((key+label).encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=value.shape)
    # value is heads x receiving rows x width; match each whole receiving row.
    def norms(a):
        return np.sqrt(np.sum(a*a, axis=(0, 2), keepdims=True))
    return v * norms(value) / norms(v)


def main(previous, regional, out):
    assert not out.exists(), "Never overwrite evidence."
    assert read(regional / "audit.json")["status"] == "passed"
    assert read(regional / "design.json")["experiment"] == 244
    for name, h in read(regional / "design.json")["source_hashes"].items():
        assert sha(ROOT / name) == h
    out.mkdir(parents=True)
    (out / "sources").mkdir()
    (out / "cases").mkdir()
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    for p in SOURCES:
        (out / "sources" / p.name).write_bytes(p.read_bytes())
    write_json(out / "design.json", dict(experiment=245, retrospective=True, site="attention2",
        primary="PV_local_and_final", source_hashes=hashes, created_utc=stamp(),
        inputs={str(p.resolve()):sha(p) for p in [regional / "design.json", regional / "status.json", regional / "audit.json",
            previous / "design.json", previous / "forecast-status.json", previous / "weights.npz"]}))
    _, checkpoint = locate(ROOT)
    assert checkpoint == read(regional / "design.json")["checkpoint"]
    assert sha(Path(qwen.__file__)) == read(regional / "design.json")["qwen_source_sha256"]
    model = load_model(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)
    assert runtime == read(regional / "runtime.json") == read(previous / "runtime.json")
    write_json(out / "runtime.json", runtime)
    cases = read(ROOT / "research/boolean_coupling/cases242.json")
    entries, counts = [], Counter()
    old_entries = {e["key"]:e for e in read(regional / "status.json")["cases"]}
    first_entries = {e["key"]:e for e in read(previous / "forecast-status.json")["cases"]}
    for number, case in enumerate(cases, 1):
        key, start = case["key"], case["position"]
        saved, checks, witnesses = {}, [], {}
        PROGRESS.update(key=key, data=saved, checks=checks, qualifications=witnesses)
        rpath, fpath = regional / "cases" / (key+".npz"), previous / "forecasts" / (key+".npz")
        assert sha(rpath) == old_entries[key]["arrays_sha256"] and sha(fpath) == first_entries[key]["arrays_sha256"]
        meta = read(fpath.with_suffix(".json"))
        original = qwen.eager_attention_forward
        captured, edit = {}, None

        def attention(module, query, key_states, value, attention_mask, scaling, dropout=0., **kwargs):
            result = original(module, query, key_states, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
            if module.layer_idx != 1:
                return result
            repeated_v = qwen.repeat_kv(value, module.num_key_value_groups)
            repeated_k = qwen.repeat_kv(key_states, module.num_key_value_groups)
            raw = torch.matmul(query, repeated_k.transpose(2, 3)) * scaling
            counts["diagnostic_QK_products"] += 1
            mask = attention_mask[:, :, start+1:, :key_states.shape[-2]]
            if edit is None:
                captured.update(q=array(query[0, :, start+1:]), k=array(key_states[0]), v=array(value[0]),
                    p=array(result[1][0, :, start+1:]), raw=array(raw[0, :, start+1:]), mask=array(mask[0]),
                    y=array(result[0][0, start+1:]).transpose(1, 0, 2), scaling=np.array(scaling), groups=np.array(module.num_key_value_groups))
                return result
            output, probability = result
            if edit in ("QK", "SM"):
                proposed = saved["pstar"] if edit == "QK" else saved["FF_p"].astype(float)-saved["SM_probability"]
                probability = probability.clone()
                probability[:, :, start+1:] = torch.tensor(proposed[None].astype(np.float32), device="mps")
                output = torch.matmul(probability, repeated_v).transpose(1, 2).contiguous()
                counts["intervention_value_products"] += 1
                checked(checks, "installed_probability_"+edit, array(probability[0, :, start+1:]), proposed.astype(np.float32), 0.)
            else:
                term = saved["IY"] if edit == "ALL" else saved["PV"] if edit == "PV" else random_rows(case["key"], edit, saved["PV"])
                output = output.clone()
                proposed = (saved["FF_y"].astype(float)-term).astype(np.float32)
                output[:, start+1:] = torch.tensor(proposed.transpose(1, 0, 2)[None], device="mps")
                checked(checks, "installed_weighted_value_"+edit, array(output[0, start+1:]).transpose(1, 0, 2), proposed, 0.)
            saved[edit+"_installed_y"] = array(output[0, start+1:]).transpose(1, 0, 2)
            checked(checks, "unchanged_attention_source_"+edit, array(output[:, :start+1]), array(result[0][:, :start+1]), 0.)
            counts["primitive_installations"] += 1
            return output, probability

        with np.load(fpath) as old, np.load(rpath) as ref, torch.no_grad():
            checked(checks, "zero_incoming_mixed_state", ref["FF_trajectory"][0].astype(float)-ref["FQ_trajectory"][0].astype(float)
                -ref["QF_trajectory"][0].astype(float)+ref["QQ_trajectory"][0].astype(float), np.zeros_like(ref["QQ_trajectory"][0]), 0.)
            # region installs its own attention wrapper. Capture AFTER that wrapper.
            for arm in ARMS:
                taylor = Taylor(bank_from(old, meta, arm))
                taylor.arm = arm
                with region(model, start, taylor, arm, counts, checks):
                    parent = qwen.eager_attention_forward
                    qwen.eager_attention_forward = attention
                    try:
                        states, context, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
                    finally:
                        qwen.eager_attention_forward = parent
                counts["native_replays" if arm == "FF" else "regional_replays"] += 1
                for i, state in enumerate(states):
                    checked(checks, "regional_tail_replay", array(state[0, start:]), ref[arm+"_trajectory"][i], 0.)
                    assert digest(array(state[0, :start])) == meta["false_prefix_hashes"][i]
                checked(checks, "regional_score_replay", array(scores), ref[arm+"_scores"], 0.)
                saved.update({arm+"_"+name:value for name, value in captured.items()})
                witnesses[arm] = dict(checks=taylor.reference.checks, witnesses=taylor.reference.witnesses, counts=dict(taylor.reference.counts))
            for label in ("q", "k", "v"):
                checked(checks, "zero_primitive_incoming_mixed_"+label, saved["FF_"+label].astype(float)-saved["FQ_"+label].astype(float)
                    -saved["QF_"+label].astype(float)+saved["QQ_"+label].astype(float), np.zeros_like(saved["QQ_"+label]), 0.)
            rawstar = (saved["FQ_raw"].astype(float)+saved["QF_raw"].astype(float)-saved["QQ_raw"].astype(float)).astype(np.float32)
            logits = torch.tensor(rawstar[None], device="mps")+torch.tensor(saved["FF_mask"][None], device="mps")
            saved["pstar"] = array(F.softmax(logits, dim=-1, dtype=torch.float32)[0])
            counts["extra_softmax"] += 1
            p = {a:saved[a+"_p"].astype(float) for a in ARMS}
            v = {a:np.repeat(saved[a+"_v"].astype(float), int(saved[a+"_groups"]), axis=0) for a in ARMS}
            saved["QK_probability"] = p["FF"]-saved["pstar"]
            saved["SM_probability"] = saved["pstar"]-p["FQ"]-p["QF"]+p["QQ"]
            saved["QK"] = saved["QK_probability"]@v["FF"]
            saved["SM"] = saved["SM_probability"]@v["FF"]
            saved["PV"] = (p["FQ"]-p["QQ"])@(v["QF"]-v["QQ"])+(p["QF"]-p["QQ"])@(v["FQ"]-v["QQ"])
            counts["component_value_products"] += 4
            saved["IY"] = sum(sign*saved[a+"_y"].astype(float) for sign,a in ((1,"FF"),(-1,"FQ"),(-1,"QF"),(1,"QQ")))
            saved["E"] = saved["IY"]-saved["QK"]-saved["SM"]-saved["PV"]
            checked(checks, "explicit_product_rounding", saved["E"], np.zeros_like(saved["E"]), 1e-4)
            for edit in EDITS:
                qwen.eager_attention_forward = attention
                try:
                    states, context, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
                finally:
                    qwen.eager_attention_forward = original
                counts["primitive_intervention_forwards"] += 1
                for i, state in enumerate(states):
                    assert digest(array(state[0, :start])) == meta["false_prefix_hashes"][i]
                checked(checks, "unchanged_block1_"+edit, array(states[0][0, start:]), ref["FF_trajectory"][0], 0.)
                saved[edit+"_final"], saved[edit+"_scores"] = array(states[-1][0, start:]), array(scores)
                assert np.isfinite(saved[edit+"_final"]).all() and np.isfinite(saved[edit+"_scores"]).all()
        assert shutil.disk_usage(out).free > 256*1024**2
        dest = out / "cases" / (case["key"]+".npz")
        np.savez_compressed(dest, **saved)
        write_json(dest.with_suffix(".json"), dict(key=case["key"], group=case["group"], arrays_sha256=sha(dest),
            prior_regional_sha256=sha(rpath), prior_forecast_sha256=sha(fpath), checks=checks, qualifications=witnesses))
        entries.append(dict(key=case["key"], arrays_sha256=sha(dest), record_sha256=sha(dest.with_suffix(".json"))))
        write_json(out / "progress.json", dict(cases=number, counts=dict(counts)))
        PROGRESS.clear()
        print(f"first mixing {number}/32", flush=True)
    assert counts["native_replays"]==32 and counts["regional_replays"]==96 and counts["primitive_intervention_forwards"]==224
    assert counts["primitive_installations"]==224
    write_json(out / "status.json", dict(status="complete", cases=entries, counts=dict(counts)))
    write_json(out / "seal.json", dict(status="sealed", created_utc=stamp(), cases=entries,
        status_sha256=sha(out / "status.json"), design_sha256=sha(out / "design.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--input-244", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_242.resolve(), args.input_244.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out / "partial.npz", **PROGRESS["data"])
                write_json(args.out / "partial.json", {k:v for k,v in PROGRESS.items() if k != "data"})
            write_json(args.out / "failure.json", dict(error=repr(exc), recorded_utc=stamp()))
        raise
