"""Retrospective native-attention SiLU factorial with native finite attention."""

import argparse
import json
import platform
import shutil
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, load_model, write_json
from .primitive242b import Taylor, Unstable, intervene
from .reference242b import digest

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("QQ", "FQ", "QF", "FF")
SOURCES = [ROOT / p for p in (
    "research/boolean_coupling/native244.py", "research/boolean_coupling/audit244.py",
    "research/boolean_coupling/primitive242b.py", "research/boolean_coupling/reference242b.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/audit242b.py",
    "research/boolean_coupling/core225b.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/settings.json", "research/boolean_coupling/cases242.json",
    "docs/experiments/244-native-attention-regions.md")]
PROGRESS = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def bank_from(data, meta, arm):
    bank = {}
    for label, kind in meta["anchor_types"].items():
        prefix = "anchor_" + label + "_"
        e = {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}
        e["kind"] = kind
        if kind == "silu" and arm in ("FQ", "QF"):
            pick = slice(1, None) if arm == "FQ" else slice(0, 1)
            e = {k: v[pick] if k in ("x", "y") else v for k, v in e.items()}
        bank[label] = e
    return bank


@contextmanager
def region(model, start, taylor, arm, counts, checks):
    handles = []
    lo, hi = (0, None) if arm == "QQ" else (1, None) if arm == "FQ" else (0, 1)

    def replacement(index):
        def change(module, args, output):
            current = array(args[0][0, start:])
            public = array(output[0, start:])
            checked(checks, "current_public_SiLU_primal", current.astype(float) / (1 + np.exp(-current.astype(float))), public, 1e-4)
            counts["finite_SiLU_primal_checks"] += 1
            if arm == "FF":
                return output
            counts["approximated_primitive_attempts"] += 1
            proposed = taylor.propose(f"g{index}", current[lo:hi])
            value = output.clone()
            end = None if hi is None else start + hi
            value[:, start+lo:end] = torch.tensor(proposed[None], device="mps")
            checked(checks, "installed_selected_rows", array(value[0, start+lo:end]), proposed, 0.)
            checked(checks, "unchanged_prefix_rows", array(value[0, :start]), array(output[0, :start]), 0.)
            if arm == "FQ":
                checked(checks, "unchanged_finite_source", array(value[0, start]), public[0], 0.)
            elif arm == "QF":
                checked(checks, "unchanged_finite_receivers", array(value[0, start+1:]), public[1:], 0.)
            counts["approximated_primitive_calls"] += 1
            return value
        return change

    # 111 leaves every primitive finite; regional hooks below replace Q SiLU rows.
    with intervene(model, start, taylor, "111", counts):
        handles.extend(layer.mlp.act_fn.register_forward_hook(replacement(i)) for i, layer in enumerate(model.model.layers))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


def execute(model, ids, start, taylor, arm, false, prefix, counts, checks):
    trajectory, handles = [], []
    taylor.arm = arm

    def recorder(index):
        def save(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            value = array(state[0, start:])
            trajectory.append(value)
            counts["block_outputs"] += 1
            assert digest(array(state[0, :start])) == prefix[index], ("prefix", arm, index)
            checked(checks, "causal_prefix_hash", np.array(0.), np.array(0.), 0.)
            if not np.isfinite(value).all():
                raise Unstable(f"block{index+1}", "nonfinite_state", np.inf, None, value)
            ratio = float(np.linalg.norm(value.astype(float)) / max(.001, np.linalg.norm(false[index].astype(float))))
            if ratio > 100:
                raise Unstable(f"block{index+1}", "state_norm_cap", ratio, 100., value)
        return save

    handles.extend(layer.register_forward_hook(recorder(i)) for i, layer in enumerate(model.model.layers))
    try:
        with region(model, start, taylor, arm, counts, checks), torch.no_grad():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        return np.stack(trajectory), array(result.logits[0, -1, [895, 830]]), None
    except Unstable as exc:
        return np.stack(trajectory) if trajectory else np.empty((0, *false.shape[1:]), np.float32), None, exc
    finally:
        for h in handles:
            h.remove()


def main(previous, regional, out):
    assert not out.exists(), "Never overwrite evidence."
    old_design, old_audit, old_status = (read(previous / n) for n in ("design.json", "audit.json", "forecast-status.json"))
    assert old_design["experiment"] == "242b" and old_audit["status"] == "passed"
    for name, h in old_design["source_hashes"].items():
        assert sha(ROOT / name) == h
    assert old_audit["outcomes_sha256"] == sha(previous / "outcomes.csv")
    _, checkpoint = locate(ROOT)
    assert checkpoint == old_design["checkpoint"]
    assert sha(Path(qwen.__file__)) == old_design["qwen_source_sha256"]
    out.mkdir(parents=True)
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    for p in SOURCES:
        (out / "sources" / p.name).write_bytes(p.read_bytes())
    write_json(out / "design.json", dict(experiment=244, retrospective=True, arms=ARMS, primary="interaction_portability",
        input_regional_design_sha256=sha(regional / "design.json"), input_regional_audit_sha256=sha(regional / "audit.json"),
        input_regional_status_sha256=sha(regional / "status.json"), prior_regional_run=str(regional),
        source_hashes=hashes, input_audit_sha256=sha(previous / "audit.json"), input_design_sha256=sha(previous / "design.json"),
        input_forecast_status_sha256=sha(previous / "forecast-status.json"), checkpoint=checkpoint,
        input_native_status_sha256=sha(previous / "status.json"), input_weights_sha256=sha(previous / "weights.npz"),
        prior_run=str(previous), qwen_source_sha256=sha(Path(qwen.__file__)), created_utc=stamp(),
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    model = load_model(True)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", coefficient_dtype="cpu_float64", revision=model.config._commit_hash, attention="eager", training=model.training)
    assert runtime == read(previous / "runtime.json")
    write_json(out / "runtime.json", runtime)
    cases = read(ROOT / "research/boolean_coupling/cases242.json")
    counts, qualifications, entries = Counter(), Counter(), []
    for number, (case, ent) in enumerate(zip(cases, old_status["cases"], strict=True), 1):
        key, start = case["key"], case["position"]
        assert ent["key"] == key
        path = previous / "forecasts" / (key + ".npz")
        meta = read(path.with_suffix(".json"))
        assert sha(path) == meta["arrays_sha256"] == ent["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == ent["record_sha256"]
        data, checks, records = {}, [], {}
        PROGRESS.clear()
        PROGRESS.update(key=key, data=data, checks=checks)
        with np.load(path) as old:
            false = old["false_trajectory"]
            for arm in ("zero", *ARMS):
                policy = "QQ" if arm == "zero" else arm
                taylor = Taylor(bank_from(old, meta, policy))
                PROGRESS["reference"] = taylor.reference
                ids = case["false_ids" if arm == "zero" else "true_ids"]
                counts["zero_edit_controls" if arm == "zero" else "ordinary_true_calls" if arm == "FF" else "intervention_forwards"] += 1
                trajectory, scores, failure = execute(model, torch.tensor([ids], device="mps"), start, taylor,
                    policy, false, meta["false_prefix_hashes"], counts, checks)
                data[arm + "_trajectory"] = trajectory
                if failure is None:
                    data[arm + "_scores"] = scores
                    counts["complete_forwards"] += 1
                    if arm in ("zero", "QQ"):
                        stem = {"zero": "false", "QQ": "110"}[arm]
                        checked(checks, arm + "_prior_trajectory_replay", trajectory, old[stem + "_trajectory"], 0.)
                        checked(checks, arm + "_prior_score_replay", scores, old[stem + "_scores"], 0.)
                    state = dict(status="complete", completed_blocks=len(trajectory))
                else:
                    assert arm not in ("zero", "QQ", "FF"), "Required replay cannot be unstable."
                    data[arm + "_failed_values"] = failure.raw
                    counts["unstable_forwards"] += 1
                    state = dict(status="unstable", completed_blocks=len(trajectory), failure=failure.record)
                state.update(reference_checks=taylor.reference.checks, witnesses=taylor.reference.witnesses,
                    qualification_counts=dict(taylor.reference.counts), softmax_statistics=taylor.statistics)
                records[arm] = state
                qualifications.update(taylor.reference.counts)
                del taylor
            for a, b in (("QQ", "QF"), ("FQ", "FF")):
                n = min(len(data[a+"_trajectory"]), len(data[b+"_trajectory"]))
                checked(checks, "receiver_cannot_change_source_" + a + "_" + b,
                    data[a+"_trajectory"][:n, 0], data[b+"_trajectory"][:n, 0], 0.)
            for a, b in (("QQ", "FQ"), ("QF", "FF")):
                if len(data[a+"_trajectory"]) and len(data[b+"_trajectory"]):
                    checked(checks, "source_MLP_cannot_change_same_block_receivers_" + a + "_" + b,
                        data[a+"_trajectory"][0, 1:], data[b+"_trajectory"][0, 1:], 0.)
        assert shutil.disk_usage(out).free > 256 * 1024**2, "Preserve evidence; less than256MiB free."
        dest = out / "cases" / (key + ".npz")
        np.savez_compressed(dest, **data)
        write_json(dest.with_suffix(".json"), dict(key=key, group=case["group"], start=start, arrays_sha256=sha(dest),
            prior_forecast_sha256=sha(path), prior_record_sha256=sha(path.with_suffix(".json")), checks=checks, arms=records,
            recorded_utc=stamp(), ordinary_true_calls=1))
        entries.append(dict(key=key, arrays_sha256=sha(dest), record_sha256=sha(dest.with_suffix(".json"))))
        write_json(out / "progress.json", dict(cases=number, counts=dict(counts)))
        PROGRESS.clear()
        print(f"native-attention SiLU factorial {number}/32", flush=True)
    assert counts["zero_edit_controls"] == 32 and counts["intervention_forwards"] == 96 and counts["ordinary_true_calls"] == 32
    assert counts["complete_forwards"] + counts["unstable_forwards"] == 160
    write_json(out / "status.json", dict(status="complete", cases=entries, counts=dict(counts), qualifications=dict(qualifications), ordinary_true_calls=32))
    write_json(out / "intervention-seal.json", dict(status="sealed", retrospective=True, created_utc=stamp(),
        design_sha256=sha(out / "design.json"), status_sha256=sha(out / "status.json"), cases=entries,
        source_hashes=hashes, native_arrays_opened_by_runner=False, ordinary_true_calls=32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-242", type=Path, required=True)
    parser.add_argument("--input-243", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.input_242.resolve(), args.input_243.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                np.savez_compressed(args.out / "partial.npz", **PROGRESS["data"])
                ref = PROGRESS.get("reference")
                write_json(args.out / "partial.json", dict(key=PROGRESS["key"], checks=PROGRESS["checks"],
                    reference_checks=ref.checks if ref else [], witnesses=ref.witnesses if ref else []))
            write_json(args.out / "failure.json", dict(error=repr(exc), recorded_utc=stamp()))
        raise
