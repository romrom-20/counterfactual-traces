"""Fresh, globally sealed forecasts from uniform False-anchored primitive rules."""

import argparse
import hashlib
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

from .checkpoint238 import locate, read_head
from .core225b import array, check, load_model, observe, sha, write_json
from .primitive242 import ARMS, Taylor, Unstable, capture, intervene
from .reference242 import digest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / p for p in (
    "research/boolean_coupling/run242.py", "research/boolean_coupling/primitive242.py",
    "research/boolean_coupling/reference242.py", "research/boolean_coupling/audit242.py",
    "research/boolean_coupling/fixtures242.py", "research/boolean_coupling/cases242.json",
    "research/boolean_coupling/historical-input-hashes-242.json", "research/boolean_coupling/core225b.py",
    "research/boolean_coupling/audit229.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/settings.json", "docs/experiments/242-uniform-primitive-families.md")]
PROGRESS = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_check(out):
    design = json.loads((out / "design.json").read_text())
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(out / "sources" / Path(name).name) == h
    assert sha(Path(qwen.__file__)) == design["qwen_source_sha256"]
    return design


def pack_bank(bank, data):
    for label, e in bank.items():
        for name, value in e.items():
            if name != "kind":
                data["anchor_" + label + "_" + name] = value


def run_arm(model, ids, start, taylor, arm, false_tail, prefix, counts, checks):
    trajectory, handles = [], []
    taylor.arm = arm

    def recorder(index):
        def record(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            value = array(state[0, start:])
            trajectory.append(value)
            counts["block_outputs"] += 1
            assert digest(array(state[0, :start])) == prefix[index], ("changed_prefix", arm, index)
            check(checks, "forecast_prefix_exact", 0., 0.)
            if not np.isfinite(value).all():
                raise Unstable(f"block{index + 1}", "nonfinite_state", np.inf, None, value)
            ratio = float(np.linalg.norm(value.astype(float)) / max(.001, np.linalg.norm(false_tail[index].astype(float))))
            if ratio > 100:
                raise Unstable(f"block{index + 1}", "state_norm_cap", ratio, 100., value)
        return record

    handles.extend(layer.register_forward_hook(recorder(i)) for i, layer in enumerate(model.model.layers))
    try:
        with intervene(model, start, taylor, arm, counts), torch.no_grad():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        return np.stack(trajectory), array(result.logits[0, -1, [895, 830]]), None
    except Unstable as exc:
        shape = false_tail.shape[1:]
        return np.stack(trajectory) if trajectory else np.empty((0, *shape), np.float32), None, exc
    finally:
        for h in handles:
            h.remove()


def forecast(out, replication):
    assert not out.exists(), "Never overwrite evidence."
    folder, checkpoint = locate(ROOT)
    out.mkdir(parents=True)
    for name in ("forecasts", "observations", "sources"):
        (out / name).mkdir()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}
    for p in SOURCES:
        (out / "sources" / p.name).write_bytes(p.read_bytes())
    model = load_model(True)
    weights = read_head(folder)
    np.savez_compressed(out / "weights.npz", **weights)
    write_json(out / "design.json", dict(experiment=242, primary="100", arms=ARMS, replication=replication,
        source_hashes=hashes, checkpoint=checkpoint, qwen_source_sha256=sha(Path(qwen.__file__)),
        source_freeze_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        created_utc=stamp(), native_measurements_before_forecast=0, weights_sha256=sha(out / "weights.npz")))
    write_json(out / "runtime.json", dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        transformers=transformers.__version__, device="mps", dtype="float32", coefficient_dtype="cpu_float64",
        revision=model.config._commit_hash, attention="eager", training=model.training))
    cases = json.loads((ROOT / "research/boolean_coupling/cases242.json").read_text())
    counts, qualifications, entries = Counter(), Counter(), []
    for number, case in enumerate(cases, 1):
        key, start = case["key"], case["position"]
        bank, data, checks = {}, {}, []
        PROGRESS.clear()
        PROGRESS.update(key=key, phase="forecast", data=data, checks=checks)
        with capture(model, start, bank):
            states, _, scores = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        counts["ordinary_false"] += 1
        counts["capture_extra_QK_products"] += 28
        prefix = [digest(array(s[0, :start])) for s in states]
        false_tail = np.stack([array(s[0, start:]) for s in states])
        data.update(false_trajectory=false_tail, false_scores=array(scores))
        pack_bank(bank, data)
        del states
        taylor = Taylor(bank)
        taylor.arm = "zero_control"
        PROGRESS["reference"] = taylor.reference
        zero, zero_score, failure = run_arm(model, torch.tensor([case["false_ids"]], device="mps"), start,
            taylor, "000", false_tail, prefix, counts, checks)
        assert failure is None, "The zero-edit control cannot be an unstable hypothesis."
        counts["zero_edit_controls"] += 1
        check(checks, "zero_edit_trajectory_exact", np.max(abs(zero - false_tail)), 0.)
        check(checks, "zero_edit_scores_exact", np.max(abs(zero_score - data["false_scores"])), 0.)
        data.update(zero_trajectory=zero, zero_scores=zero_score)
        for stat in taylor.statistics:
            stat["arm"] = "zero_control"
        records = {}
        for arm in ARMS:
            counts["forecast_arms"] += 1
            trajectory, score, failure = run_arm(model, torch.tensor([case["true_ids"]], device="mps"), start,
                taylor, arm, false_tail, prefix, counts, checks)
            data[arm + "_trajectory"] = trajectory
            if failure is None:
                data[arm + "_scores"] = score
                records[arm] = dict(status="complete", completed_blocks=len(trajectory))
                counts["completed_forecasts"] += 1
            else:
                data[arm + "_failed_values"] = failure.raw
                records[arm] = dict(status="unstable", completed_blocks=len(trajectory), failure=failure.record)
                counts["unstable_forecasts"] += 1
        assert shutil.disk_usage(out).free > 1024**3, "Preserve evidence; less than1GiB free."
        path = out / "forecasts" / (key + ".npz")
        np.savez_compressed(path, **data)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], start=start,
            arrays_sha256=sha(path), checks=checks, reference_checks=taylor.reference.checks,
            qualification_witnesses=taylor.reference.witnesses, qualification_counts=dict(taylor.reference.counts),
            anchor_types={k: e["kind"] for k, e in bank.items()}, false_prefix_hashes=prefix,
            primitive_statistics=taylor.statistics,
            arms=records, recorded_utc=stamp(), ordinary_true=0))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        qualifications.update(taylor.reference.counts)
        write_json(out / "forecast-progress.json", dict(completed=number, counts=dict(counts), qualifications=dict(qualifications)))
        print(f"uniform primitive forecasts {number}/32", {a: r["status"] for a, r in records.items()}, flush=True)
        del taylor, data, bank
        PROGRESS.clear()
    assert counts["ordinary_false"] == counts["zero_edit_controls"] == 32 and counts["forecast_arms"] == 256
    assert counts["completed_forecasts"] + counts["unstable_forecasts"] == 256
    write_json(out / "forecast-status.json", dict(status="complete", cases=entries, counts=dict(counts),
        qualifications=dict(qualifications), ordinary_true=0, completed_utc=stamp()))
    write_json(out / "forecast-manifest.json", dict(cases=entries, source_hashes=hashes, design_sha256=sha(out / "design.json"),
        status_sha256=sha(out / "forecast-status.json"), created_utc=stamp(), ordinary_true=0))


def seal(out):
    assert not (out / "forecast-seal.json").exists()
    design = source_check(out)
    audit = json.loads((out / "pre-measurement-audit.json").read_text())
    assert audit["status"] == "passed" and audit["ordinary_true"] == 0
    write_json(out / "forecast-seal.json", dict(status="sealed", created_utc=stamp(), ordinary_true=0,
        design_sha256=sha(out / "design.json"), manifest_sha256=sha(out / "forecast-manifest.json"),
        pre_audit_sha256=sha(out / "pre-measurement-audit.json"), source_hashes=design["source_hashes"],
        replication=design["replication"]))
    print("Sealed every forecast before native measurements:", sha(out / "forecast-seal.json"), flush=True)


def measure(out, proof_path):
    assert not (out / "status.json").exists() and not list((out / "observations").glob("*.json"))
    design = source_check(out)
    seal_data = json.loads((out / "forecast-seal.json").read_text())
    proof = json.loads(proof_path.read_text())
    assert seal_data["status"] == "sealed" and seal_data["ordinary_true"] == 0
    assert seal_data["design_sha256"] == sha(out / "design.json")
    assert seal_data["manifest_sha256"] == sha(out / "forecast-manifest.json")
    assert seal_data["pre_audit_sha256"] == sha(out / "pre-measurement-audit.json")
    assert seal_data["source_hashes"] == design["source_hashes"] and seal_data["replication"] == design["replication"]
    assert proof["seal_sha256"] == sha(out / "forecast-seal.json")
    assert proof["pre_audit_sha256"] == sha(out / "pre-measurement-audit.json")
    if design["replication"]:
        assert proof["mode"] == "local_replication"
    else:
        assert proof["mode"] == "private_remote" and proof["remote_head"] == proof["commit"]
        remote = subprocess.check_output(["git", "ls-remote", "origin", proof["ref"]], cwd=ROOT, text=True).split()[0]
        assert remote == proof["commit"], "Remote proof must still resolve before the first native measurement."
        stored = subprocess.check_output(["git", "show", proof["commit"] + ":results/uniform-primitive-242/forecast-seal.json"], cwd=ROOT)
        assert hashlib.sha256(stored).hexdigest() == proof["seal_sha256"]
    model, entries = load_model(True), []
    write_json(out / "measurement-proof.json", dict(**proof, accepted_utc=stamp()))
    for number, case in enumerate(json.loads((ROOT / "research/boolean_coupling/cases242.json").read_text()), 1):
        start, key = case["position"], case["key"]
        meta = json.loads((out / "forecasts" / (key + ".json")).read_text())
        started = stamp()
        states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        assert all(digest(array(s[0, :start])) == h for s, h in zip(states, meta["false_prefix_hashes"], strict=True))
        data = dict(true_trajectory=np.stack([array(s[0, start:]) for s in states]), true_scores=array(scores))
        path = out / "observations" / (key + ".npz")
        np.savez_compressed(path, **data)
        write_json(path.with_suffix(".json"), dict(key=key, group=case["group"], arrays_sha256=sha(path),
            forecast_record_sha256=sha(out / "forecasts" / (key + ".json")), native_started_utc=started, completed_utc=stamp(),
            seal_sha256=sha(out / "forecast-seal.json")))
        entries.append(dict(key=key, arrays_sha256=sha(path), record_sha256=sha(path.with_suffix(".json"))))
        print(f"post-seal native measurement {number}/32", flush=True)
    write_json(out / "status.json", dict(status="complete", cases=entries, ordinary_true=32,
        forecast_seal_sha256=sha(out / "forecast-seal.json"), proof_sha256=sha(out / "measurement-proof.json"), replication=design["replication"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "seal", "measure"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--replication", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            forecast(args.out.resolve(), args.replication)
        elif args.phase == "seal":
            seal(args.out.resolve())
        else:
            measure(args.out.resolve(), args.proof.resolve())
    except Exception as exc:
        if args.out.exists():
            if PROGRESS:
                data = PROGRESS["data"]
                np.savez_compressed(args.out / ("partial-" + args.phase + ".npz"), **data)
                reference = PROGRESS.get("reference")
                write_json(args.out / ("partial-" + args.phase + ".json"), dict(key=PROGRESS["key"], checks=PROGRESS["checks"],
                    reference_checks=reference.checks if reference else [], witnesses=reference.witnesses if reference else []))
            write_json(args.out / ("failure-" + args.phase + ".json"), dict(error=repr(exc), recorded_utc=stamp()))
        raise
