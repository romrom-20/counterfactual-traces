"""Finite, derivative-free sublayer interaction diagnostic; see protocol 226."""

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from time import time

import numpy as np
import torch
import transformers
from torch.nn import functional as F

from .core225b import HERE, array, capture_mlp, check, load_model, sha, write_json

ROOT = HERE.parents[1]
PROTOCOL = ROOT / "docs/experiments/226-finite-tail-interaction.md"
SOURCES = [Path(__file__), HERE / "audit226.py", HERE / "core225b.py",
           HERE / "cases.json", HERE / "settings.json", PROTOCOL]
EXPECTED = dict(reference_blocks=640, additive_blocks=160, additive_mlps=160,
                intervention_mlps=480, continuation_blocks=1920, heads=1120)


def stored(path):
    record = json.loads(path.with_suffix(".json").read_text())
    assert sha(path) == record["arrays_sha256"]
    assert all(c["error"] <= c["bound"] for c in record["checks"])
    return dict(np.load(path))


def full_state(reference, receiving, start):
    value = torch.tensor(reference[None], device="mps", dtype=torch.float32)
    value[:, start:] = torch.tensor(receiving[None], device="mps", dtype=torch.float32)
    return value


def head(model, state, counts):
    counts["heads"] += 1
    return array(F.linear(model.model.norm(state)[0, -1],
                          model.lm_head.weight[[895, 830]]))


def block(model, index, state, context):
    value = model.model.layers[index](state, attention_mask=context["mask"],
                                      position_embeddings=context["rope"])
    return value[0] if isinstance(value, tuple) else value


def extract_mid(model, index, state, context):
    with capture_mlp(model.model.layers[index]) as captured:
        output = block(model, index, state, context)
    return captured["mid"], output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/baseline-interaction-225b")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/finite-tail-interaction-226")
    args = parser.parse_args()
    out, old = args.out.resolve(), args.input.resolve()
    assert not out.exists(), "Never overwrite an experiment."
    cases = json.loads((HERE / "cases.json").read_text())
    inputs = [old / name for name in ("design.json", "runtime.json", "audit.json",
                                      "qualification-audit.json", "qualification-seal.json", "weights.npz")]
    for case in cases:
        inputs.extend(old / phase / (case["key"] + suffix)
                      for phase in ("qualification", "interventions") for suffix in (".json", ".npz"))
    out.mkdir(parents=True)
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    for source in SOURCES:
        (out / "sources" / source.name).write_bytes(source.read_bytes())
    write_json(out / "design.json", dict(
        experiment=226, created_unix=time(), retrospective=True, model_calls_before_seal=0,
        contexts=[c["key"] for c in cases], expected=EXPECTED,
        sources={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        inputs={str(p): sha(p) for p in inputs},
        thresholds=dict(guard=.10, margin_ratio=.5, vector_ratio=.75,
                        stable_descriptive=12, stable_per_order=6),
        derivative_claim=False))
    model = load_model(True)
    write_json(out / "runtime.json", dict(python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__,
        device="mps", dtype="float32", attention="eager", training=model.training,
        revision=model.config._commit_hash))
    counts = Counter({k: 0 for k in EXPECTED})
    weights = dict(norm=array(model.model.norm.weight),
                   head=array(model.lm_head.weight[[895, 830]]),
                   eps=np.array(model.model.norm.variance_epsilon))
    np.savez_compressed(out / "weights.npz", **weights)
    for key in weights:
        check([], "old_weight_" + key,
              np.max(abs(weights[key].astype(float) - np.load(old / "weights.npz")[key])))
    with torch.no_grad():
        for number, case in enumerate(cases, 1):
            q = stored(old / "qualification" / (case["key"] + ".npz"))
            t = stored(old / "interventions" / (case["key"] + ".npz"))
            start = case["position"] + 1
            context = dict(mask=torch.tensor(q["mask"], device="mps"),
                rope=tuple(torch.tensor(q[k], device="mps") for k in ("rope_cos", "rope_sin")))
            reference = q["native_tail"][2:]  # after blocks 23 .. 28
            paths = dict(zero=q["s23_b0_trajectory"], d=t["s23_K_only_trajectory"],
                         b=q["s23_b1_trajectory"], both=q["s23_s11_trajectory"])
            scores = dict(zero=q["s23_b0_scores"], d=t["s23_K_only_scores"],
                          b=q["s23_b1_scores"], both=q["s23_s11_scores"])
            checks, saved = [], {"path_" + k: v for k, v in paths.items()}
            saved.update({"scores_" + k: v for k, v in scores.items()})
            saved["reference_prefix"] = reference[:, :start]
            # Full finite replay; intermediates are checked BEFORE any causal edits.
            mids = {name: [] for name in paths}
            for name, trajectory in paths.items():
                state = full_state(reference[0], trajectory[0], start)
                for j in range(5):
                    mid, state = extract_mid(model, 23+j, state, context)
                    counts["reference_blocks"] += 1
                    check(checks, "reference_" + name,
                          np.max(abs(array(state[0, start:]) - trajectory[j+1])))
                    check(checks, "reference_prefix",
                          np.max(abs(array(state[0, :start]) - reference[j+1, :start])))
                    mids[name].append(array(mid[0]))
                actual = head(model, state, counts)
                check(checks, "reference_head_" + name, np.max(abs(actual-scores[name])), .005)
            for name in mids:
                saved["mid_" + name] = np.stack(mids[name])[:, start:]
            final_add = (paths["b"][-1].astype(float) + paths["d"][-1].astype(float)
                         - paths["zero"][-1].astype(float))
            final_state = full_state(reference[-1], final_add, start)
            saved["readout_add_final"] = array(final_state[0, start:])
            saved["readout_add_scores"] = head(model, final_state, counts)
            for j in range(5):
                index, block_number = 23+j, 24+j
                incoming = {k: v[j].astype(float) for k, v in paths.items()}
                additive = incoming["b"] + incoming["d"] - incoming["zero"]
                add_state = full_state(reference[j], additive, start)
                check(checks, "additive_prefix",
                      np.max(abs(array(add_state[0, :start])-reference[j, :start])), 0.)
                attn_add, _ = extract_mid(model, index, add_state, context)
                counts["additive_blocks"] += 1
                layer = model.model.layers[index]
                for stage in ("attention", "mlp"):
                    prefix = f"b{block_number}_{stage}_"
                    if stage == "attention":
                        parents = incoming
                        ys = {k: v[j][start:].astype(float) for k, v in mids.items()}
                        gadd = array(attn_add[0, start:]).astype(float)
                        original = torch.tensor(mids["both"][j][None], device="mps")
                        native_prefix = mids["both"][j][:start]
                        add_installed = array(add_state[0, start:])
                    else:
                        parents = {k: v[j][start:].astype(float) for k, v in mids.items()}
                        mid_add = parents["b"] + parents["d"] - parents["zero"]
                        mid_tensor = full_state(mids["both"][j], mid_add, start)
                        gadd = array((mid_tensor + layer.mlp(layer.post_attention_layernorm(mid_tensor)))[0, start:]).astype(float)
                        counts["additive_mlps"] += 1
                        ys = {k: v[j+1].astype(float) for k, v in paths.items()}
                        original = full_state(reference[j+1], paths["both"][j+1], start)
                        native_prefix = reference[j+1, :start]
                        add_installed = array(mid_tensor[0, start:])
                    local = gadd - ys["b"] - ys["d"] + ys["zero"]
                    propagated = ys["both"] - gadd
                    total = ys["both"] - ys["b"] - ys["d"] + ys["zero"]
                    check(checks, "finite_identity", np.max(abs(total-local-propagated)), 1e-9)
                    seed = int(hashlib.sha256((case["key"]+prefix+"226").encode()).hexdigest()[:16], 16)
                    random = np.random.default_rng(seed).standard_normal(local.shape)
                    random *= np.linalg.norm(local) / np.linalg.norm(random)
                    check(checks, "random_norm", abs(np.linalg.norm(random)-np.linalg.norm(local)), 1e-9)
                    for label, value in dict(local=local, propagated=propagated, total=total,
                        gadd=gadd, add_installed=add_installed,
                        incoming_interaction=parents["both"]-parents["b"]-parents["d"]+parents["zero"],
                        random=random).items():
                        saved[prefix+label] = value
                    for arm in ("remove", "random", "reinsert"):
                        proposed = ys["both"] - (random if arm == "random" else local)
                        if arm == "reinsert":
                            proposed += local
                        state = original.clone()
                        state[:, start:] = torch.tensor(proposed[None], device="mps", dtype=torch.float32)
                        check(checks, "installed_prefix",
                              np.max(abs(array(state[0, :start])-native_prefix)), 0.)
                        saved[prefix+arm+"_installed"] = array(state[0, start:])
                        if stage == "attention":
                            state = state + layer.mlp(layer.post_attention_layernorm(state))
                            counts["intervention_mlps"] += 1
                        check(checks, "continuation_prefix",
                              np.max(abs(array(state[0, :start])-reference[j+1, :start])))
                        trajectory = [array(state[0, start:])]
                        for following in range(index+1, 28):
                            state = block(model, following, state, context)
                            counts["continuation_blocks"] += 1
                            trajectory.append(array(state[0, start:]))
                            check(checks, "continuation_prefix",
                                  np.max(abs(array(state[0, :start])-reference[following-22, :start])))
                        result = head(model, state, counts)
                        saved[prefix+arm+"_trajectory"] = np.stack(trajectory)
                        saved[prefix+arm+"_scores"] = result
                        if arm == "reinsert":
                            check(checks, "reinsert_final",
                                  np.max(abs(trajectory[-1]-paths["both"][-1])))
                            check(checks, "reinsert_scores", np.max(abs(result-scores["both"])), .005)
            p = out / "cases" / (case["key"] + ".npz")
            np.savez_compressed(p, **saved)
            write_json(p.with_suffix(".json"), dict(key=case["key"], group=case["group"],
                arrays_sha256=sha(p), checks=checks))
            write_json(out / "status.json", dict(completed=number, counts=dict(counts)))
            print(number, case["group"], "checks", len(checks), flush=True)
    assert dict(counts) == EXPECTED, counts
    design = json.loads((out / "design.json").read_text())
    assert all(sha(ROOT/p) == h for p, h in design["sources"].items())
    assert all(sha(p) == h for p, h in design["inputs"].items())
    write_json(out / "status.json", dict(status="complete", completed=len(cases), counts=dict(counts)))


if __name__ == "__main__":
    main()
