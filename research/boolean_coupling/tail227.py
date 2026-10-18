"""Finite MLP24 mixed-operation decomposition and product interventions."""

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

from .core225b import HERE, array, capture_mlp, check, load_model, sha, write_json
from .tail226b import block, full_state, head, stored

ROOT = HERE.parents[1]
SOURCES = [Path(__file__), HERE / "audit227.py", HERE / "tail226b.py",
           HERE / "core225b.py", HERE / "cases.json", HERE / "settings.json",
           ROOT / "docs/experiments/227-mlp24-mixed-primitives.md"]
EXPECTED = dict(reference_blocks=128, local_mlps=96, down_projections=320,
                continuation_blocks=1280, heads=320)


def local(layer, forward, start, checks, norm=None, phi=None):
    saved, handles = {}, []

    def normalized(module, args, output):
        value = output.clone()
        if norm is not None:
            value[:, start:] = torch.tensor(norm[None], device="mps", dtype=torch.float32)
        check(checks, "norm_prefix", (value[:, :start]-output[:, :start]).abs().max(), 0.)
        saved["norm"] = array(value[0, start:])
        return value

    def silu(module, args, output):
        value = output.clone()
        if phi is not None:
            value[:, start:] = torch.tensor(phi[None], device="mps", dtype=torch.float32)
        check(checks, "phi_prefix", (value[:, :start]-output[:, :start]).abs().max(), 0.)
        return value

    handles.append(layer.post_attention_layernorm.register_forward_hook(normalized))
    handles.append(layer.mlp.act_fn.register_forward_hook(silu))
    try:
        with capture_mlp(layer) as captured:
            output = forward()
    finally:
        for handle in handles:
            handle.remove()
    mid = captured["mid"]
    saved.update({k: array(v[0, start:]) for k, v in captured.items()})
    saved["out"] = array((mid+captured["write"])[0, start:])
    return output, mid, saved, captured["product"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/baseline-interaction-225b")
    parser.add_argument("--tail", type=Path, default=ROOT / "runs/finite-tail-interaction-226b")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mlp24-mixed-primitives-227")
    args = parser.parse_args()
    out, old, tail = args.out.resolve(), args.input.resolve(), args.tail.resolve()
    assert not out.exists()
    cases = json.loads((HERE / "cases.json").read_text())
    inputs = [tail / "audit.json", tail / "design.json", tail / "status.json"]
    for case in cases:
        inputs.extend(old / phase / (case["key"]+suffix)
                      for phase in ("qualification", "interventions") for suffix in (".npz", ".json"))
        inputs.extend(tail / "cases" / (case["key"]+suffix) for suffix in (".npz", ".json"))
    out.mkdir(parents=True)
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    for p in SOURCES:
        (out / "sources" / p.name).write_bytes(p.read_bytes())
    write_json(out / "design.json", dict(experiment=227, retrospective=True, created_unix=time(),
        sources={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
        inputs={str(p): sha(p) for p in inputs}, contexts=[c["key"] for c in cases],
        expected=EXPECTED, thresholds=dict(guard=.10, margin_ratio=.5, vector_ratio=.75,
        stable_descriptive=12, stable_per_order=6), derivative_claim=False))
    model = load_model(True)
    layer = model.model.layers[23]
    write_json(out / "runtime.json", dict(python=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        device="mps", dtype="float32", attention="eager", revision=model.config._commit_hash))
    np.savez_compressed(out / "weights.npz", down=array(layer.mlp.down_proj.weight),
        norm=array(model.model.norm.weight), head=array(model.lm_head.weight[[895, 830]]),
        eps=np.array(model.model.norm.variance_epsilon))
    counts = Counter({k: 0 for k in EXPECTED})
    with torch.no_grad():
        for number, case in enumerate(cases, 1):
            key, start = case["key"], case["position"]+1
            q = stored(old / "qualification" / (key+".npz"))
            t = stored(old / "interventions" / (key+".npz"))
            previous = stored(tail / "cases" / (key+".npz"))
            paths = dict(zero=q["s23_b0_trajectory"], b=q["s23_b1_trajectory"],
                         d=t["s23_K_only_trajectory"], both=q["s23_s11_trajectory"])
            context = dict(mask=torch.tensor(q["mask"], device="mps"),
                rope=tuple(torch.tensor(q[k], device="mps") for k in ("rope_cos", "rope_sin")))
            reference = q["native_tail"][1:]
            checks, data, mids, captured = [], {}, {}, {}
            for name, trajectory in paths.items():
                state = full_state(reference[0], trajectory[0], start)
                output, mid, capture, live_product = local(layer, lambda: block(model, 23, state, context), start, checks)
                counts["reference_blocks"] += 1
                check(checks, "reference_"+name, np.max(abs(array(output[0, start:])-trajectory[1])))
                check(checks, "reference_mid", np.max(abs(array(mid[0, start:])-previous["mid_"+name][0])))
                mids[name], captured[name] = mid, capture
                if name == "both":
                    both_product = live_product
                data.update({name+"_"+k: v for k, v in capture.items()})
                data["path_"+name] = trajectory
                data["scores_"+name] = previous["scores_"+name]
            add_mid64 = array(mids["b"]).astype(float)+array(mids["d"]).astype(float)-array(mids["zero"]).astype(float)
            add_mid = torch.tensor(add_mid64, device="mps", dtype=torch.float32)
            norm_sum = captured["b"]["norm"].astype(float)+captured["d"]["norm"].astype(float)-captured["zero"]["norm"].astype(float)
            phi_sum = captured["b"]["phi"].astype(float)+captured["d"]["phi"].astype(float)-captured["zero"]["phi"].astype(float)
            for name, norm, phi in (("add", None, None), ("X", norm_sum, None), ("Y", norm_sum, phi_sum)):
                _, _, capture, _ = local(layer, lambda: layer.mlp(layer.post_attention_layernorm(add_mid)),
                                      start, checks, norm=norm, phi=phi)
                counts["local_mlps"] += 1
                captured[name] = capture
                data.update({name+"_"+k: v for k, v in capture.items()})
            def get(name):
                return captured[name]["product"].astype(float)
            terms = dict(N=get("add")-get("X"), S=get("X")-get("Y"),
                         P=get("Y")-get("b")-get("d")+get("zero"))
            terms["all"] = get("add")-get("b")-get("d")+get("zero")
            check(checks, "product_split", np.max(abs(terms["N"]+terms["S"]+terms["P"]-terms["all"])), 1e-9)
            for name, term in terms.items():
                data["R"+name] = term
            recipes = dict(sham=get("both"), reinsert=get("both")-terms["all"]+terms["all"])
            for name, term in terms.items():
                seed = int(hashlib.sha256((key+name+"227").encode()).hexdigest()[:16], 16)
                random = np.random.default_rng(seed).standard_normal(term.shape)
                random *= np.linalg.norm(term)/np.linalg.norm(random)
                data["random_"+name] = random
                check(checks, "random_norm", abs(np.linalg.norm(random)-np.linalg.norm(term)), 1e-9)
                recipes["remove_"+name] = get("both")-term
                recipes["random_"+name] = get("both")-random
            full_product = both_product
            for name, proposed in recipes.items():
                product = full_product.clone()
                product[:, start:] = torch.tensor(proposed[None], device="mps", dtype=torch.float32)
                check(checks, "product_prefix", (product[:, :start]-full_product[:, :start]).abs().max(), 0.)
                write = layer.mlp.down_proj(product)
                counts["down_projections"] += 1
                state = mids["both"]+write
                data[name+"_installed"] = array(product[0, start:])
                data[name+"_write"] = array(write[0, start:])
                trajectory = [array(state[0, start:])]
                for index in range(24,28):
                    state = block(model, index, state, context)
                    counts["continuation_blocks"] += 1
                    trajectory.append(array(state[0, start:]))
                    check(checks, "continuation_prefix", np.max(abs(array(state[0, :start])-reference[index-22, :start])))
                scores = head(model, state, counts)
                data[name+"_trajectory"] = np.stack(trajectory)
                data[name+"_scores"] = scores
                if name in ("sham", "reinsert"):
                    check(checks, "native_final", np.max(abs(trajectory[-1]-paths["both"][-1])))
                    check(checks, "native_head", np.max(abs(scores-previous["scores_both"])), .005)
                if name == "remove_all":
                    check(checks, "226b_bridge", np.max(abs(trajectory[-1]-previous["b24_mlp_remove_trajectory"][-1])))
                    check(checks, "226b_head", np.max(abs(scores-previous["b24_mlp_remove_scores"])), .005)
            p = out / "cases" / (key+".npz")
            np.savez_compressed(p, **data)
            write_json(p.with_suffix(".json"), dict(key=key, group=case["group"],
                arrays_sha256=sha(p), checks=checks))
            write_json(out / "status.json", dict(completed=number, counts=dict(counts)))
            print(number, case["group"], flush=True)
    assert dict(counts) == EXPECTED, counts
    design = json.loads((out / "design.json").read_text())
    assert all(sha(ROOT/p) == h for p,h in design["sources"].items())
    assert all(sha(p) == h for p,h in design["inputs"].items())
    write_json(out / "status.json", dict(status="complete", completed=len(cases), counts=dict(counts)))


if __name__ == "__main__":
    main()
