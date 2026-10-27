"""Tiny CPU qualification of cached gradients, full-model gradients and 286 audit."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from . import audit286, preflight285, run286
from .cache272 import save_json
from .core225b import array
from .readout286 import Suffix, source_write


def tiny():
    torch.manual_seed(285)
    torch.set_num_threads(1)
    cfg = Qwen2Config(
        vocab_size=1000,
        hidden_size=16,
        intermediate_size=23,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim == 2:
                p.normal_(0, 0.2)
    return model


def full_gradient(model, case, base, anchor, false_gate, mwrite):
    point = base[0, 0].detach().clone().requires_grad_(True)
    start = case["position"]

    def install(module, args, output):
        return torch.cat(
            (output[:, :start].detach(), point[None, None], base[:, 1:].detach()), dim=1
        )

    handle = model.model.layers[0].register_forward_hook(install)
    values, handles = [], []

    def capture(module, args):
        values.append(args[0])

    handles += [layer.register_forward_pre_hook(capture) for layer in model.model.layers[1:]]
    policy = nullcontext() if anchor == "F" else source_write(model, start, false_gate, mwrite)
    ids = torch.tensor([case["false_ids"] if anchor == "F" else case["true_ids"]])
    try:
        with policy, torch.enable_grad():
            result = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            )
            scores = result.logits[0, -1, [895, 830]]
            results = torch.autograd.grad(scores[1] - scores[0], [point, *values])
            grad = results[0]
        return array(grad), array(scores), np.stack([array(g[0, start]) for g in results[1:]])
    finally:
        handle.remove()
        for h in handles:
            h.remove()


def main(out, reference):
    assert not out.exists()
    out.mkdir()
    assert reference.is_dir()
    assert run286.read(reference / "native-audit.json")["status"] == "passed"
    model = tiny()
    run286.main(reference, out / "complete", model=model, sentinel=True)
    audit286.main(out / "complete", reference)
    checks = []
    selected = [
        c
        for c in preflight285.cases()
        if c["group"]["schema"] == "generic" and c["group"]["reverse"] == c["group"]["target"] == 0
    ]
    for case in selected:
        with np.load(reference / "forecasts" / (case["key"] + ".npz")) as saved:
            mwrite = saved["M_write"].copy()
        engine = Suffix(model, case)
        bases = {"F": engine.false_tail, "M": engine.first(mwrite)}
        for anchor, base in bases.items():
            cached, cached_scores, _, trace, _ = engine.gradient(base)
            full, scores, full_trace = full_gradient(
                model, case, base, anchor, engine.false_nodes["gate"], mwrite
            )
            error = float(np.max(abs(full.astype(float) - cached.astype(float))))
            head_error = float(np.max(abs(scores.astype(float) - cached_scores.astype(float))))
            trace_error = float(np.max(abs(trace.astype(float) - full_trace.astype(float))))
            assert trace_error <= 1e-4
            assert error <= 1e-4 and head_error <= 1e-4, (case["key"], anchor, error, head_error)
            checks.append(
                dict(
                    key=case["key"],
                    anchor=anchor,
                    full_vs_cached_gradient_error=error,
                    full_vs_cached_score_error=head_error,
                    source_trace_error=trace_error,
                    bound=0.0001,
                )
            )
        engine.unchanged()
    save_json(out / "full-gradient-checks.json", dict(checks=checks, synthetic=True))
    status = run286.read(out / "complete/status.json")
    assert status["counts"] == dict(full_F=4, first_block=4, suffix_forwards=208, backwards=8)
    summary = dict(
        passed=True,
        synthetic=True,
        checkpoint_loaded=False,
        cases=4,
        qualified_285_reference_reused=True,
        all_286_numeric_and_saved_audit_checks=True,
        cached_vs_full_gradient_all_coordinates=True,
        sampled_FD_directions=True,
        source_clamp_unchanged=True,
        native_outcomes_not_required=True,
        counts=status["counts"],
    )
    save_json(out / "preflight-summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-285-fixture", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve(), a.reference_285_fixture.resolve())
