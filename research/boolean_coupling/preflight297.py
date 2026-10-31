"""Tiny CPU phases, softmax-gauge checks and coherent corruption rejection."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit229 import sha
from .audit297 import audit_bundle, local, native_sources, scientific
from .cache272 import save_json
from .routes297 import ROOT, SOURCES, execute, probabilities, record, sites


def gauge(checks):
    generator = torch.Generator().manual_seed(297)
    c = torch.randn(1, 4, 5, 11, generator=generator)
    f = torch.randn(1, 4, 5, 11, generator=generator)
    mask = torch.zeros_like(c)
    equal = torch.zeros(1, 4, 5, 1, dtype=torch.bool)
    p = torch.softmax(c, -1)
    reference, _ = probabilities(c, f, mask, p, equal, 4)
    shifted, _ = probabilities(c + 2, f - 3, mask, torch.softmax(c + 2, -1), equal, 4)
    for arm in ["B", "S", "C", "N"]:
        record(
            checks,
            arm + "/constant_logit_offset_invariance",
            reference[arm].numpy(),
            shifted[arm].numpy(),
            2e-7,
        )
    f = torch.zeros_like(f)
    c = f + 2
    same, _ = probabilities(c, f, mask, torch.softmax(c, -1), equal, 4)
    for arm in ["B", "S", "C", "N"]:
        record(
            checks,
            arm + "/uniform_shift_no_route_effect",
            same[arm].numpy(),
            same["B"].numpy(),
            2e-7,
        )
    wrong = f.clone()
    wrong[..., 4] = 2
    assert (torch.softmax(wrong, -1) - same["B"]).abs().max().item() > 0.1


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.manual_seed(297)
    torch.set_num_threads(2)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=1024,
                hidden_size=16,
                intermediate_size=23,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                tie_word_embeddings=False,
                attention_dropout=0.0,
                _attn_implementation="eager",
            )
        )
        .float()
        .eval()
    )
    model.requires_grad_(False)
    w = dict(
        norm=model.model.norm.weight.detach().numpy().astype(float),
        eps=np.array(model.config.rms_norm_eps),
        head=model.lm_head.weight.detach().numpy()[[895, 830]].astype(float),
    )
    maps = {
        str(j): model.model.layers[j].self_attn.o_proj.weight.detach().numpy() for j in sites(model)
    }
    checks, raw, rejected = [], [], []
    gauge(checks)
    rng = np.random.default_rng(297)
    total = 0
    for i, length in enumerate([12, 15, 13, 17]):
        f = rng.integers(3, 400, length).tolist()
        position = length - 5
        f[position] = 1
        t = f.copy()
        t[position] = 2
        case = dict(
            key="tiny297-" + str(i),
            group=dict(block=str(i), schema="tiny", reverse=0, target=0),
            position=position,
            false_ids=f,
            true_ids=t,
        )
        data, wit, ck, counts, meta = execute(model, case, "forecast")
        meta["checks"] = ck
        native, nw, nc, ncounts, nm = execute(model, case, "native", data, meta)
        nm["checks"] = nc
        total += counts["full_forwards"] + ncounts["full_forwards"]
        np.savez_compressed(
            out / f"fixture-{i}.npz",
            **{f"forecast_{k}": v for k, v in data.items()},
            **{f"native_{k}": v for k, v in native.items()},
        )
        audit_bundle(case, data, wit, meta, w, "forecast", checks, raw, data)
        audit_bundle(case, native, nw, nm, w, "native", checks, raw, data)
        native_sources(case, meta, nm, checks)
        local(case, nw, maps, checks)
        assert len(scientific(case, data, native)[0]) == 3
        bad = {k: v.copy() for k, v in wit.items()}
        prefix = "S_1_"
        p = bad[prefix + "S_prob"].astype(float)
        p[..., position] *= 4
        p /= p.sum(-1, keepdims=True)
        vv = np.repeat(bad[prefix + "v"], int(bad[prefix + "groups"]), axis=0)
        bad[prefix + "S_prob"] = bad[prefix + "selected_prob"] = p
        bad[prefix + "S_output"] = bad[prefix + "selected_output"] = p @ vv
        cc = []
        audit_bundle(case, data, bad, meta, w, "forecast", cc, [], data)
        assert any(not r["passes"] for r in cc), "Coherent wrong allocation not rejected"
        rejected.append("coherent_wrong_allocation" + str(i))
        wrong = dict(
            meta,
            source_hashes={
                **meta["source_hashes"],
                "S": ["corrupt"] + meta["source_hashes"]["S"][1:],
            },
        )
        cc = []
        native_sources(case, wrong, nm, cc)
        assert any(not r["passes"] for r in cc)
        rejected.append("source" + str(i))
        assert all(r["passes"] for r in checks), [r for r in checks if not r["passes"]]
    save_json(out / "progress-0001.json", dict(cases=1))
    save_json(out / "progress-0002.json", dict(cases=2))
    try:
        save_json(out / "progress-0001.json", dict(cases=999))
    except AssertionError:
        rejected.append("immutable_progress_overwrite")
    else:
        raise AssertionError("Immutable writer accepted overwrite")
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            model_forwards=total,
            lengths=[12, 15, 13, 17],
            required_checks=len(checks),
            failed_checks=0,
            raw_checks=len(raw),
            raw_failures=sum(not r["passes"] for r in raw),
            rejected=rejected,
            gauge_invariant=True,
            scientific_outcomes_not_required=True,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )
    print("297 CPU qualification PASS", len(checks), "checks", total, "tiny forwards", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
