"""Unequal-length tiny CPU models, exact controls and coherent corruption tests."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit229 import sha
from .audit299 import audit_bundle, local, native_sources, scientific
from .cache272 import save_json
from .generator299 import ROOT, SOURCES, execute, sites


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.manual_seed(299)
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
    maps = {"eps": np.array(model.config.rms_norm_eps)}
    for j in sites(model):
        layer = model.model.layers[j]
        for n, t in (
            ("query", layer.self_attn.q_proj.weight),
            ("bias", layer.self_attn.q_proj.bias),
            ("norm", layer.input_layernorm.weight),
            ("output", layer.self_attn.o_proj.weight),
        ):
            maps[f"{j}_{n}"] = t.detach().numpy()
    checks, raw, rejected, total = [], [], [], 0
    counts = []
    rng = np.random.default_rng(299)
    for i, length in enumerate((12, 15, 13, 17)):
        f = rng.integers(3, 400, length).tolist()
        pos = length - 5
        f[pos] = 1
        t = f.copy()
        t[pos] = 2
        case = dict(
            key="tiny299-" + str(i),
            group=dict(block=str(i), schema="tiny", reverse=0, target=0),
            position=pos,
            false_ids=f,
            true_ids=t,
        )
        data, wit, ck, used, meta = execute(model, case, "forecast")
        meta["checks"] = ck
        native, nw, nc, nused, nm = execute(model, case, "native", data, meta)
        nm["checks"] = nc
        total += used["full_forwards"] + nused["full_forwards"]
        counts.append(dict(forecast=used, native=nused))
        np.savez_compressed(
            out / f"fixture-{i}.npz",
            **{"forecast_" + k: v for k, v in data.items()},
            **{"native_" + k: v for k, v in native.items()},
        )
        audit_bundle(case, data, wit, meta, w, "forecast", checks, raw, data, maps)
        audit_bundle(case, native, nw, nm, w, "native", checks, raw, data, maps)
        native_sources(case, meta, nm, checks)
        local(case, nw, maps, checks)
        assert len(scientific(case, data, native)[0]) == 3
        # Replace EVERY downstream field of the C corner with a coherent M
        # corner, including selected outputs. The declared operand must reject it.
        bad = {k: v.copy() for k, v in wit.items()}
        prefix = "C_1_"
        for field in ("input", "normal", "q", "prob", "output"):
            bad[prefix + "C_" + field] = bad[prefix + "M_" + field].copy()
        for field in ("q", "prob", "output"):
            bad[prefix + "selected_" + field] = bad[prefix + "M_" + field].copy()
        cc = []
        audit_bundle(case, data, bad, meta, w, "forecast", cc, [], data, maps)
        assert any(not r["passes"] for r in cc), "coherent wrong operand accepted"
        rejected.append("coherent_wrong_operand-" + str(i))
        bad = {k: v.copy() for k, v in wit.items()}
        bad["C_1_fm"] += 0.2
        cc = []
        audit_bundle(case, data, bad, meta, w, "forecast", cc, [], data, maps)
        assert any(not r["passes"] for r in cc)
        rejected.append("false_parent_corruption-" + str(i))
        wrong = dict(
            meta,
            source_hashes={
                **meta["source_hashes"],
                "C": ["corrupt"] + meta["source_hashes"]["C"][1:],
            },
        )
        cc = []
        native_sources(case, wrong, nm, cc)
        assert any(not r["passes"] for r in cc)
        rejected.append("source_corruption-" + str(i))
    failed = [r for r in checks if not r["passes"]]
    save_json(out / "checks.json", dict(checks=checks, raw=raw))
    assert not failed, failed[:20]
    save_json(out / "progress-0001.json", dict(cases=1))
    save_json(out / "progress-0002.json", dict(cases=2))
    try:
        save_json(out / "progress-0001.json", dict(cases=999))
    except AssertionError:
        rejected.append("immutable_progress_overwrite")
    else:
        raise AssertionError("immutable writer accepted overwrite")
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            model_forwards=total,
            counts=counts,
            unequal_lengths=[12, 15, 13, 17],
            checks=len(checks),
            raw_checks=len(raw),
            raw_failed=sum(not r["passes"] for r in raw),
            rejected=rejected,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )
    print(dict(passed=True, checks=len(checks), tiny_forwards=total, rejected=len(rejected)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
