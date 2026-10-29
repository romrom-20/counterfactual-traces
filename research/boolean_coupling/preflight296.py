"""Tiny CPU full-phase geometry checks and coherent wrong-map/source corruptions."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit229 import sha
from .audit296 import audit_bundle, mapped, native_sources
from .cache272 import save_json
from .query296 import ROOT, SOURCES, execute, geometry


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.manual_seed(296)
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
    checks = []
    raw = []
    rejected = []
    rng = np.random.default_rng(296)
    for i, length in enumerate([12, 15, 13, 17]):
        f = rng.integers(3, 400, length).tolist()
        pos = length - 5
        f[pos] = 1
        t = f.copy()
        t[pos] = 2
        case = dict(
            key="tiny296-" + str(i),
            group=dict(block=str(i), schema="tiny", reverse=0, target=0),
            position=pos,
            false_ids=f,
            true_ids=t,
        )
        data, wit, ck, counts, meta = execute(model, case, "forecast")
        meta["checks"] = ck
        audit_bundle(case, data, wit, meta, w, "forecast", checks, raw)
        native, nw, nck, ncounts, nmeta = execute(model, case, "native", data, meta)
        nmeta["checks"] = nck
        audit_bundle(case, native, nw, nmeta, w, "native", checks, raw)
        native_sources(case, meta, nmeta, checks)
        assert all(r["passes"] for r in checks), [r for r in checks if not r["passes"]]
        bad = {k: v.copy() for k, v in wit.items()}
        prefix = "A_1_"
        bad[prefix + "q"] = mapped(
            "D", bad[prefix + "current_q"], bad[prefix + "false_q"], case["key"], 1
        )
        # Coherent attention values cannot hide a wrong query map.
        k, v = bad[prefix + "k"], bad[prefix + "v"]
        groups = int(bad[prefix + "groups"])
        scale = float(bad[prefix + "scaling"])
        rawq = (
            bad[prefix + "q"].astype(float)
            @ np.repeat(k, groups, axis=0).astype(float).swapaxes(-1, -2)
            * scale
        )
        from .audit229 import softmax

        pp = softmax(rawq + bad[prefix + "mask"])
        bad[prefix + "raw"] = rawq
        bad[prefix + "prob"] = pp
        bad[prefix + "output"] = pp @ np.repeat(v, groups, axis=0)
        cc = []
        audit_bundle(case, data, bad, meta, w, "forecast", cc, [])
        assert any(not r["passes"] for r in cc)
        rejected.append("map" + str(i))
        wrong = dict(
            meta,
            source_hashes={
                **meta["source_hashes"],
                "A": ["corrupt"] + meta["source_hashes"]["A"][1:],
            },
        )
        cc = []
        native_sources(case, wrong, nmeta, cc)
        assert any(not r["passes"] for r in cc)
        rejected.append("source" + str(i))
    for a in ["A", "D", "B", "R"]:
        z = np.zeros((2, 3, 4), np.float32)
        assert np.array_equal(geometry(a, z, z, "zero", 0), z)
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            model_forwards=40,
            lengths=[12, 15, 13, 17],
            required_checks=len(checks),
            failed_checks=0,
            raw_checks=len(raw),
            raw_failures=sum(not r["passes"] for r in raw),
            rejected=rejected,
            zero_query_bypass=True,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
            scientific_outcomes_not_required=True,
        ),
    )
    print("296 CPU qualification PASS", len(checks), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
