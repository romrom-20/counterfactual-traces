"""Tiny finite-suffix fixtures and fail-fast native-reference corruption tests."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit229 import sha
from .audit300 import bundle, scientific
from .cache272 import save_json
from .generator299 import execute as parents
from .interaction300 import ROOT, SOURCES, CalibrationFailure, execute


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.manual_seed(300)
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
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.mul_(10)
            layer.self_attn.k_proj.weight.mul_(10)
            layer.mlp.down_proj.weight.mul_(3)
    weights = dict(
        output=model.model.layers[1].self_attn.o_proj.weight.detach().numpy(),
        norm=model.model.norm.weight.detach().numpy(),
        eps=np.array(model.config.rms_norm_eps),
        head=model.lm_head.weight.detach().numpy()[[895, 830]],
    )
    checks, raw, rejected, counts = (
        [],
        [],
        [],
        dict(parent_tiny_forwards=0, diagnostic_tiny_forwards=0, failed_control_forwards=0),
    )
    rng = np.random.default_rng(300)
    for i, length in enumerate((12, 15, 13, 17)):
        f = rng.integers(3, 400, length).tolist()
        s = length - 5
        f[s] = 1
        t = f.copy()
        t[s] = 2
        case = dict(
            key="tiny300-" + str(i),
            position=s,
            false_ids=f,
            true_ids=t,
            group=dict(block=str(i), schema="tiny", reverse=0, target=0),
        )
        fdata, _, fc, used, fmeta = parents(model, case, "forecast")
        fmeta["checks"] = fc
        ref, p, _, nused, nmeta = parents(model, case, "native", fdata, fmeta)
        counts["parent_tiny_forwards"] += used["full_forwards"] + nused["full_forwards"]
        data, ck, used, meta = execute(model, case, p, ref, nmeta)
        counts["diagnostic_tiny_forwards"] += used["full_forwards"]
        meta["checks"] = ck
        bundle(case, data, p, ref, meta, weights, checks, raw)
        assert len(scientific(case, data)[0]) == 4
        bad = {k: v.copy() for k, v in data.items()}
        for suffix in ("final", "trace", "scores"):
            bad["A_" + suffix] = data["R_" + suffix].copy()
        bad["write_A"] = data["write_R"].copy()
        cc = []
        bundle(case, bad, p, ref, meta, weights, cc, [])
        assert any(not r["passes"] for r in cc), "coherent wrong primitive accepted"
        rejected.append("coherent_wrong_primitive-" + str(i))
        bad = {k: v.copy() for k, v in data.items()}
        bad["random_delta"] *= 2
        bad["write_J"] = np.add(bad["write_L"], bad["random_delta"], dtype=np.float32)
        cc = []
        bundle(case, bad, p, ref, meta, weights, cc, [])
        assert any(not r["passes"] for r in cc)
        rejected.append("coherent_wrong_control_norm-" + str(i))
        bad = {k: v.copy() for k, v in ref.items()}
        bad["T_scores"][0] += 1
        try:
            execute(model, case, p, bad, nmeta)
        except CalibrationFailure as error:
            assert error.counts["full_forwards"] == 1 and "T0_final" in error.data
            counts["failed_control_forwards"] += 1
            np.savez_compressed(out / f"stopped-{i}.npz", **error.data)
            rejected.append("bad_reference_stops_after_one-" + str(i))
        else:
            raise AssertionError("bad reference reached conditional interventions")
        np.savez_compressed(out / f"fixture-{i}.npz", **data)
    save_json(out / "checks.json", dict(checks=checks, raw=raw))
    failed = [r for r in checks if not r["passes"]]
    assert not failed, failed[:20]
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            checks=len(checks),
            failed_checks=0,
            raw_checks=len(raw),
            raw_failed=sum(not r["passes"] for r in raw),
            counts=counts,
            rejected=rejected,
            unequal_lengths=[12, 15, 13, 17],
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )
    print(dict(passed=True, checks=len(checks), counts=counts, rejected=len(rejected)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
