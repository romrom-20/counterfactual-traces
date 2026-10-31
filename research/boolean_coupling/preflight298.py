"""Tiny native reconstruction, coherent wrong groups and fail-fast qualification."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit298 import qualify, scientific
from .cache272 import save_json
from .messages298 import ROOT, SOURCES, cpu_weights, roles, run_case
from .publish285 import sha
from .routes297 import execute


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.manual_seed(298)
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
        .requires_grad_(False)
    )
    layer = model.model.layers[-1]
    terminal = SimpleNamespace(
        mlp=layer.mlp,
        middle_norm=layer.post_attention_layernorm,
        final_norm=model.model.norm,
        wo=layer.self_attn.o_proj.weight,
        head=model.lm_head.weight,
        device=model.device,
        layers=3,
    )
    weights = cpu_weights(terminal)
    checks, raw, rejected = [], [], []
    rng = np.random.default_rng(298)
    terminal_calls, model_calls = 0, 0
    for i, length in enumerate([12, 15, 13, 17]):
        f = rng.integers(3, 400, length).tolist()
        start = length - 5
        f[1], f[3], f[4], f[6], f[start] = 997, 998, 999, 999, 895
        t = f.copy()
        t[start] = 830
        schema = "generic" if i % 2 == 0 else "descriptive"
        case = dict(
            key="tiny298-" + str(i),
            group=dict(block=str(i), schema=schema, reverse=0, target=0),
            position=start,
            false_ids=f,
            true_ids=t,
        )
        fd, _, fc, fn, fm = execute(model, case, "forecast")
        fm["checks"] = fc
        nd, nw, _, nn, _ = execute(model, case, "native", fd, fm)
        model_calls += fn["full_forwards"] + nn["full_forwards"]
        parent = dict(
            input=nd["T_trace"][-2],
            expected_final=nd["T_final"][1:],
            expected_scores=nd["T_scores"],
        )
        for name, key in dict(
            head_native="N_output",
            head_false_query="B_output",
            p_native="N_prob",
            p_false_query="B_prob",
            values="v",
            groups="groups",
        ).items():
            parent[name] = nw["T_2_" + key]
        pos = roles(f, start, schema, 999, {997, 998})
        assert pos["chosen"] == ([4, 6] if schema == "generic" else [1, 3])
        assert len(pos["neighbor"]) == len(pos["chosen"]) and not set(pos["neighbor"]) & set(
            pos["union"]
        )
        selection = dict(
            key=case["key"], group=case["group"], position=start, length=length, positions=pos
        )
        data, producer, counts, complete = run_case(terminal, parent, selection)
        terminal_calls += counts["terminal_MLPs"]
        np.savez_compressed(out / (case["key"] + ".npz"), **data)
        assert complete and counts["terminal_MLPs"] == 20
        qualify(selection, parent, data, producer, weights, checks, raw)
        effects, _, joint = scientific(selection, data)
        assert len(effects) == 18 and len(joint) == 9
        assert all(r["passes"] for r in checks), [r for r in checks if not r["passes"]]
        if i == 0:
            wrong = dict(selection, positions={**pos, "chosen": pos["swapped"]})
            bad, badchecks, badcounts, badcomplete = run_case(terminal, parent, wrong)
            terminal_calls += badcounts["terminal_MLPs"]
            assert badcomplete
            cc = []
            qualify(selection, parent, bad, badchecks, weights, cc, [])
            assert any(not r["passes"] for r in cc), (
                "Coherent wrong position group was not rejected"
            )
            rejected.append("coherent_wrong_group")
            badparent = dict(parent, expected_scores=parent["expected_scores"] + 1)
            partial, _, used, complete = run_case(terminal, badparent, selection)
            terminal_calls += used["terminal_MLPs"]
            assert not complete and used["terminal_MLPs"] == 1 and "chosen_add_final" not in partial
            rejected.append("native_calibration_fail_fast")
            aliased = {k: v.copy() for k, v in data.items()}
            for field in ["write", "middle", "final", "scores"]:
                aliased["native_reference_" + field] = aliased["frozen_query_reference_" + field]
            cc = []
            qualify(selection, parent, aliased, producer, weights, cc, [])
            assert any(not r["passes"] for r in cc), "Reference alias corruption not rejected"
            rejected.append("native_reference_alias")
    save_json(out / "progress-0001.json", dict(cases=1))
    save_json(out / "progress-0002.json", dict(cases=2))
    try:
        save_json(out / "progress-0001.json", dict(cases=999))
    except AssertionError:
        rejected.append("immutable_progress_overwrite")
    else:
        raise AssertionError("Immutable progress overwritten")
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            checkpoint_loaded=False,
            synthetic=True,
            full_tiny_forwards=model_calls,
            terminal_MLPs=terminal_calls,
            lengths=[12, 15, 13, 17],
            required_checks=len(checks),
            failed_checks=0,
            raw_checks=len(raw),
            raw_failed=sum(not r["passes"] for r in raw),
            rejected=rejected,
            scientific_outcomes_not_required=True,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )
    print(
        "298 CPU qualification PASS",
        len(checks),
        "checks",
        model_calls,
        "tiny full forwards",
        terminal_calls,
        "terminal MLPs",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
