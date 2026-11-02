"""Unequal-GQA synthetic qualification and coherent wrong-route rejection."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .alias301 import execute as parent_execute
from .audit229 import sha
from .audit302 import finite_mlp, geometry_checks, population, qualify
from .cache272 import save_json
from .core225b import array
from .geometry302 import ARMS, ROOT, SOURCES, components, execute, metrics


def algebra():
    labels = []
    for winner in ("K", "V", "interaction"):
        data = {}
        for arm in ARMS:
            for alias in "AB":
                center = 1.3 if alias == "A" else 2.1
                native = 1.0 if alias == "A" else 2.0
                signal = 3.0 - native if arm == "KV" or arm == winner else native
                for bit in ("F", "T"):
                    tag = f"{arm}_{alias}_{bit}"
                    shift = signal if bit == "T" else 0.0
                    for name, shape in [("source", (5,)), ("emit", (3, 4)), ("final", (3, 5))]:
                        data[tag + "_" + name] = np.full(shape, center + shift)
                    data[tag + "_scores"] = np.array([0.0, center + shift])
        rows = []
        for i in range(8):
            case = dict(key=str(i), group=dict(block=str(i % 4), mention_order=i // 4))
            rr, _ = metrics(data, case)
            rows.extend(rr)
        for arm in ("K", "V"):
            assert population(rows, arm)["passes"] == (8 if winner == arm else 0)
        labels.append(winner)
    return labels


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(302)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=1024,
                hidden_size=48,
                intermediate_size=67,
                num_hidden_layers=3,
                num_attention_heads=6,
                num_key_value_heads=3,
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
    layer = model.model.layers[0]
    w = dict(
        norm=array(layer.post_attention_layernorm.weight).astype(float),
        eps=np.array(model.config.rms_norm_eps),
        o_weight=array(layer.self_attn.o_proj.weight).astype(float),
        embedding=array(model.model.embed_tokens.weight[[895, 830]]).astype(float),
        next_norm=array(model.model.layers[1].input_layernorm.weight).astype(float),
        next_eps=np.array(model.config.rms_norm_eps),
        next_v_weight=array(model.model.layers[1].self_attn.v_proj.weight).astype(float),
        next_v_bias=array(model.model.layers[1].self_attn.v_proj.bias).astype(float),
        width=np.array(8),
    )
    for name in ("gate", "up", "down"):
        w[name + "_weight"] = array(getattr(layer.mlp, name + "_proj").weight).astype(float)
    hw = dict(
        norm=array(model.model.norm.weight).astype(float),
        eps=np.array(model.config.rms_norm_eps),
        head=array(model.lm_head.weight[[895, 830]]).astype(float),
    )
    inv = array(model.model.rotary_emb.inv_freq).astype(float)
    checks, raw, rejected = [], [], []
    for i, length in enumerate((14, 19)):
        rng = np.random.default_rng(302 + i)
        a = rng.integers(4, 400, length).tolist()
        pos = length - 5
        a[pos] = 895
        b = a.copy()
        b[pos - 3 : pos - 1] = [401, 402]
        at, bt = a.copy(), b.copy()
        at[pos] = bt[pos] = 830
        case = dict(
            key=f"tiny302-{i}",
            position=pos,
            alias_start=pos - 3,
            alias_changed=[pos - 3, pos - 2],
            group=dict(block=str(i), mention_order=i, reverse=0, target=0),
            aliases=dict(A=dict(false_ids=a, true_ids=at), B=dict(false_ids=b, true_ids=bt)),
        )
        f, fw, _, _, fm = parent_execute(model, case, "forecast")
        n, nw, _, _, _ = parent_execute(model, case, "native", f, fm)
        parents = {**fw, **nw}
        refs = {**f, **n}
        comp = components(case, parents, inv)
        data, ck, used = execute(model, case, parents, refs, comp, ARMS)
        assert used["full_forwards"] == 48
        checks.extend(ck)
        qualify(case, parents, comp, data, w, hw, checks, raw)
        assert all(r["passes"] for r in checks), [r for r in checks if not r["passes"]][:5]
        # Direct Torch Double module evaluation checks the finite MLP formula.
        for alias in "AB":
            mid = data[f"N_{alias}_T_mid"].astype(float)
            z = torch.tensor(mid, dtype=torch.float64)
            norm = torch.tensor(w["norm"])
            zz = z / torch.sqrt(z.square().mean() + float(w["eps"])) * norm
            direct = torch.nn.functional.silu(
                torch.nn.functional.linear(zz, torch.tensor(w["gate_weight"]))
            ) * torch.nn.functional.linear(zz, torch.tensor(w["up_weight"]))
            expected = torch.nn.functional.linear(direct, torch.tensor(w["down_weight"])).numpy()
            assert np.max(abs(finite_mlp(w, mid) - expected)) < 1e-10
        # Full finite wrong-donor executions remain numerically coherent but must
        # fail the independent declared-route audit, including both bit states.
        for arm, key in [("K", "dk"), ("V", "dv")]:
            bad = {k: v.copy() for k, v in comp.items()}
            bad[key] *= 0
            wrong, _, count = execute(model, case, parents, refs, bad, (arm,), data)
            assert count["full_forwards"] == 4
            local = []
            qualify(case, parents, comp, {**data, **wrong}, w, hw, local, [])
            assert any(not r["passes"] for r in local)
            rejected.append(f"coherent_wrong_{arm}-{i}")
        bad = {k: v.copy() for k, v in comp.items()}
        for g in range(3):
            q = parents["A_F_0_query"].astype(float)[g * 2, 0]
            bad["hidden"][g] = comp["dk"][g] - np.outer(comp["dk"][g] @ q, q) / (q @ q)
        local = []
        geometry_checks(case, parents, bad, local)
        assert any(not r["passes"] for r in local)
        rejected.append("single_query_only_nullspace-" + str(i))
        bad = {k: v.copy() for k, v in comp.items()}
        bad["phase_difference"] *= 1.1
        local = []
        geometry_checks(case, parents, bad, local)
        assert any(not r["passes"] for r in local)
        rejected.append("corrupt_phase_norm-" + str(i))
        np.savez_compressed(out / f"fixture-{i}.npz", **data)
    known = algebra()
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            required_checks=len(checks),
            raw_checks=len(raw),
            raw_failed=sum(r["error"] > r["bound"] for r in raw),
            full_forwards=140,
            unequal_shapes=dict(
                lengths=[14, 19],
                query_heads=6,
                kv_heads=3,
                edited_rows=2,
                head_width=8,
                hidden=48,
                intermediate=67,
            ),
            coherent_route_forwards=16,
            rejected=rejected,
            known_causal_tables=known,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )
    print(
        dict(passed=True, checks=len(checks), tiny_forwards=140, rejected=len(rejected)), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
