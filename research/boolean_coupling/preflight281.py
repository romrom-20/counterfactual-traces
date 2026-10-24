"""Synthetic source2 product controls; no checkpoint or study case access."""

import json

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit279 import coefficient_checks
from .composition280 import digest, padded
from .core225b import array, observe
from .products281 import ARMS, Engine, masks, random_writes
from .source279 import derivatives


def main():
    torch.manual_seed(281)
    torch.set_num_threads(1)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=16, intermediate_size=23, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, attention_dropout=0., _attn_implementation='eager')
    model = Qwen2ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for value in model.parameters():
            if value.ndim == 2:
                value.normal_(0, .3)
    case = dict(key='synthetic281', position=3, false_ids=[13, 14, 15, 895, 19, 20], true_ids=[13, 14, 15, 830, 19, 20])
    engine, bank, metadata = Engine(model, case), {}, {}
    column = np.linalg.norm(array(model.model.layers[1].mlp.down_proj.weight).astype(float), axis=0)
    with engine:
        for a in ARMS:
            bank[a], metadata[a], nodes = engine.forward(a)
            if a == 'F':
                false = nodes
            if a == 'B':
                engine.plan, score = masks(case, false[1], nodes[1], column)
                base = {n: nodes[1][n].copy() for n in ('mid', 'gate', 'up')}
            if a == 'P':
                engine.draws = random_writes(case, bank[a]['write'].astype(float) - bank['B']['write'].astype(float))
            if a not in ('F', 'FZ', 'T'):
                for n in base:
                    assert np.array_equal(base[n], nodes[1][n])
    for a in ('D',):
        for n in ('source', 'final', 'scores', 'write'):
            assert np.array_equal(bank[a][n], bank['P'][n])
    for n in ('source', 'final', 'scores', 'write'):
        assert np.array_equal(bank['FZ'][n], bank['F'][n])
    for a in ('A', 'A0', 'A1', 'A2'):
        mask = engine.plan['P'] if a == 'A' else engine.plan['R' + a[-1]]
        product = (bank['C1']['phi'] * base['up']).astype(np.float32)
        product[mask] = (bank['B']['phi'][mask] * base['up'][mask]).astype(np.float32)
        assert digest(product) == metadata[a]['product_hashes'][1]
        projected = array(model.model.layers[1].mlp.down_proj(padded(product, 6, 3, 'cpu'))[0, 3])
        assert np.array_equal(projected, bank[a]['raw_write'])
    for a in ('G0', 'G1', 'G2'):
        assert np.array_equal(bank[a]['write'], (bank['B']['write'].astype(float) + engine.draws[a]).astype(np.float32))
        assert abs(np.linalg.norm(engine.draws[a]) - np.linalg.norm(bank['P']['write'].astype(float) - bank['B']['write'].astype(float))) < 1e-10
    for a in bank:
        assert metadata[a]['prefix_hashes'] == metadata['F']['prefix_hashes']
    for a, ids in (('F', case['false_ids']), ('T', case['true_ids'])):
        states, _, scores = observe(model, torch.tensor([ids]))
        assert np.array_equal(array(scores), bank[a]['scores'])
        assert np.array_equal(array(states[-1][0, 3:]), bank[a]['final'])
    for r in false:
        coefficient_checks(r['gate'], derivatives(r['gate']))
    z = dict(gate=false[1]['gate'], up=false[1]['up'], phi=false[1]['phi'])
    selected, score = masks(case, false[1], z, column)
    assert np.all(score['P'] == 0) and np.all(score['H'] == 0)
    assert np.array_equal(selected['P'], np.arange(len(selected['P'])))
    print(json.dumps(dict(passed=True, synthetic=True, checkpoint_loaded=False, layers=3, hidden=16, neurons=23,
        zero_edits=True, rank_ties=True, native_controls=True, masked_source_primal=True,
        direct_products_equal_activation_restoration=True, all_removal_controls=True,
        own_input_only=True, primitive_derivatives=True)))


if __name__ == '__main__':
    main()
