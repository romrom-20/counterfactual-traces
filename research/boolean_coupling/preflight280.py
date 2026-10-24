"""Synthetic CPU-only checks for the 280 source policy; never loads a checkpoint."""

import json

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit279 import coefficient_checks
from .composition280 import BASE, REPAIRS, Engine, local, polynomial, repair_choice
from .core225b import array, observe
from .source279 import derivatives


def main():
    torch.manual_seed(280)
    torch.set_num_threads(1)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=16, intermediate_size=23,
                     num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                     max_position_embeddings=64, attention_dropout=0., _attn_implementation='eager')
    model = Qwen2ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if value.ndim == 2:
                value.normal_(0, .3)
    case = dict(key='synthetic280', position=3, false_ids=[13, 14, 15, 895, 19, 20], true_ids=[13, 14, 15, 830, 19, 20])
    engine, bank, metadata = Engine(model, case), {}, {}
    with engine:
        for a in ('F', *BASE, 'T'):
            bank[a], metadata[a], _ = engine.forward(a)
    for n in bank['F']:
        assert np.array_equal(bank['F'][n], bank['FZ'][n]), n
    for a in bank:
        for n in ('prefix_hashes', 'K_prefix', 'V_prefix'):
            assert metadata[a][n] == metadata['F'][n]
    for n in ('source', 'gate', 'up', 'phi', 'write'):
        assert np.array_equal(bank['C'][n][0], bank['C1'][n][0])
    for a, ids in (('F', case['false_ids']), ('T', case['true_ids'])):
        states, _, scores = observe(model, torch.tensor([ids]))
        assert np.array_equal(array(scores), bank[a]['scores'])
        assert np.array_equal(np.stack([array(s[0, 3]) for s in states]), bank[a]['source'])
    finite = []
    for i, layer in enumerate(model.model.layers):
        result = local(layer, bank['C']['gate'][i], bank['C']['up'][i], 6, 3)
        finite.append(result)
        coefficient_checks(bank['F']['gate'][i], derivatives(bank['F']['gate'][i]))
        phi = polynomial({'gate': bank['F']['gate'][i], 'phi': bank['F']['phi'][i]}, bank['C']['gate'][i], 3)
        assert np.array_equal(phi, bank['C']['phi'][i])
    missing = np.stack([r['write'] for r in finite]).astype(float) - bank['C']['write'].astype(float)
    j, draws = repair_choice(case, bank['C'], missing)
    engine.selected, engine.draws = j, draws
    with engine:
        for a in REPAIRS:
            bank[a], metadata[a], _ = engine.forward(a)
            for n in ('before', 'mid', 'gate', 'up'):
                assert np.array_equal(bank[a][n][j], bank['C'][n][j])
            assert np.array_equal(bank[a]['source'][:j], bank['C']['source'][:j])
    assert np.array_equal(bank['R']['write'][j], finite[j]['write'])
    x = (bank['C']['mid'][j] + finite[j]['write']).astype(np.float32)
    assert np.array_equal(bank['R']['source'][j], x)
    for a in ('G0', 'G1', 'G2'):
        assert np.array_equal(bank[a]['write'][j], (bank['C']['write'][j].astype(float) + draws[a]).astype(np.float32))
    assert metadata['C']['current_source_finite_sites'] == 0
    assert metadata['C1']['current_source_finite_sites'] == 2
    assert metadata['R']['current_source_finite_sites'] == 1
    print(json.dumps(dict(passed=True, synthetic=True, checkpoint_loaded=False, layers=3, hidden=16, neurons=23,
        unhooked_controls=True, all_false_zeros=True, causal_prefixes=True, own_input_polynomials=True,
        local_finite_removal=True, matched_random_installs=True, coefficient_derivatives=True)))


if __name__ == '__main__':
    main()
