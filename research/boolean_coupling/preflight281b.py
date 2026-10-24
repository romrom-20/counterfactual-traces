"""End-to-end synthetic producer fixture for281b; no checkpoint/study inputs."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from . import run281b
from .cache272 import save_json
from .composition280 import Engine as OldEngine
from .run281b import main as produce


def main(directory):
    assert not directory.exists()
    directory.mkdir()
    reference = directory / 'reference'
    reference.mkdir()
    (reference / 'cases').mkdir()
    torch.manual_seed(2811)
    torch.set_num_threads(1)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=16, intermediate_size=23, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, attention_dropout=0., _attn_implementation='eager')
    model = Qwen2ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for v in model.parameters():
            if v.ndim == 2:
                v.normal_(0, .3)
    case = dict(key='synthetic281b', group={'schema': 'generic', 'reverse': 0}, position=3,
        false_ids=[13, 14, 15, 895, 19, 20], true_ids=[13, 14, 15, 830, 19, 20])
    values = {}
    with OldEngine(model, case) as engine:
        for a in ('F', 'C', 'C1', 'T'):
            packed, _, _ = engine.forward(a)
            for n, v in packed.items():
                values[a + '_' + n] = v
    np.savez_compressed(reference / 'cases/synthetic281b.npz', **values)
    save_json(reference / 'design.json', dict(experiment=280, cases=[case]))
    save_json(reference / 'status.json', dict(status='complete'))
    save_json(reference / 'audit.json', dict(status='passed', synthetic_reference=True))
    # Substitute ONLY environment/checkpoint loading, not producer packing/control logic.
    run281b.load_model = lambda offline: model
    run281b.locate = lambda root: (None, {'synthetic_checkpoint': True})
    run281b.runtime = lambda current: dict(synthetic=True, device='cpu', layers=3)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        produce(reference, directory / 'produced')
    (directory / 'producer.log').write_text(output.getvalue())
    status = json.loads((directory / 'produced/status.json').read_text())
    assert len(status['cases']) == 1 and status['counts']['full_forwards'] == 18
    assert status['qualification'] == '281b-baseline-storage-order'
    entry = status['cases'][0]
    assert entry['key'] == case['key']
    with np.load(directory / 'produced/cases/synthetic281b.npz') as actual:
        for n in ('source', 'final', 'scores', 'write'):
            assert np.array_equal(actual['P_' + n], actual['D_' + n])
            assert np.array_equal(actual['F_' + n], actual['FZ_' + n])
    print(json.dumps(dict(passed=True, synthetic=True, checkpoint_loaded=False, real_study_cases=False,
        entire_producer=True, packing=True, archive_replays=True, actual_forecast_count=18,
        baseline_store_before_compare=True, qualification='281b-baseline-storage-order')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
