"""Entire synthetic32-case cold and30+2 resumed producer; no checkpoint access."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from . import run281d
from .audit229 import sha
from .cache272 import save_json
from .composition280 import Engine as OldEngine


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
    cases = [dict(key=f'synthetic281d-{i:02d}', group={'schema': 'generic', 'reverse': 0}, position=3,
        false_ids=[13, 14, 15, 895, 19, 20], true_ids=[13, 14, 15, 830, 19, 20]) for i in range(32)]
    values = {}
    with OldEngine(model, cases[0]) as engine:
        for a in ('F', 'C', 'C1', 'T'):
            packed, _, _ = engine.forward(a)
            values.update({a + '_' + n: v for n, v in packed.items()})
    first = reference / 'cases' / (cases[0]['key'] + '.npz')
    np.savez_compressed(first, **values)
    for case in cases[1:]:
        os.link(first, reference / 'cases' / (case['key'] + '.npz'))
    save_json(reference / 'design.json', dict(experiment=280, cases=cases))
    save_json(reference / 'status.json', dict(status='complete', synthetic=True))
    save_json(reference / 'audit.json', dict(status='passed', synthetic_reference=True))
    run281d.load_model = lambda offline: model
    run281d.locate = lambda root: (None, {'synthetic_checkpoint': True})
    run281d.runtime = lambda current: dict(synthetic=True, device='cpu', layers=3)
    full = directory / 'full'
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        run281d.main(reference, None, full)
    (directory / 'cold.log').write_text(captured.getvalue())
    status = json.loads((full / 'status.json').read_text())
    assert len(status['cases']) == 32 and status['counts']['full_forwards'] == 576
    assert not status['reused_unchanged_cases'] and status['newly_executed_cases'] == 32
    partial = directory / 'partial'
    partial.mkdir()
    (partial / 'cases').mkdir()
    design = json.loads((full / 'design.json').read_text())
    design['qualification'] = '281b-baseline-storage-order'
    save_json(partial / 'design.json', design)
    save_json(partial / 'failure.json', dict(error='AssertionError()', synthetic_fixture=True))
    for case in cases[:30]:
        for suffix in ('.npz', '.json'):
            p = full / 'cases' / (case['key'] + suffix)
            os.link(p, partial / 'cases' / p.name)
    orphan = full / 'cases' / (cases[30]['key'] + '.npz')
    os.link(orphan, partial / 'cases' / orphan.name)
    resumed = directory / 'resumed'
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        run281d.main(reference, partial, resumed)
    (directory / 'resumed.log').write_text(captured.getvalue())
    status2 = json.loads((resumed / 'status.json').read_text())
    assert len(status2['reused_unchanged_cases']) == 30 and status2['newly_executed_cases'] == 2
    for case in cases:
        p = full / 'cases' / (case['key'] + '.npz')
        q = resumed / 'cases' / p.name
        assert sha(p) == sha(q)
        with np.load(q) as data:
            for n in ('source', 'final', 'scores', 'write'):
                assert np.array_equal(data['P_' + n], data['D_' + n])
                assert np.array_equal(data['F_' + n], data['FZ_' + n])
    print(json.dumps(dict(passed=True, synthetic=True, checkpoint_loaded=False, real_study_cases=False,
        cold_cases=32, resumed_reused=30, resumed_new=2, all_array_hashes_exact=True,
        actual_CPU_forwards=612, entire_producer=True, packing=True, archive_replays=True,
        qualification='281d-resume-path-binding')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
