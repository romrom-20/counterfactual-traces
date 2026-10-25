"""Unequal-length32-case synthetic producer, expected282b failure, qualified audit."""

import argparse
import contextlib
import csv
import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from . import audit282b, audit282c, diagnose_input_bridges282, qualify282b, qualify282c, run282
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .products281 import Engine as ReferenceEngine


def main(out):
    assert not out.exists()
    out.mkdir()
    reference, checkpoint = out/'reference', out/'tiny-checkpoint'
    (reference/'cases').mkdir(parents=True)
    checkpoint.mkdir()
    torch.manual_seed(2823)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=1000, hidden_size=16, intermediate_size=23, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
        attention_dropout=0., _attn_implementation='eager')
    model = Qwen2ForCausalLM(config).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for value in model.parameters():
            if value.ndim == 2:
                value.normal_(0, .3)
    model.config.save_pretrained(checkpoint)
    save_model(model, str(checkpoint/'model.safetensors'))
    cases, entries = [], []
    for i in range(8):
        for reverse in (0, 1):
            for schema in ('generic', 'descriptive'):
                key = hashlib.sha256(f'282unequal:{i}:{reverse}:{schema}'.encode()).hexdigest()
                suffix = [43]+([45] if schema == 'descriptive' else [])+[44]
                case = dict(key=key, position=3,
                    group=dict(assignment=i%2, block='synthetic'+str(i), reverse=reverse, schema=schema, target=i%2),
                    false_ids=[13+i, 23+reverse, 33+int(schema=='descriptive'), 895]+suffix,
                    true_ids=[13+i, 23+reverse, 33+int(schema=='descriptive'), 830]+suffix)
                values = {}
                with ReferenceEngine(model, case) as engine:
                    for arm in ('F', 'B', 'C1', 'T'):
                        packed, _, _ = engine.forward(arm)
                        values.update({arm+'_'+n: v for n, v in packed.items()})
                path = reference/'cases'/(key+'.npz')
                np.savez_compressed(path, **values)
                entries.append(dict(key=key, arrays_sha256=sha(path)))
                cases.append(case)
    save_json(reference/'design.json', dict(experiment=281, cases=cases, source_hashes={}, synthetic_reference=True))
    save_json(reference/'status.json', dict(status='complete', cases=entries, synthetic=True))
    save_json(reference/'audit.json', dict(status='passed', synthetic_reference=True))
    fake = dict(synthetic_checkpoint=True)
    run282.load_model = lambda offline: model
    run282.locate = lambda root: (checkpoint, fake)
    run282.runtime = lambda current: dict(synthetic=True, device='cpu', layers=3)
    diagnose_input_bridges282.locate = lambda root: (checkpoint, fake)
    for module in (audit282b, audit282c):
        module.load_model = lambda offline: model
        module.locate = lambda root: (checkpoint, fake)
    run = out/'complete'
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            run282.main(reference, run)
            save_json(run/'failure-audit.json', dict(
                error=repr(AssertionError(dict(key=cases[0]['key'], name='CPU64_U_up', error=.000104, bound=.0001))),
                synthetic_fixture_mock=True, not_an_observed_CPU_failure=True))
            diagnostic = out/'numerical-diagnostic'
            diagnose_input_bridges282.main(run, reference, diagnostic)
            qualify282b.main(run, diagnostic)
            try:
                audit282b.main(run, reference)
            except ValueError as exc:
                assert str(exc).startswith('operands could not be broadcast together with shapes ')
                save_json(run/'failure-audit-b.json', dict(error=repr(exc), synthetic=True, actually_observed=True))
            else:
                raise AssertionError('Unequal receiving lengths did not reproduce282b failure')
            qualify282c.main(run)
            audit282c.main(run, reference)
    finally:
        (out/'execution.log').write_text(captured.getvalue())
    result = read(run/'audit.json')
    assert result['status'] == 'passed' and result['paired_whole_secondary_evaluable'] == 0
    assert result['coefficient_vectors'] == 192 and result['public_down_maps'] == 768
    def rows(name):
        with (run/(name+'.csv')).open() as stream:
            return list(csv.DictReader(stream))
    pair = {}
    paired = rows('paired')
    assert len(paired) == 208
    for row in paired:
        assert row['original_secondary_effect_evaluable'] == row['effect_pass'] == 'False'
        assert row['generic_receiving_rows'] != row['descriptive_receiving_rows']
        for key in (row['generic_key'], row['descriptive_key']):
            pair[key, row['arm']] = row['local_pass'] == 'True'
    indices = [{(r['key'], r['arm']): r for r in rows(n)} for n in ('native', 'local', 'effect')]
    for row in rows('joint'):
        ix = row['key'], row['arm']
        passed = all(index[ix]['passes'] == 'True' for index in indices) and pair[ix]
        assert (row['passes'] == 'True') == passed
    print(dict(passed=True, synthetic=True, real_checkpoint_loaded=False, reference_CPU_forwards=128,
        new_CPU_forwards=480, unequal_pairs=16, original282b_shape_failure_reproduced=True,
        qualified_32_case_audit=True, primary_decisions_independently_recomputed=True,
        whole_secondary_not_claimed=True, numerical_checks=result['numerical_checks']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out.resolve())
