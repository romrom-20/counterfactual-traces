"""Full qualified audit of the existing32-case synthetic producer; no real model."""

import argparse
import contextlib
import io
import os
import shutil
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM

from . import audit282b, diagnose_input_bridges282, qualify282b
from .audit272 import read
from .cache272 import save_json


def main(fixture, out):
    assert not out.exists()
    out.mkdir()
    original = fixture/'complete'
    assert read(original/'audit.json')['status'] == 'passed'
    run = out/'audit-repetition'
    run.mkdir()
    for name in ('design.json', 'status.json', 'runtime.json'):
        shutil.copy2(original/name, run/name)
    for name in ('sources', 'cases'):
        shutil.copytree(original/name, run/name, copy_function=os.link)
    key = read(run/'status.json')['cases'][0]['key']
    mock = dict(error=repr(AssertionError(dict(key=key, name='CPU64_U_up', error=.000104, bound=.0001))),
        synthetic_fixture_mock=True, not_an_observed_CPU_failure=True)
    bad = dict(error=repr(AssertionError(dict(key=key, name='CPU64_U_gate', error=.000104, bound=.0001))))
    try:
        qualify282b.validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Qualifier accepted a different primitive failure')
    save_json(run/'failure-audit.json', mock)
    checkpoint = fixture/'tiny-checkpoint'
    torch.set_num_threads(1)
    model = Qwen2ForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation='eager').eval()
    model.requires_grad_(False)
    fake = dict(synthetic_checkpoint=True)
    diagnose_input_bridges282.locate = lambda root: (checkpoint, fake)
    audit282b.locate = lambda root: (checkpoint, fake)
    audit282b.load_model = lambda offline: model
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        diagnostic = out/'numerical-diagnostic'
        diagnose_input_bridges282.main(run, fixture/'reference', diagnostic)
        qualify282b.main(run, diagnostic)
        audit282b.main(run, fixture/'reference')
    (out/'audit.log').write_text(captured.getvalue())
    result, previous = read(run/'audit.json'), read(original/'audit.json')
    assert result['status'] == 'passed' and not result['original_audit_passed']
    for name in ('native_counts', 'local_counts', 'effect_counts', 'joint_counts', 'paired_local_counts', 'paired_effect_counts', 'K_strata', 'primary_passes', 'account_counts', 'account_passes'):
        assert result[name] == previous[name], name
    assert result['public_corner_maps'] == 128 and result['public_down_maps'] == 768
    assert result['public_primal_maps'] == 256 and result['coefficient_vectors'] == 192
    print(dict(passed=True, synthetic=True, real_checkpoint_loaded=False, full_model_forwards=0,
        full_32_case_qualified_audit=True, rejects_wrong_primitive_failure=True,
        all_endpoint_counts_unchanged=True, numerical_checks=result['numerical_checks'],
        original_282_CPU_producer_already_passed=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.fixture.resolve(), a.out.resolve())
