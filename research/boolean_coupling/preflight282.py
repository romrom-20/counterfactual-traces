"""Entire32-case CPU producer AND separate audit; no real checkpoint or cases."""

import argparse
import contextlib
import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from . import audit282, run282
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .products281 import Engine as ReferenceEngine


def main(out):
    assert not out.exists()
    out.mkdir()
    reference, checkpoint = out / 'reference', out / 'tiny-checkpoint'
    reference.mkdir()
    (reference / 'cases').mkdir()
    checkpoint.mkdir()
    torch.manual_seed(2821)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=1000, hidden_size=16, intermediate_size=23, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, attention_dropout=0., _attn_implementation='eager')
    model = Qwen2ForCausalLM(config).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for value in model.parameters():
            if value.ndim == 2:
                value.normal_(0, .3)
    model.config.save_pretrained(checkpoint)
    save_model(model, str(checkpoint / 'model.safetensors'))
    cases, entries = [], []
    for i in range(8):
        for reverse in (0, 1):
            for schema in ('generic', 'descriptive'):
                key = hashlib.sha256(f'282synthetic:{i}:{reverse}:{schema}'.encode()).hexdigest()
                case = dict(key=key, position=3, group=dict(assignment=i%2, block='synthetic'+str(i), reverse=reverse, schema=schema, target=i%2),
                    false_ids=[13+i, 23+reverse, 33+int(schema=='descriptive'), 895, 43, 44],
                    true_ids=[13+i, 23+reverse, 33+int(schema=='descriptive'), 830, 43, 44])
                values = {}
                with ReferenceEngine(model, case) as engine:
                    for arm in ('F', 'B', 'C1', 'T'):
                        packed, _, _ = engine.forward(arm)
                        values.update({arm+'_'+n:v for n,v in packed.items()})
                path = reference / 'cases' / (key+'.npz')
                np.savez_compressed(path, **values)
                entries.append(dict(key=key,arrays_sha256=sha(path)))
                cases.append(case)
    save_json(reference/'design.json',dict(experiment=281,cases=cases,source_hashes={},synthetic_reference=True))
    save_json(reference/'status.json',dict(status='complete',cases=entries,synthetic=True))
    save_json(reference/'audit.json',dict(status='passed',synthetic_reference=True))
    fake = dict(synthetic_checkpoint=True)
    run282.load_model = lambda offline:model
    run282.locate = lambda root:(checkpoint,fake)
    run282.runtime = lambda current:dict(synthetic=True,device='cpu',layers=3)
    audit282.load_model = lambda offline:model
    audit282.locate = lambda root:(checkpoint,fake)
    run = out/'complete'
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        run282.main(reference,run)
    (out/'producer.log').write_text(captured.getvalue())
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        audit282.main(run,reference)
    (out/'audit.log').write_text(captured.getvalue())
    status,audit=(read(run/(n+'.json')) for n in ('status','audit'))
    assert len(status['cases'])==32 and status['counts']['full_forwards']==480 and audit['status']=='passed'
    assert audit['native_counts']['U']==audit['native_counts']['C1'] and audit['native_counts']['T']==32
    assert audit['False_anchors']==64 and len(audit['paired_local_counts'])==13
    for entry in status['cases']:
        aliases=read(run/'cases'/(entry['key']+'.json'))['array_aliases']
        assert any('reference' in a for a in aliases.values()) and any('local' in a for a in aliases.values())
    print(dict(passed=True,synthetic=True,checkpoint_loaded=False,real_study_cases=False,
        entire_producer_and_audit=True,reference_CPU_forwards=128,new_CPU_forwards=480,local_corner_maps=128,
        numerical_checks=audit['numerical_checks'],all_full_zero_and_archive_controls=True,paired_decisions=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
