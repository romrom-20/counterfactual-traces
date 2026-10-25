"""Seal paired-secondary accounting without changing282 forecasts or primary."""

import argparse
import shutil
import subprocess
from pathlib import Path

from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .origin282 import ROOT
from .run272 import stamp

SOURCES = ('research/boolean_coupling/audit282c.py',
           'research/boolean_coupling/qualify282c.py',
           'research/boolean_coupling/preflight282c.py',
           'docs/experiments/282c-paired-secondary-accounting.md')


def main(run):
    assert not (run/'audit.json').exists() and not (run/'audit-c-design.json').exists()
    design, status, failure, numerical = (read(run/(n+'.json')) for n in ('design', 'status', 'failure-audit-b', 'audit-b-design'))
    assert design['experiment'] == 282 and status['status'] == 'complete'
    assert failure['error'].startswith("ValueError('operands could not be broadcast together with shapes ")
    assert numerical['qualification'] == '282b-public-Float32-up-with-rounding-envelope'
    assert numerical['design_sha256'] == sha(run/'design.json') and numerical['status_sha256'] == sha(run/'status.json')
    for n, h in {**design['source_hashes'], **numerical['source_hashes']}.items():
        assert sha(ROOT/n) == h
    groups = {}
    for case in design['cases']:
        g = case['group']
        index = tuple(g[n] for n in ('assignment', 'block', 'reverse', 'target'))
        meta = read(run/'cases'/(case['key']+'.json'))
        assert sha(run/'cases'/(case['key']+'.npz')) == meta['arrays_sha256']
        shape = meta['logical_arrays']['B_final']['shape']
        assert case['false_ids'][-1] == case['true_ids'][-1]
        groups.setdefault(index, {})[g['schema']] = dict(rows=shape[0]-1, last_token=case['false_ids'][-1])
    assert len(groups) == 16 and all(set(g) == {'generic', 'descriptive'} for g in groups.values())
    assert all(g['generic']['last_token'] == g['descriptive']['last_token'] for g in groups.values())
    unequal = sum(g['generic']['rows'] != g['descriptive']['rows'] for g in groups.values())
    assert unequal == 16
    hashes = {n: sha(ROOT/n) for n in SOURCES}
    (run/'audit-c-sources').mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT/n, run/'audit-c-sources'/Path(n).name)
    save_json(run/'audit-c-design.json', dict(qualification='282c-paired-whole-secondary-unavailable',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, design_sha256=sha(run/'design.json'), status_sha256=sha(run/'status.json'),
        original_failure_b_sha256=sha(run/'failure-audit-b.json'), numerical_qualification_sha256=sha(run/'audit-b-design.json'),
        unequal_length_pairs=unequal, primary_criteria_changed=False, numerical_scope_changed_from_282b=False,
        forecasts_changed=False, original_paired_whole_secondary_unavailable=True,
        new_verdict_only_diagnostic=True, began=stamp(),
        boundary='Do not invent a token correspondence or treat the new verdict-only diagnostic as a pass of the original whole-vector secondary. Primary paired source-write criterion unchanged.'))
    print('Sealed282c paired accounting; original whole-state secondary unavailable, primary unchanged.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args().run.resolve())
