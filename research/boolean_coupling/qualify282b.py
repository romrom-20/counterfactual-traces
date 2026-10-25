"""Seal a narrower282b numerical contract, preserving the original failed audit."""

import argparse
import ast
import math
import shutil
import subprocess
from pathlib import Path

from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .origin282 import ROOT
from .run272 import stamp

SOURCES = ('research/boolean_coupling/audit282b.py',
           'research/boolean_coupling/qualify282b.py',
           'research/boolean_coupling/diagnose_input_bridges282.py',
           'research/boolean_coupling/preflight282b.py',
           'docs/experiments/282b-public-up-qualification.md')


def validate(failure):
    message = failure['error']
    assert message.startswith('AssertionError(') and message.endswith(')')
    row = ast.literal_eval(message[len('AssertionError('):-1])
    assert isinstance(row, dict) and row['name'].startswith('CPU64_') and row['name'].endswith('_up')
    assert row['bound'] == .0001 and math.isfinite(row['error']) and row['error'] > .0001
    return row


def main(run, diagnostic):
    assert not (run/'audit.json').exists() and not (run/'audit-b-design.json').exists()
    design, status, failure = (read(run/(n+'.json')) for n in ('design', 'status', 'failure-audit'))
    assert design['experiment'] == 282 and status['status'] == 'complete' and len(status['cases']) == 32
    original = validate(failure)
    report = read(diagnostic/'diagnostic.json')
    assert report['posthoc'] and report['numerical_only'] and not report['behavior_outcomes_inspected']
    assert report['design_sha256'] == sha(run/'design.json') and report['status_sha256'] == sha(run/'status.json')
    assert report['failure_sha256'] == sha(run/'failure-audit.json')
    assert report['source_sha256'] == sha(ROOT/'research/boolean_coupling/diagnose_input_bridges282.py')
    assert report['csv_sha256'] == sha(diagnostic/'bridges.csv') and report['comparisons'] == 256
    for n, h in design['source_hashes'].items():
        assert sha(ROOT/n) == sha(run/'sources'/Path(n).name) == h
    source_hashes = {n: sha(ROOT/n) for n in SOURCES}
    (run/'audit-b-sources').mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT/n, run/'audit-b-sources'/Path(n).name)
    dest = run/'input-bridge-diagnostic'
    dest.mkdir()
    for n in ('diagnostic.json', 'bridges.csv'):
        shutil.copy2(diagnostic/n, dest/n)
    save_json(run/'audit-b-design.json', dict(qualification='282b-public-Float32-up-with-rounding-envelope',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=source_hashes, design_sha256=sha(run/'design.json'), status_sha256=sha(run/'status.json'),
        original_failure_sha256=sha(run/'failure-audit.json'), original_failed_check=original,
        diagnostic_sha256=sha(dest/'diagnostic.json'), began=stamp(),
        behavioral_predictions_or_thresholds_changed=False, original_absolute_tolerance_changed=False,
        numerical_validation_scope_changed=True, original_audit_passed=False,
        boundary='CPU64 up map at original 1e-4 remains failed; exact public Float32 maps and explicit dimension-derived numerical envelope qualify only that execution. Other required checks unchanged.'))
    print('Sealed282b sources and narrower numerical scope before audit. Original audit remains failed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.diagnostic.resolve())
