"""Seal the published281e qualification for a NEW repetition, preserving failure."""

import argparse
import ast
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(run):
    assert not (run / 'audit.json').exists() and not (run / 'audit-e-design.json').exists()
    design, status, failure = (json.loads((run / (n + '.json')).read_text()) for n in ('design', 'status', 'failure-audit'))
    assert design['experiment'] == 281 and status['status'] == 'complete' and len(status['cases']) == 32
    message = failure['error']
    assert message.startswith('AssertionError(') and message.endswith(')')
    check = ast.literal_eval(message[len('AssertionError('):-1])
    assert isinstance(check, dict) and check['name'] == 'random_direction' and check['bound'] == 0
    assert check['key'] == status['cases'][0]['key'] and math.isfinite(check['error']) and 0 < check['error'] <= 1e-15
    original = json.loads((ROOT / 'results/source2-products-281d/audit-e-design.json').read_text())
    for name, h in original['source_hashes'].items():
        assert sha(ROOT / name) == h
    dest = run / 'audit-e-sources'
    dest.mkdir()
    for name in original['source_hashes']:
        shutil.copy2(ROOT / name, dest / Path(name).name)
    sealed = dict(qualification='281e-generator-order',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        design_sha256=sha(run / 'design.json'), status_sha256=sha(run / 'status.json'),
        original_failure_sha256=sha(run / 'failure-audit.json'), source_hashes=original['source_hashes'],
        exact_bound_changed=False, published_literal_repetition=True, fresh_confirmation=False)
    (run / 'audit-e-design.json').write_text(json.dumps(sealed, indent=2) + '\n')
    print('Sealed qualified audit sources before rerun; original exact failure retained. No fresh support.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args().run.resolve())
