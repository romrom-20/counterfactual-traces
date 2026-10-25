"""Posthoc numerical diagnostic only: CPU64 chained versus consumed-input maps."""

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read, table
from .audit279 import maximum
from .cache272 import save_json
from .checkpoint238 import locate, weights
from .composition280 import digest
from .origin282 import ROOT


def main(run, reference, out):
    assert not out.exists() and not (run/'audit.json').exists()
    design, status = (read(run/(n+'.json')) for n in ('design', 'status'))
    assert status['status'] == 'complete' and len(status['cases']) == 32
    assert (run/'failure-audit.json').exists()
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design['checkpoint']
    with safe_open(folder/'model.safetensors', framework='pt', device='cpu') as handle:
        w = weights(handle, read(folder/'config.json'), 1, 'mlp')
    rows = []
    for entry in status['cases']:
        key = entry['key']
        path = run/'cases'/(key+'.npz')
        assert sha(path) == entry['arrays_sha256']
        meta = read(path.with_suffix('.json'))
        ref = reference/'cases'/path.name
        assert sha(ref) == meta['reference_arrays_sha256']
        with np.load(path) as archive, np.load(ref) as old:
            def get(name):
                alias = meta['array_aliases'].get(name)
                value = archive[name] if alias is None else old[alias['reference']] if 'reference' in alias else archive[alias['local']]
                assert digest(value) == meta['logical_arrays'][name]['sha256']
                return value.astype(float)
            for corner in ('F', 'C', 'A', 'U'):
                prefix = 'corner_'+corner+'_'
                x = get(prefix+'mid')
                z = x/np.sqrt(np.mean(x*x)+w['eps'])*w['norm']
                consumed = get(prefix+'norm')
                for node in ('gate', 'up'):
                    weight = w[node+'_weight']
                    bias = w.get(node+'_bias', 0.)
                    chained, local = z@weight.T+bias, consumed@weight.T+bias
                    observed = get(prefix+node)
                    inherited = (consumed-z)@weight.T
                    operation = observed-local
                    total = observed-chained
                    closure = maximum(total, inherited+operation)
                    i = int(np.argmax(np.abs(total)))
                    rows.append(dict(key=key, corner=corner, node=node,
                        original_chained_error=maximum(chained, observed), original_bound=.0001,
                        original_passes=bool(maximum(chained, observed)<=.0001),
                        norm_error=maximum(z, consumed), consumed_input_error=maximum(local, observed),
                        propagated_norm_error=float(np.max(np.abs(inherited))),
                        signed_closure=closure, max_error_coordinate=i,
                        total_at_max=float(total[i]), propagated_at_max=float(inherited[i]),
                        operation_at_max=float(operation[i])))
    out.mkdir()
    table(out/'bridges.csv', rows)
    save_json(out/'diagnostic.json', dict(posthoc=True, numerical_only=True, model_forwards=0,
        behavior_outcomes_inspected=False, original_audit_passed=False,
        design_sha256=sha(run/'design.json'), status_sha256=sha(run/'status.json'),
        failure_sha256=sha(run/'failure-audit.json'), source_sha256=sha(Path(__file__)),
        csv_sha256=sha(out/'bridges.csv'), comparisons=len(rows),
        original_chained_failures=sum(not r['original_passes'] for r in rows),
        maximum_chained_error=max(r['original_chained_error'] for r in rows),
        maximum_norm_error=max(r['norm_error'] for r in rows),
        maximum_consumed_input_error=max(r['consumed_input_error'] for r in rows),
        maximum_signed_closure=max(r['signed_closure'] for r in rows),
        boundary='Numerical error decomposition, not an attribution of a learned computation. Original chained 1e-4 checks remain failed.'))
    print(read(out/'diagnostic.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference-281d', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.reference_281d.resolve(), a.out.resolve())
