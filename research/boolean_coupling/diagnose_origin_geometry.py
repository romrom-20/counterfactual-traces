"""Posthoc signed paired and product geometry, no new model forwards or forecasts."""

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


def metrics(value, target):
    n, t = float(np.linalg.norm(value)), float(np.linalg.norm(target))
    return dict(component_norm=n, target_norm=t, relative_norm=n/max(t, 1e-30),
        relative_error=float(np.linalg.norm(value-target))/max(t, 1e-30),
        cosine=float(np.vdot(value, target))/max(n*t, 1e-30),
        signed_projection=float(np.vdot(value, target))/max(t*t, 1e-30))


def main(run, reference, out):
    assert not out.exists()
    audit = read(run/'audit.json')
    assert audit['status'] == 'passed' and audit['analysis_qualification'] == '282c-paired-whole-secondary-unavailable'
    design, status = (read(run/(n+'.json')) for n in ('design', 'status'))
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design['checkpoint']
    with safe_open(folder/'model.safetensors', framework='pt', device='cpu') as handle:
        w = weights(handle, read(folder/'config.json'), 1, 'mlp')
    down = w['down_weight']
    cases = {c['key']: c for c in design['cases']}
    bank, rows, norms, closures = {}, [], [], []
    def partition(key, kind, target, pieces):
        pieces = {**pieces, 'rounding': target-sum(pieces.values())}
        error = maximum(sum(pieces.values()), target)
        closures.append(error)
        assert error <= 1e-10
        for component, value in pieces.items():
            rows.append(dict(key=key, kind=kind, component=component, **metrics(value, target), closure=error))
    for entry in status['cases']:
        key = entry['key']
        path = run/'cases'/(key+'.npz')
        meta = read(path.with_suffix('.json'))
        assert sha(path) == entry['arrays_sha256'] and sha(reference/'cases'/path.name) == meta['reference_arrays_sha256']
        with np.load(path) as archive, np.load(reference/'cases'/path.name) as old:
            def get(name):
                alias = meta['array_aliases'].get(name)
                value = archive[name] if alias is None else old[alias['reference']] if 'reference' in alias else archive[alias['local']]
                assert digest(value) == meta['logical_arrays'][name]['sha256']
                return value.astype(float)
            product, e, up = {}, {}, {}
            for c in ('F', 'C', 'A', 'U'):
                p = 'corner_'+c+'_'
                product[c] = get(p+'finite_product')-get(p+'cubic_product')
                e[c] = get(p+'phi')-get(p+'cubic')
                up[c] = get(p+'up')
            assert np.array_equal(e['F'], np.zeros_like(e['F']))
            mixed = product['U']-product['C']-product['A']
            # This exact known-function ledger uses the full saved U operand.
            # It is not an independently installed causal decomposition.
            gate = (e['U']-e['C']-e['A'])*up['U']
            up_part = e['C']*(up['U']-up['C'])+e['A']*(up['U']-up['A'])
            partition(key, 'local_mixed_write', down@mixed, dict(mixed_gate_remainder=down@gate, mixed_up_reweighting=down@up_part))
            target = get('C1_write')-get('B_write')
            pieces = dict(carry=down@product['C'], attention=down@product['A'],
                mixed_gate_remainder=down@gate, mixed_up_reweighting=down@up_part)
            bank[key] = dict(target=target, pieces=pieces)
            carry, attention = get('component_carry'), get('component_attention')
            norms.append(dict(key=key, **cases[key]['group'], carry_input_norm=float(np.linalg.norm(carry)),
                attention_input_norm=float(np.linalg.norm(attention)), attention_to_carry=float(np.linalg.norm(attention)/np.linalg.norm(carry)),
                missing_write_norm=float(np.linalg.norm(target)), carry_write_norm=float(np.linalg.norm(pieces['carry'])),
                attention_write_norm=float(np.linalg.norm(pieces['attention'])), mixed_write_norm=float(np.linalg.norm(down@mixed)),
                attention_write_relative=float(np.linalg.norm(pieces['attention'])/np.linalg.norm(target)),
                mixed_write_relative=float(np.linalg.norm(down@mixed)/np.linalg.norm(target))))
    groups = {}
    for key, case in cases.items():
        g = case['group']
        groups.setdefault(tuple(g[n] for n in ('assignment', 'block', 'reverse', 'target')), {})[g['schema']] = key
    assert len(groups) == 16
    for index, keys in groups.items():
        d, g = bank[keys['descriptive']], bank[keys['generic']]
        target = d['target']-g['target']
        partition('|'.join(map(str, index)), 'paired_missing_write', target,
            {n: d['pieces'][n]-g['pieces'][n] for n in d['pieces']})
    out.mkdir()
    table(out/'signed-ledger.csv', rows)
    table(out/'input-output-norms.csv', norms)
    save_json(out/'provenance.json', dict(posthoc=True, model_forwards=0, predictions_modified=False,
        uses_saved_full_calibration=True, direct_primitive_interventions=False,
        source_sha256=sha(Path(__file__)), audit_sha256=sha(run/'audit.json'),
        checkpoint=checkpoint, down_weight_sha256=digest(down), cases=32, dependent_pairs=16,
        signed_closure_max=max(closures),
        csv_sha256={p.name: sha(p) for p in out.glob('*.csv')},
        boundary='Exact signed product/write identities, not learned features, causal shares, a fresh predictor or rescue of the failed primary. Nonlinear gate remainder includes input normalization and SiLU jointly.'))
    print('Posthoc282 geometry:32 cases/16 pairs, no new forwards, signed closure', max(closures))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference-281d', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.reference_281d.resolve(), a.out.resolve())
