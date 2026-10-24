"""Posthoc281 geometry and exact SiLU/rectifier ledger; zero model forwards."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read, table
from .checkpoint238 import locate

ROOT = Path(__file__).resolve().parents[2]


def cosine(x, y):
    return float(np.vdot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def hump(g):
    z = np.exp(-abs(g))
    return abs(g) * z / (1 + z)


def main(run, out):
    assert not out.exists() and read(run / 'audit.json')['status'] == 'passed'
    design, status = read(run / 'design.json'), read(run / 'status.json')
    folder, checkpoint = locate(ROOT)
    with safe_open(folder / 'model.safetensors', framework='pt', device='cpu') as handle:
        down = handle.get_tensor('model.layers.1.mlp.down_proj.weight').float().numpy().astype(float)
    out.mkdir()
    geometry, ledger = [], []
    groups = {c['key']: c['group'] for c in design['cases']}
    for entry in status['cases']:
        p = run / 'cases' / (entry['key'] + '.npz')
        assert sha(p) == entry['arrays_sha256']
        with np.load(p) as d:
            key, group = entry['key'], groups[entry['key']]
            local = {a: d[a + '_write'].astype(float) - d['B_write'].astype(float) for a in ('P', 'H', 'C1')}
            final = {a: d[a + '_final'][1:].astype(float) - d['B_final'][1:].astype(float) for a in ('P', 'H', 'C1')}
            norm, final_norm = np.linalg.norm(local['C1']), np.linalg.norm(final['C1'])
            native_norm = np.linalg.norm(d['T_final'][1:].astype(float) - d['F_final'][1:].astype(float))
            assert norm >= .001 and final_norm >= .001
            geometry.append(dict(key=key, **group,
                overlap=int(len(set(d['mask_P']) & set(d['mask_H']))),
                local_P_H_cosine=cosine(local['P'], local['H']),
                local_P_H_difference_relative=float(np.linalg.norm(local['P'] - local['H']) / norm),
                final_P_H_cosine=cosine(final['P'], final['H']),
                final_P_H_difference_relative=float(np.linalg.norm(final['P'] - final['H']) / final_norm),
                missing_effect_over_native_norm=float(final_norm / native_norm),
                final_P_H_difference_over_native_norm=float(np.linalg.norm(final['P'] - final['H']) / native_norm)))
            gf, g, up, pf, cubic = (d[n].astype(float) for n in ('F_node2_gate', 'B_node2_gate', 'B_node2_up', 'F_node2_phi', 'B_node2_phi'))
            proxy = pf + np.maximum(g, 0) - np.maximum(gf, 0) - cubic
            qchange = hump(gf) - hump(g)
            anchor_rounding = np.maximum(gf, 0) - hump(gf) - pf
            direct = g / (1 + np.exp(-g)) - cubic
            identity = float(np.max(abs(proxy + qchange + anchor_rounding - direct)))
            assert identity <= 1e-12
            finite = d[read(p.with_suffix('.json'))['arms']['C1']['phi_ref']].astype(float)
            # Rank with the finite target only AFTER the original experiment: diagnostic, not forecast.
            coefficients = (finite * up).astype(np.float32).astype(float) - (cubic * up).astype(np.float32).astype(float)
            energy = coefficients**2 * np.sum(down**2, axis=0)
            top90 = np.lexsort((np.arange(len(g)), -energy))[:90]
            topvector = down[:, top90] @ coefficients[top90]
            for arm in ('all', 'P', 'H'):
                unselected = np.ones(len(g), dtype=bool)
                missing = local['C1'].copy()
                if arm != 'all':
                    unselected[d['mask_' + arm]] = False
                    missing -= local[arm]
                den = float(missing @ missing)
                pieces = {n: down[:, unselected] @ (x[unselected] * up[unselected]) for n, x in (
                    ('proxy', proxy), ('hump_change', qchange), ('anchor_rounding', anchor_rounding))}
                pieces['remaining_rounding'] = missing - sum(pieces.values())
                projections = {n: float(v @ missing / den) for n, v in pieces.items()}
                closure = abs(sum(projections.values()) - 1)
                assert closure <= 1e-10
                ledger.append(dict(key=key, **group, arm=arm, identity_error=identity,
                    missing_norm=float(np.sqrt(den)),
                    **{n + '_signed_projection': v for n, v in projections.items()},
                    signed_closure_error=closure,
                    rounding_relative=float(np.linalg.norm(pieces['remaining_rounding']) / np.sqrt(den)),
                    target_aware_top90_energy_fraction=float(energy[top90].sum() / energy.sum()),
                    target_aware_top90_write_relative=float(np.linalg.norm(topvector - local['C1']) / norm)))
    table(out / 'geometry-cases.csv', geometry)
    table(out / 'rectifier-ledger.csv', ledger)
    summary = dict(geometry={n: float(np.median([r[n] for r in geometry])) for n in geometry[0] if n not in ('key', *groups[key])},
        ledger={a: {n: float(np.median([r[n] for r in ledger if r['arm'] == a])) for n in ledger[0] if n not in ('key', 'arm', *groups[key])} for a in ('all', 'P', 'H')})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'provenance.json').write_text(json.dumps(dict(posthoc=True, model_forwards=0, native_targets_used=True,
        source_sha256=sha(__file__), audit_sha256=sha(run / 'audit.json'), checkpoint=checkpoint,
        source_arrays={e['key']: e['arrays_sha256'] for e in status['cases']},
        csv_sha256={p.name: sha(p) for p in out.glob('*.csv')},
        boundary='Saved interventions and exact SiLU identity; no new causal intervention, receiver-filtering explanation, forecast, feature or learned-circuit claim.'), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    main(args.run.resolve(), args.out.resolve())
