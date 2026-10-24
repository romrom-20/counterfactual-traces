"""Separate arithmetic qualification, source-error ledger and full endpoints280."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .audit279 import coefficient_checks, maximum
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .composition280 import EVAL, ROOT, digest, padded
from .core225b import array, load_model
from .run272 import stamp
from .source279 import derivatives


def good(v, relative=.1, cosine=.995):
    return bool(v['norm'] >= .001 and v['relative'] <= relative and v['cosine'] >= cosine)


def flatvector(a, b):
    return vector(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))


def main(run):
    assert not (run / 'audit.json').exists()
    design, status = read(run / 'design.json'), read(run / 'status.json')
    assert design['experiment'] == 280 and design['retrospective'] and status['status'] == 'complete'
    assert len(status['cases']) == 32 and design['installed_sources'] == installed_sources()
    for n, h in design['source_hashes'].items():
        assert sha(ROOT / n) == sha(run / 'sources' / Path(n).name) == h
    folder, checkpoint = locate(ROOT)
    assert design['checkpoint'] == checkpoint
    model, head = load_model(True), read_head(folder)
    cases = {c['key']: c for c in design['cases']}
    checks, coefficients, raw, heads, native, trajectory, joint, correction, ledgers, locals_, scores = ([] for _ in range(11))
    replay_count = 0

    def check(key, label, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(key=key, name=label, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, arm, layer, node, a, b):
        error = maximum(a, b)
        raw.append(dict(key=key, arm=arm, layer=layer, node=node, error=error, original_bound=.0001, passes=bool(np.isfinite(error) and error <= .0001)))

    for e in status['cases']:
        key, case = e['key'], cases[e['key']]
        path = run / 'cases' / (key + '.npz')
        assert sha(path) == e['arrays_sha256'] and sha(path.with_suffix('.json')) == e['metadata_sha256']
        meta = read(path.with_suffix('.json'))
        length, start = len(case['false_ids']), case['position']
        with np.load(path) as archive, torch.no_grad():
            d = {n: archive[n] for n in archive.files}
            j = int(d['selected'])
            assert j + 1 == meta['selected_layer']
            missing = d['C_finite_write'].astype(float) - d['C_write'].astype(float)
            assert j == int(np.argmax(np.linalg.norm(missing, axis=-1)))
            for a in ('F', *EVAL, 'FZ'):
                m = meta['arms'][a]
                for n in ('prefix_hashes', 'K_prefix', 'V_prefix'):
                    assert m[n] == meta['arms']['F'][n]
                finite_sites = 28 if a in ('F', 'T') else 27 if a == 'C1' else 1 if a == 'R' else 0
                assert m['current_source_finite_sites'] == finite_sites
                assert m['false_primal_checks'] == 28 - finite_sites
                check(key, 'source_addition', (d[a + '_mid'] + d[a + '_write']).astype(np.float32), d[a + '_source'])
                if a == 'FZ':
                    for n in ('source', 'final', 'scores', 'gate', 'up', 'phi', 'write', 'K', 'V'):
                        check(key, 'zero_' + n, d[a + '_' + n], d['F_' + n])
                if a in ('R', 'G0', 'G1', 'G2'):
                    for n in ('before', 'mid', 'gate', 'up'):
                        check(key, 'repair_input_' + n, d[a + '_' + n][j], d['C_' + n][j])
                    if j:
                        check(key, 'repair_earlier_source', d[a + '_source'][:j], d['C_source'][:j])
                if a.startswith('G'):
                    check(key, 'random_norm', np.linalg.norm(d['draw_' + a]), np.linalg.norm(missing[j]), 1e-10)
                    check(key, 'random_write', d[a + '_write'][j], (d[a + '_raw_write'][j].astype(float) + d['draw_' + a]).astype(np.float32))
                else:
                    check(key, 'unmodified_write', d[a + '_write'], d[a + '_raw_write'])
                full = padded(np.zeros(1536, dtype=np.float32), length, start, 'mps')
                full[0, start:] = torch.tensor(d[a + '_final'], device='mps')
                actual = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                check(key, 'finite_head', actual, d[a + '_scores'])
                h = d[a + '_final'][-1].astype(float)
                cpu = h / np.sqrt(np.mean(h * h) + head['final_eps']) * head['final_norm'] @ head['final_head'].T
                check(key, 'CPU_head', cpu, d[a + '_scores'], .005)
                heads.append(dict(key=key, arm=a, public_error=maximum(actual, d[a + '_scores']), cpu_error=maximum(cpu, d[a + '_scores'])))
                scores.append(dict(key=key, arm=a, false_score=float(d[a + '_scores'][0]), true_score=float(d[a + '_scores'][1]), margin=margin(d[a + '_scores'])))
            check(key, 'removed_local_write', d['R_write'][j], d['C_finite_write'][j])
            check(key, 'removed_local_source', d['R_source'][j], d['finite_C_source'][j])
            for i, layer in enumerate(model.model.layers):
                coeff, errors = coefficient_checks(d['F_gate'][i], derivatives(d['F_gate'][i]))
                coefficients.append(dict(key=key, layer=i + 1, bound=.0001, **errors))
                for name, err in errors.items():
                    check(key, name, err, 0., .0001)
                check(key, 'False_smooth_primal', functional.silu(torch.tensor(d['F_gate'][i].astype(float))).numpy(), d['F_phi'][i], .0001)
                down = array(layer.mlp.down_proj.weight).astype(float)
                for a in ('F', *EVAL, 'FZ'):
                    degree = 0 if a in ('F', 'T') or (a == 'C1' and i != 0) or (a == 'R' and i == j) else 1 if a == 'L' else 2 if a == 'Q' else 3
                    if degree:
                        dg = d[a + '_gate'][i].astype(float) - d['F_gate'][i].astype(float)
                        ph = d['F_phi'][i].astype(float) + coeff['d1'] * dg
                        if degree >= 2:
                            ph += .5 * coeff['d2'] * dg**2
                        if degree >= 3:
                            ph += coeff['d3'] * dg**3 / 6
                        check(key, 'AD_coefficient_polynomial_install', ph.astype(np.float32), d[a + '_phi'][i], .0001)
                    else:
                        ph = functional.silu(torch.tensor(d[a + '_gate'][i].astype(float))).numpy()
                    product = (d[a + '_phi'][i] * d[a + '_up'][i]).astype(np.float32)
                    assert digest(product) == meta['arms'][a]['product_hashes'][i]
                    tensor = padded(product, length, start, 'mps')
                    public = array(layer.mlp.down_proj(tensor)[0, start])
                    replay_count += 1
                    check(key, 'public_down_replay', public, d[a + '_raw_write'][i])
                    rawcheck(key, a, i + 1, 'phi', ph, d[a + '_phi'][i])
                    rawcheck(key, a, i + 1, 'product', ph * d[a + '_up'][i].astype(float), product)
                    rawcheck(key, a, i + 1, 'write', (ph * d[a + '_up'][i].astype(float)) @ down.T, d[a + '_raw_write'][i])
                for a, prefix in (('C', 'C_finite'), ('T', 'T_cubic')):
                    ph = functional.silu(torch.tensor(d['C_gate'][i].astype(float))).numpy() if a == 'C' else None
                    if a == 'T':
                        dg = d['T_gate'][i].astype(float) - d['F_gate'][i].astype(float)
                        ph = d['F_phi'][i].astype(float) + coeff['d1'] * dg + .5 * coeff['d2'] * dg**2 + coeff['d3'] * dg**3 / 6
                    rawcheck(key, prefix, i + 1, 'phi', ph, d[prefix + '_phi'][i])
                    product = (d[prefix + '_phi'][i] * d[a + '_up'][i]).astype(np.float32)
                    public = array(layer.mlp.down_proj(padded(product, length, start, 'mps'))[0, start])
                    replay_count += 1
                    check(key, 'diagnostic_public_down', public, d[prefix + '_write'][i])
                    finite_phi = array(layer.mlp.act_fn(padded(d['C_gate'][i], length, start, 'mps'))[0, start]) if a == 'C' else None
                    if a == 'C':
                        check(key, 'diagnostic_finite_phi', finite_phi, d[prefix + '_phi'][i])
                error = d['T_source'][i].astype(float) - d['C_source'][i].astype(float)
                inherited, new = d['inherited_error'][i], d['new_error'][i]
                check(key, 'exact_source_error_closure', error, inherited + new, 1e-10)
                denominator = float(error @ error)
                ledgers.append(dict(key=key, **case['group'], layer=i + 1, selected=i == j,
                    error_norm=float(np.linalg.norm(error)), inherited_norm=float(np.linalg.norm(inherited)), new_norm=float(np.linalg.norm(new)),
                    inherited_projection=float(inherited @ error / denominator) if denominator else 0.,
                    new_projection=float(new @ error / denominator) if denominator else 0.,
                    cross=float(2 * inherited @ new), closure=maximum(error, inherited + new)))
                for a, finite, approximation in (('C', d['C_finite_write'][i], d['C_write'][i]), ('T', d['T_write'][i], d['T_cubic_write'][i])):
                    target = finite.astype(float) - d['F_write'][i].astype(float)
                    v = flatvector(approximation.astype(float) - d['F_write'][i].astype(float), target)
                    locals_.append(dict(key=key, **case['group'], incoming=a, layer=i + 1, **v,
                        missing_norm=float(np.linalg.norm(finite.astype(float) - approximation.astype(float))), passes=good(v)))
            for a in EVAL:
                row = dict(key=key, **case['group'], arm=a)
                ok = []
                for label, pick in (('all', slice(1, None)), ('verdict', -1)):
                    v = flatvector((d[a + '_final'].astype(float) - d['F_final'].astype(float))[pick], (d['T_final'].astype(float) - d['F_final'].astype(float))[pick])
                    row.update({label + '_' + k: x for k, x in v.items()})
                    ok.append(good(v, .05, .99))
                row.update(margin_error=abs(margin(d[a + '_scores']) - margin(d['T_scores'])))
                row['passes'] = bool(all(ok) and row['margin_error'] <= .1)
                native.append(row)
                tr = dict(key=key, **case['group'], arm=a)
                tr_ok = []
                for label in ('source', 'V', 'K'):
                    v = flatvector(d[a + '_' + label].astype(float) - d['F_' + label].astype(float), d['T_' + label].astype(float) - d['F_' + label].astype(float))
                    tr.update({label + '_' + k: x for k, x in v.items()})
                    if label != 'K':
                        tr_ok.append(good(v))
                tr['passes'] = all(tr_ok)
                trajectory.append(tr)
                joint.append(dict(key=key, **case['group'], arm=a, native_pass=row['passes'], trajectory_pass=tr['passes'], passes=bool(row['passes'] and tr['passes'])))
            own_correction = []
            for a in ('R', 'G0', 'G1', 'G2'):
                row = dict(key=key, **case['group'], arm=a, selected_layer=j + 1)
                ok = []
                for label, pick in (('all', slice(1, None)), ('verdict', -1)):
                    v = flatvector((d[a + '_final'].astype(float) - d['C_final'].astype(float))[pick], (d['T_final'].astype(float) - d['C_final'].astype(float))[pick])
                    row.update({label + '_' + k: x for k, x in v.items()})
                    ok.append(good(v, .25, .95))
                tm = margin(d['T_scores']) - margin(d['C_scores'])
                err = abs((margin(d[a + '_scores']) - margin(d['C_scores'])) - tm)
                row.update(target_margin=tm, margin_error=err, passes=bool(all(ok) and abs(tm) >= .02 and err <= .005 + .25 * abs(tm)))
                own_correction.append(row)
            rf = sum(not r['passes'] for r in own_correction if r['arm'].startswith('G'))
            for r in own_correction:
                r.update(random_failures=rf, account_pass=bool(r['passes'] and rf >= 2))
            correction += own_correction
        print('280 qualified', len(native) // len(EVAL), '/32', flush=True)
    for name, rows in dict(checks=checks, coefficients=coefficients, raw=raw, heads=heads, native=native, trajectory=trajectory, joint=joint, correction=correction, ledger=ledgers, local=locals_, scores=scores).items():
        table(run / (name + '.csv'), rows)
    def counts(rows):
        return {a: sum(r['passes'] for r in rows if r['arm'] == a) for a in EVAL}
    def strata(rows, arm, field='passes'):
        return {s + '/' + str(rev): sum(r[field] for r in rows if r['arm'] == arm and r['schema'] == s and r['reverse'] == rev) for s in ('generic', 'descriptive') for rev in (0, 1)}
    nc, tc, jc = counts(native), counts(trajectory), counts(joint)
    primary = jc['C'] >= 24 and min(strata(joint, 'C').values()) >= 6
    account_count = sum(r['account_pass'] for r in correction if r['arm'] == 'R')
    account_strata = strata(correction, 'R', 'account_pass')
    save_json(run / 'audit.json', dict(status='passed', experiment=280, retrospective=True, source_sha256=sha(__file__),
        design_sha256=sha(run / 'design.json'), status_sha256=sha(run / 'status.json'),
        native_counts=nc, trajectory_counts=tc, joint_counts=jc, C_strata=strata(joint, 'C'), primary_passes=bool(primary),
        correction_account_count=account_count, correction_account_strata=account_strata,
        correction_account_passes=bool(account_count >= 24 and min(account_strata.values()) >= 6),
        numerical_checks=len(checks), numerical_max_error=max(r['error'] for r in checks),
        False_gate_anchors=len(coefficients), coefficient_vectors_checked=3 * len(coefficients),
        public_down_replays=replay_count, raw_checks=dict(total=len(raw), failed=sum(not r['passes'] for r in raw), original_bound=.0001),
        csv_sha256={p.name: sha(p) for p in run.glob('*.csv')}, ended=stamp(),
        boundary='Old dependent examples; all receiving/attention/norm/projections/product finite. Qualified primitive/source-only intervention, no old-check recertification, efficient whole-network result, learned circuit or complete QB.'))
    print(dict(native=nc, trajectory=tc, joint=jc, primary=primary, correction_account=account_count), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.run.resolve())
    except Exception as exc:
        if not (args.run / 'failure-audit.json').exists():
            save_json(args.run / 'failure-audit.json', dict(error=repr(exc), time=stamp()))
        raise
