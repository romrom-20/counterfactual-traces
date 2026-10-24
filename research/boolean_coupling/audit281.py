"""Separate source2 mask, primitive/product and behavioral qualification281."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as fn

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .audit279 import coefficient_checks, maximum
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .composition280 import digest, padded, polynomial
from .core225b import array, load_model
from .products281 import ARMS, EVAL, ROOT
from .run272 import stamp
from .source279 import derivatives


def good(v, bound=.25, cos=.95):
    return bool(v['norm'] >= .001 and v['relative'] <= bound and v['cosine'] >= cos)


def difference(d, a, b, name):
    return d[a + '_' + name].astype(float) - d[b + '_' + name].astype(float)


def main(run):
    assert not (run / 'audit.json').exists()
    design, status = read(run / 'design.json'), read(run / 'status.json')
    assert design['experiment'] == 281 and design['retrospective'] and status['status'] == 'complete'
    assert len(status['cases']) == 32 and tuple(design['arms']) == ARMS
    assert design['installed_sources'] == installed_sources()
    for n, h in design['source_hashes'].items():
        assert sha(ROOT / n) == sha(run / 'sources' / Path(n).name) == h
    folder, checkpoint = locate(ROOT)
    assert design['checkpoint'] == checkpoint
    model, head = load_model(True), read_head(folder)
    layer2 = model.model.layers[1]
    column = np.linalg.norm(array(layer2.mlp.down_proj.weight).astype(float), axis=0)
    cases = {c['key']: c for c in design['cases']}
    checks, raw, coefficients, heads, native, local, effect, joint, removal, accounts, scores, selection_rows = ([] for _ in range(12))
    down_replays, input_replays, primal_replays = 0, 0, 0

    def check(key, name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(key=key, name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, arm, site, name, a, b):
        error = maximum(a, b)
        raw.append(dict(key=key, arm=arm, site=site, node=name, error=error, original_bound=.0001, passes=bool(np.isfinite(error) and error <= .0001)))

    for entry in status['cases']:
        key, case = entry['key'], cases[entry['key']]
        path = run / 'cases' / (key + '.npz')
        assert sha(path) == entry['arrays_sha256'] and sha(path.with_suffix('.json')) == entry['metadata_sha256']
        meta = read(path.with_suffix('.json'))
        assert meta['native_source_input_not_used_for_masks'] and meta['fixed_F_B_forecasts_before_full_source2_controls']
        length, start = len(case['false_ids']), case['position']
        with np.load(path) as archive, torch.no_grad():
            d = {n: archive[n] for n in archive.files}
            check(key, 'column_norm', d['column_norm'], column)
            for i, layer in enumerate(model.model.layers[:2]):
                gf, pf = (d[f'F_node{i + 1}_' + n] for n in ('gate', 'phi'))
                independent, errors = coefficient_checks(gf, derivatives(gf))
                coefficients.append(dict(key=key, site=i + 1, bound=.0001, **errors))
                for n, err in errors.items():
                    check(key, n, err, 0., .0001)
                for a in ('F', 'B', 'T'):
                    prefix = f'{a}_node{i + 1}_'
                    mid = padded(d[prefix + 'mid'], length, start, 'mps')
                    z = layer.post_attention_layernorm(mid)
                    g, up = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
                    input_replays += 1
                    check(key, 'public_gate', array(g[0, start]), d[prefix + 'gate'])
                    check(key, 'public_up', array(up[0, start]), d[prefix + 'up'])
                    phi = d[prefix + 'phi']
                    if a == 'B':
                        anchor = dict(gate=gf, phi=pf)
                        check(key, 'polynomial_install', polynomial(anchor, d[prefix + 'gate'], 3), phi)
                        dg = d[prefix + 'gate'].astype(float) - gf.astype(float)
                        shadow = pf.astype(float) + independent['d1'] * dg + .5 * independent['d2'] * dg**2 + independent['d3'] * dg**3 / 6
                        check(key, 'AD_polynomial_bridge', shadow.astype(np.float32), phi, .0001)
                    else:
                        shadow = fn.silu(torch.tensor(d[prefix + 'gate'].astype(float))).numpy()
                    product = (phi * d[prefix + 'up']).astype(np.float32)
                    actual = array(layer.mlp.down_proj(padded(product, length, start, 'mps'))[0, start])
                    down_replays += 1
                    check(key, 'shared_public_down', actual, d[prefix + 'raw_write'])
                    check(key, 'shared_write', d[prefix + 'write'], d[prefix + 'raw_write'])
                    check(key, 'source_addition', d[prefix + 'mid'] + d[prefix + 'write'], d[a + '_source'][i])
                    down = array(layer.mlp.down_proj.weight).astype(float)
                    rawcheck(key, a, i + 1, 'phi', shadow, phi)
                    rawcheck(key, a, i + 1, 'product', shadow * d[prefix + 'up'].astype(float), product)
                    rawcheck(key, a, i + 1, 'write', (shadow * d[prefix + 'up'].astype(float)) @ down.T, actual)
            gf, pf, gb, ub, cb = (d[n] for n in ('F_node2_gate', 'F_node2_phi', 'B_node2_gate', 'B_node2_up', 'B_node2_phi'))
            tg = torch.tensor(gb.astype(float))
            tf = torch.tensor(gf.astype(float))
            proxy = torch.tensor(pf.astype(float)) + torch.relu(tg) - torch.relu(tf) - torch.tensor(cb.astype(float))
            pscore = proxy.abs().numpy() * abs(ub.astype(float)) * column
            hscore = abs(gb.astype(float) - gf.astype(float))**4 * abs(ub.astype(float)) * column
            for a, score in (('P', pscore), ('H', hscore)):
                check(key, 'independent_score_' + a, score, d['score_' + a], .0001)
                expected = np.sort(np.lexsort((np.arange(len(gf)), -d['score_' + a]))[:90])
                check(key, 'rank_' + a, d['mask_' + a], expected)
            for i in range(3):
                seed = int(hashlib.sha256((key + f':281-mask:{i}').encode()).hexdigest()[:16], 16)
                check(key, 'random_mask', d['mask_R' + str(i)], np.sort(np.random.default_rng(seed).choice(len(gf), 90, replace=False)))
            primary = d['mask_P']
            dg = gb.astype(float) - gf.astype(float)
            shadow_cubic = pf.astype(float) + independent['d1'] * dg + .5 * independent['d2'] * dg**2 + independent['d3'] * dg**3 / 6
            down2 = array(layer2.mlp.down_proj.weight).astype(float)
            delta = difference(d, 'P', 'B', 'write')
            for i in range(3):
                seed = int(hashlib.sha256((key + f':281-write:{i}').encode()).hexdigest()[:16], 16)
                draw = np.random.default_rng(seed).normal(size=delta.shape)
                draw *= np.linalg.norm(delta) / np.linalg.norm(draw)
                check(key, 'random_direction', draw, d['draw_G' + str(i)])
                check(key, 'matched_norm', np.linalg.norm(draw), np.linalg.norm(delta), 1e-10)
            for a in ARMS:
                info = meta['arms'][a]
                assert info['prefix_hashes'] == meta['arms']['F']['prefix_hashes'] and info['other_rows_unchanged']
                phi = d[info['phi_ref']]
                mask = primary if a in ('P', 'D', 'A') else d['mask_R' + a[-1]] if a.startswith('A') else d['mask_' + a] if a in ('H', 'R0', 'R1', 'R2') else np.arange(90) if a == 'FZ' else None
                full = a in ('F', 'C1', 'A', 'A0', 'A1', 'A2', 'T')
                current_gate = d['F_node2_gate'] if a in ('F', 'FZ') else d['T_node2_gate'] if a == 'T' else gb
                up = d['F_node2_up'] if a in ('F', 'FZ') else d['T_node2_up'] if a == 'T' else ub
                actual_gate = current_gate.copy() if full else gf.copy()
                if not full and mask is not None:
                    actual_gate[mask] = current_gate[mask]
                assert digest(actual_gate) == info['actual_gate_hashes'][1]
                primal = array(layer2.mlp.act_fn(padded(actual_gate, length, start, 'mps'))[0, start])
                primal_replays += 1
                assert digest(primal) == info['primal_hashes'][1]
                expected_phi = primal.copy() if full else pf.copy() if a == 'FZ' else cb.copy()
                if not full and mask is not None and a != 'D':
                    expected_phi[mask] = primal[mask]
                check(key, 'actual_phi', expected_phi, phi)
                expected_count = len(gf) if full else 90 if mask is not None else 0
                assert info['current_MLP2_finite_coordinates'] == expected_count
                product = (phi * up).astype(np.float32)
                if a == 'D':
                    product[mask] = (primal[mask] * up[mask]).astype(np.float32)
                if a.startswith('A'):
                    product[mask] = (cb[mask] * up[mask]).astype(np.float32)
                assert digest(product) == info['product_hashes'][1]
                raw_write = array(layer2.mlp.down_proj(padded(product, length, start, 'mps'))[0, start])
                down_replays += 1
                check(key, 'actual_public_down', raw_write, d[a + '_raw_write'])
                smooth_primal = fn.silu(torch.tensor(actual_gate.astype(float))).numpy()
                smooth_phi = smooth_primal.copy() if full else pf.astype(float).copy() if a == 'FZ' else shadow_cubic.copy()
                if not full and mask is not None and a != 'D':
                    smooth_phi[mask] = smooth_primal[mask]
                smooth_product = smooth_phi * up.astype(float)
                if a == 'D':
                    smooth_product[mask] = smooth_primal[mask] * up[mask].astype(float)
                if a.startswith('A'):
                    smooth_product[mask] = shadow_cubic[mask] * up[mask].astype(float)
                rawcheck(key, a, 2, 'actual_phi', smooth_phi, phi)
                rawcheck(key, a, 2, 'actual_product', smooth_product, product)
                rawcheck(key, a, 2, 'actual_write', smooth_product @ down2.T, raw_write)
                expected_write = (raw_write.astype(float) + d['draw_' + a]).astype(np.float32) if a.startswith('G') else raw_write
                check(key, 'actual_write', expected_write, d[a + '_write'])
                mid = d['F_node2_mid'] if a in ('F', 'FZ') else d['T_node2_mid'] if a == 'T' else d['B_node2_mid']
                check(key, 'second_source_addition', (mid + expected_write).astype(np.float32), d[a + '_source'][1])
                if a not in ('F', 'FZ', 'T'):
                    check(key, 'first_source_unchanged', d[a + '_source'][0], d['B_source'][0])
                    assert info['product_hashes'][0] == meta['arms']['B']['product_hashes'][0]
                # Source1 actual approximate primal must be False, not current.
                fg1, fp1 = d['F_node1_gate'], d['F_node1_phi']
                gate1 = d['T_node1_gate'] if a == 'T' else fg1
                primal1 = array(model.model.layers[0].mlp.act_fn(padded(gate1, length, start, 'mps'))[0, start])
                primal_replays += 1
                assert digest(gate1) == info['actual_gate_hashes'][0] and digest(primal1) == info['primal_hashes'][0]
                if a not in ('F', 'T'):
                    check(key, 'first_False_primal', primal1, fp1)
                final = padded(np.zeros(1536, dtype=np.float32), length, start, 'mps')
                final[0, start:] = torch.tensor(d[a + '_final'], device='mps')
                public = array(model.lm_head(model.model.norm(final)[:, -1:])[0, -1, [895, 830]])
                check(key, 'finite_head', public, d[a + '_scores'])
                h = d[a + '_final'][-1].astype(float)
                cpu = h / np.sqrt(np.mean(h * h) + head['final_eps']) * head['final_norm'] @ head['final_head'].T
                check(key, 'CPU_head', cpu, d[a + '_scores'], .005)
                heads.append(dict(key=key, arm=a, public_error=maximum(public, d[a + '_scores']), cpu_error=maximum(cpu, d[a + '_scores'])))
                scores.append(dict(key=key, arm=a, false_score=float(d[a + '_scores'][0]), true_score=float(d[a + '_scores'][1]), margin=margin(d[a + '_scores'])))
            for n in ('source', 'final', 'scores', 'write'):
                check(key, 'direct_product_equivalence', d['P_' + n], d['D_' + n])
                check(key, 'zero_edit', d['F_' + n], d['FZ_' + n])
            for a in EVAL:
                base = dict(key=key, **case['group'], arm=a)
                nr = base.copy()
                oks = []
                for scope, pick in (('all', slice(1, None)), ('verdict', -1)):
                    v = vector(difference(d, a, 'F', 'final')[pick], difference(d, 'T', 'F', 'final')[pick])
                    nr.update({scope + '_' + k: val for k, val in v.items()})
                    oks.append(good(v, .05, .99))
                nr.update(margin_error=abs(margin(d[a + '_scores']) - margin(d['T_scores'])))
                nr['passes'] = bool(all(oks) and nr['margin_error'] <= .1)
                native.append(nr)
                v = vector(difference(d, a, 'B', 'write'), difference(d, 'C1', 'B', 'write'))
                lr = dict(**base, **v, passes=good(v))
                local.append(lr)
                er = base.copy()
                oks = []
                for scope, pick in (('all', slice(1, None)), ('verdict', -1)):
                    v = vector(difference(d, a, 'B', 'final')[pick], difference(d, 'C1', 'B', 'final')[pick])
                    er.update({scope + '_' + k: val for k, val in v.items()})
                    oks.append(good(v))
                target_margin = margin(d['C1_scores']) - margin(d['B_scores'])
                me = abs(margin(d[a + '_scores']) - margin(d['C1_scores']))
                er.update(target_margin=target_margin, margin_error=me, passes=bool(all(oks) and abs(target_margin) >= .02 and me <= .005 + .25 * abs(target_margin)))
                effect.append(er)
                joint.append(dict(**base, native_pass=nr['passes'], local_pass=lr['passes'], effect_pass=er['passes'], passes=bool(nr['passes'] and lr['passes'] and er['passes'])))
            removed = []
            for a in ('A', 'A0', 'A1', 'A2'):
                row = dict(key=key, **case['group'], arm=a)
                v = vector(difference(d, 'C1', a, 'write'), difference(d, 'C1', 'B', 'write'))
                row.update({'local_' + k: val for k, val in v.items()})
                oks = [good(v)]
                for scope, pick in (('all', slice(1, None)), ('verdict', -1)):
                    v = vector(difference(d, 'C1', a, 'final')[pick], difference(d, 'C1', 'B', 'final')[pick])
                    row.update({scope + '_' + k: val for k, val in v.items()})
                    oks.append(good(v))
                target_margin = margin(d['C1_scores']) - margin(d['B_scores'])
                me = abs(margin(d[a + '_scores']) - margin(d['B_scores']))
                row.update(target_margin=target_margin, margin_error=me, passes=bool(all(oks) and abs(target_margin) >= .02 and me <= .005 + .25 * abs(target_margin)))
                removed.append(row)
            removal += removed
            own_joint = {r['arm']: r for r in joint if r['key'] == key}
            own_local = {r['arm']: r for r in local if r['key'] == key}
            own_effect = {r['arm']: r for r in effect if r['key'] == key}
            mask_fail = sum(not (own_local['R' + str(i)]['passes'] and own_effect['R' + str(i)]['passes']) for i in range(3))
            direction_fail = sum(not (own_local['G' + str(i)]['passes'] and own_effect['G' + str(i)]['passes']) for i in range(3))
            removal_fail = sum(not r['passes'] for r in removed if r['arm'] != 'A')
            restore = bool(own_joint['P']['passes'] and mask_fail >= 2 and direction_fail >= 2)
            remove = bool(removed[0]['passes'] and removal_fail >= 2)
            damage = not next(r['passes'] for r in native if r['key'] == key and r['arm'] == 'B')
            accounts.append(dict(key=key, **case['group'], mask_failures=mask_fail, direction_failures=direction_fail, removal_control_failures=removal_fail,
                restoration_pass=restore, removal_pass=remove, baseline_damage=damage, behavioral_account_pass=bool(restore and remove and damage)))
            full_phi = d[meta['arms']['C1']['phi_ref']]
            product_delta = (full_phi * ub).astype(np.float32).astype(float) - (cb * ub).astype(np.float32).astype(float)
            missing_write = difference(d, 'C1', 'B', 'write')
            denom = float(missing_write @ missing_write)
            all_energy = np.sum(down2 * down2, axis=0) * product_delta**2
            for a in ('P', 'H', 'R0', 'R1', 'R2'):
                indices = d['mask_' + a]
                selected_vector = down2[:, indices] @ product_delta[indices]
                actual_delta = difference(d, a, 'B', 'write')
                selection_rows.append(dict(key=key, **case['group'], arm=a, count=len(indices), overlap_primary=len(set(indices) & set(primary)),
                    signed_local_projection=float(selected_vector @ missing_write / max(denom, 1e-30)),
                    selected_vector_relative=float(np.linalg.norm(selected_vector - missing_write) / max(np.linalg.norm(missing_write), 1e-30)),
                    rounding_relative=float(np.linalg.norm(actual_delta - selected_vector) / max(np.linalg.norm(missing_write), 1e-30)),
                    individual_energy_fraction=float(all_energy[indices].sum() / max(all_energy.sum(), 1e-30))))
        print('281 audited', len(accounts), '/32', flush=True)
    datasets = dict(checks=checks, raw=raw, coefficients=coefficients, heads=heads, native=native, local=local, effect=effect, joint=joint, removal=removal, accounts=accounts, scores=scores, selection=selection_rows)
    for name, records in datasets.items():
        table(run / (name + '.csv'), records)
    def counts(records):
        return {a: sum(r['passes'] for r in records if r['arm'] == a) for a in EVAL}
    def strata(records, field, arm=None):
        return {s + '/' + str(v): sum(r[field] for r in records if r['schema'] == s and r['reverse'] == v and (arm is None or r['arm'] == arm)) for s in ('generic', 'descriptive') for v in (0, 1)}
    def accepted(records, field, arm=None):
        return sum(r[field] for r in records if arm is None or r['arm'] == arm) >= 24 and min(strata(records, field, arm).values()) >= 6
    save_json(run / 'audit.json', dict(status='passed', experiment=281, retrospective=True, design_sha256=sha(run / 'design.json'), status_sha256=sha(run / 'status.json'), source_sha256=sha(__file__),
        native_counts=counts(native), local_counts=counts(local), effect_counts=counts(effect), joint_counts=counts(joint), P_strata=strata(joint, 'passes', 'P'),
        primary_passes=bool(accepted(joint, 'passes', 'P')),
        account_counts={f: sum(r[f] for r in accounts) for f in ('restoration_pass', 'removal_pass', 'baseline_damage', 'behavioral_account_pass')},
        account_strata={f: strata(accounts, f) for f in ('restoration_pass', 'removal_pass', 'baseline_damage', 'behavioral_account_pass')},
        account_passes={f: bool(accepted(accounts, f)) for f in ('restoration_pass', 'removal_pass', 'baseline_damage', 'behavioral_account_pass')},
        numerical_checks=len(checks), numerical_max_error=max(r['error'] for r in checks), False_anchors=len(coefficients), coefficient_vectors=3 * len(coefficients),
        public_down_replays=down_replays, public_input_replays=input_replays, public_primal_replays=primal_replays,
        raw_checks=dict(total=len(raw), failed=sum(not r['passes'] for r in raw), original_bound=.0001), csv_sha256={p.name: sha(p) for p in run.glob('*.csv')}, ended=stamp(),
        boundary='Old contexts/finite other operations. No learned feature/circuit, whole-model efficiency, priority, fresh confirmation or fullQB. Keep all failed accounts and older numerical failures.'))
    print(dict(native=counts(native), joint=counts(joint), primary=accepted(joint, 'passes', 'P'), accounts={f: sum(r[f] for r in accounts) for f in ('restoration_pass', 'removal_pass', 'baseline_damage', 'behavioral_account_pass')}), flush=True)


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
