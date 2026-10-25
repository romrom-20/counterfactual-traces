"""282b: public Float32 up-map qualification; original CPU64 failures retained."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as fn

from .audit229 import sha
from .audit272 import margin, read, table, vector
from .audit279 import coefficient_checks, maximum
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head, weights
from .composition280 import digest, padded, polynomial
from .core225b import array, load_model
from .origin282 import ARMS, EVAL, ROOT
from .run272 import stamp
from .source279 import derivatives


def good(v, bound=.25, cosine=.95):
    return bool(v['norm'] >= .001 and v['relative'] <= bound and v['cosine'] >= cosine)


def diff(d, a, b, n):
    return d[a + '_' + n].astype(float) - d[b + '_' + n].astype(float)


def metrics(value, target):
    v = vector(value, target)
    v['signed_projection'] = float(np.vdot(value, target) / max(np.vdot(target, target), 1e-30))
    return v


def main(run, reference):
    assert not (run / 'audit.json').exists() and (run / 'failure-audit.json').exists()
    qualification = read(run/'audit-b-design.json')
    assert qualification['qualification'] == '282b-public-Float32-up-with-rounding-envelope'
    for n, h in qualification['source_hashes'].items():
        assert sha(ROOT/n) == sha(run/'audit-b-sources'/Path(n).name) == h
    assert qualification['design_sha256'] == sha(run/'design.json')
    assert qualification['status_sha256'] == sha(run/'status.json')
    assert qualification['original_failure_sha256'] == sha(run/'failure-audit.json')
    design, status = (read(run / (n + '.json')) for n in ('design', 'status'))
    assert design['experiment'] == 282 and design['retrospective'] and status['status'] == 'complete'
    assert len(status['cases']) == 32 and tuple(design['arms']) == ARMS
    assert design['installed_sources'] == installed_sources()
    for n, h in design['source_hashes'].items():
        assert sha(ROOT / n) == sha(run / 'sources' / Path(n).name) == h
    for n, h in design['reference_hashes'].items():
        assert sha(reference / n) == h
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design['checkpoint']
    model, head = load_model(True), read_head(folder)
    config = read(folder / 'config.json')
    with safe_open(folder / 'model.safetensors', framework='pt', device='cpu') as handle:
        w = weights(handle, config, 1, 'mlp')
    down = w['down_weight']
    cases = {c['key']: c for c in design['cases']}
    entries = {e['key']: e for e in read(reference / 'status.json')['cases']}
    checks, raw, coefficients, native, local, effect, removals, ledger, heads, pairs, joint, accounts = ([] for _ in range(12))
    bank, public_corner_maps, public_down_maps, primal_maps = {}, 0, 0, 0
    up_bridges = []

    def check(key, name, a, b, bound=0.):
        error = maximum(a, b)
        checks.append(dict(key=key, name=name, error=error, bound=bound))
        assert np.isfinite(error) and error <= bound, checks[-1]

    def rawcheck(key, corner, node, a, b, required=False):
        error = maximum(a, b)
        raw.append(dict(key=key, corner=corner, node=node, error=error, original_bound=.0001, passes=bool(np.isfinite(error) and error <= .0001), required_bridge=required, original_required_bridge=node in ('norm', 'gate', 'up', 'phi')))
        if required:
            check(key, 'CPU64_' + corner + '_' + node, error, 0., .0001)

    def outcome(key, group, arm, delta, target, scores, desired_scores, bound, cosine, effect_margin=False):
        row = dict(key=key, **group, arm=arm)
        oks = []
        for scope, pick in (('all', slice(1, None)), ('verdict', -1)):
            v = metrics(delta[pick], target[pick])
            row.update({scope + '_' + n: x for n, x in v.items()})
            oks.append(good(v, bound, cosine))
        me = abs(scores - desired_scores)
        row.update(margin_error=me)
        if effect_margin:
            row.update(target_margin=desired_scores)
            passed = all(oks) and abs(desired_scores) >= .02 and me <= .005 + .25 * abs(desired_scores)
        else:
            passed = all(oks) and me <= .1
        row['passes'] = bool(passed)
        return row

    for entry in status['cases']:
        key, case = entry['key'], cases[entry['key']]
        group, s, length = case['group'], case['position'], len(case['false_ids'])
        path, old_path = run / 'cases' / (key + '.npz'), reference / 'cases' / (key + '.npz')
        assert sha(path) == entry['arrays_sha256'] and sha(path.with_suffix('.json')) == entry['metadata_sha256']
        meta = read(path.with_suffix('.json'))
        assert sha(old_path) == entries[key]['arrays_sha256'] == meta['reference_arrays_sha256']
        assert all(meta[n] for n in ('candidate_inputs_own_B', 'candidates_before_full_calibration', 'archive_replays_exact', 'only_source2_products_or_matched_writes_changed', 'full_control_exact', 'zero_exact'))
        with np.load(path) as archive, np.load(old_path) as old, torch.no_grad():
            d = {n: archive[n] for n in archive.files}
            for name, alias in meta['array_aliases'].items():
                assert set(alias) in ({'reference'}, {'local'}) and name not in d
                d[name] = old[alias['reference']] if 'reference' in alias else d[alias['local']]
            assert set(d) == set(meta['logical_arrays'])
            for name, value in d.items():
                expected = meta['logical_arrays'][name]
                assert str(value.dtype) == expected['dtype'] and list(value.shape) == expected['shape'] and digest(value) == expected['sha256']
            for arm in ('F', 'B', 'C1', 'T'):
                for n in ('source', 'final', 'scores', 'write'):
                    check(key, 'archive_' + arm + '_' + n, d[arm + '_' + n], old[arm + '_' + n])
            for n in ('source', 'final', 'scores', 'write'):
                check(key, 'zero_' + n, d['FZ_' + n], d['F_' + n])
                check(key, 'full_' + n, d['U_' + n], d['C1_' + n])
            for site in (1, 2):
                gf, pf = (d[f'F_node{site}_' + n] for n in ('gate', 'phi'))
                independent, errors = coefficient_checks(gf, derivatives(gf))
                coefficients.append(dict(key=key, site=site, bound=.0001, **errors))
                for n, error in errors.items():
                    check(key, n, error, 0., .0001)
                for arm in ('F', 'B'):
                    g, phi = (d[f'{arm}_node{site}_' + n] for n in ('gate', 'phi'))
                    if arm == 'B':
                        dg = g.astype(float) - gf.astype(float)
                        expected = pf.astype(float) + independent['d1'] * dg + .5 * independent['d2'] * dg**2 + independent['d3'] * dg**3 / 6
                        check(key, 'independent_cubic_' + str(site), phi, expected.astype(np.float32), .0001)
                    primal = model.model.layers[site-1].mlp.act_fn(padded(gf, length, s, model.device))
                    check(key, 'False_public_primal', array(primal[0, s]), pf)
                    primal_maps += 1
                if site == 1:
                    layer1 = model.model.layers[0]
                    for arm in ('F', 'B'):
                        full = padded(d[arm + '_node1_mid'], length, s, model.device)
                        z = layer1.post_attention_layernorm(full)
                        gg, uu = layer1.mlp.gate_proj(z), layer1.mlp.up_proj(z)
                        check(key, 'first_public_gate', array(gg[0, s]), d[arm + '_node1_gate'])
                        check(key, 'first_public_up', array(uu[0, s]), d[arm + '_node1_up'])
                        phi = padded(d[arm + '_node1_phi'], length, s, model.device)
                        ww = layer1.mlp.down_proj(phi * uu)
                        check(key, 'first_public_down', array(ww[0, s]), d[arm + '_node1_raw_write'])
                        check(key, 'first_source_addition', d[arm + '_node1_mid'] + d[arm + '_node1_raw_write'], d[arm + '_source'][0])
                        public_down_maps += 1
            xf = d['F_node2_mid'].astype(float)
            carry = diff(d, 'B', 'F', 'source')[0]
            attention = (d['B_node2_mid'].astype(float) - d['B_source'][0].astype(float)) - (xf - d['F_source'][0].astype(float))
            expected_points = dict(F=xf.astype(np.float32), C=(xf+carry).astype(np.float32), A=(xf+attention).astype(np.float32), U=d['B_node2_mid'])
            check(key, 'carry_provenance', d['component_carry'], carry)
            check(key, 'attention_provenance', d['component_attention'], attention)
            check(key, 'input_closure', xf + carry + attention, d['B_node2_mid'], 1e-10)
            anchor = dict(gate=d['F_node2_gate'], phi=d['F_node2_phi'])
            rs = {}
            for corner in ('F', 'C', 'A', 'U'):
                prefix = 'corner_' + corner + '_'
                check(key, 'point_' + corner, d['point_' + corner], expected_points[corner])
                check(key, 'corner_mid_' + corner, d[prefix + 'mid'], expected_points[corner])
                layer = model.model.layers[1]
                full = padded(d['point_' + corner], length, s, model.device)
                z = layer.post_attention_layernorm(full)
                g, up = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
                phi = layer.mlp.act_fn(g)
                cubic = padded(polynomial(anchor, array(g[0, s]), 3), length, s, model.device)
                pf, pc = phi*up, cubic*up
                wf, wc = layer.mlp.down_proj(pf), layer.mlp.down_proj(pc)
                for n, x in dict(norm=z, gate=g, up=up, phi=phi, cubic=cubic, finite_product=pf, cubic_product=pc, finite_write=wf, cubic_write=wc).items():
                    check(key, 'public_corner_' + corner + '_' + n, array(x[0, s]), d[prefix + n])
                public_corner_maps += 1
                public_down_maps += 2
                primal_maps += 1
                x = d['point_' + corner].astype(float)
                zz = x / np.sqrt(np.mean(x*x) + w['eps']) * w['norm']
                gg, uu = (zz @ w[n + '_weight'].T + w.get(n + '_bias', 0.) for n in ('gate', 'up'))
                ph = fn.silu(torch.tensor(d[prefix + 'gate'].astype(float))).numpy()
                for n, value in dict(norm=zz, gate=gg, up=uu, phi=ph).items():
                    rawcheck(key, corner, n, value, d[prefix + n], n != 'up')
                # A new, narrower contract: exact public Float32 projection plus
                # a dimension-derived rounding envelope. This does NOT pass the
                # original absolute CPU64 up bridge or identify a learned feature.
                consumed = d[prefix+'norm'].astype(float)
                weight, bias = w['up_weight'], w.get('up_bias', 0.)
                observed = d[prefix+'up'].astype(float)
                cpu_local = consumed @ weight.T + bias
                propagated = (consumed-zz) @ weight.T
                dot_error = observed-cpu_local
                total_error = observed-uu
                check(key, 'up_signed_numerical_closure', total_error, propagated+dot_error, 1e-10)
                unit = np.finfo(np.float32).eps/2
                operations = 2*consumed.size+3
                gamma = operations*unit/(1-operations*unit)
                propagation_bound = np.abs(consumed-zz) @ np.abs(weight).T
                dot_bound = gamma*(np.abs(consumed) @ np.abs(weight).T + np.abs(bias))
                ratio = float(np.max(np.abs(total_error)/np.maximum(propagation_bound+dot_bound, np.finfo(float).tiny)))
                check(key, 'up_Float32_rounding_envelope_ratio', ratio, 0., 1.)
                up_bridges.append(dict(key=key, corner=corner, original_error=maximum(observed, uu),
                    original_bound=.0001, original_passes=bool(maximum(observed, uu)<=.0001),
                    consumed_input_error=maximum(observed, cpu_local), consumed_original_passes=bool(maximum(observed, cpu_local)<=.0001),
                    propagated_norm_error=float(np.max(np.abs(propagated))),
                    signed_closure=maximum(total_error, propagated+dot_error),
                    operations=operations, Float32_unit_roundoff=unit, gamma=gamma,
                    envelope_max=float(np.max(propagation_bound+dot_bound)), envelope_ratio=ratio))
                rawcheck(key, corner, 'finite_product', ph*d[prefix + 'up'].astype(float), d[prefix + 'finite_product'])
                rawcheck(key, corner, 'finite_write', d[prefix + 'finite_product'].astype(float) @ down.T, d[prefix + 'finite_write'])
                rawcheck(key, corner, 'cubic_write', d[prefix + 'cubic_product'].astype(float) @ down.T, d[prefix + 'cubic_write'])
                rs[corner] = d[prefix + 'finite_product'].astype(float) - d[prefix + 'cubic_product'].astype(float)
            check(key, 'zero_remainder_product', rs['F'], np.zeros_like(rs['F']))
            check(key, 'False_corner_write', d['corner_F_finite_write'], d['F_write'])
            check(key, 'base_corner_write', d['corner_U_cubic_write'], d['B_write'])
            check(key, 'full_corner_write', d['corner_U_finite_write'], d['C1_write'])
            planned = dict(K=rs['C'], W=rs['A'], S=rs['C']+rs['A'], U=rs['U'], X=rs['U']-rs['C']-rs['A'], NC=rs['U']-rs['C'], NW=rs['U']-rs['A'])
            base_product = d['corner_U_cubic_product']
            target_write = diff(d, 'C1', 'B', 'write')
            for arm in ARMS:
                am = meta['arms'][arm]
                assert am['prefix_hashes'] == meta['arms']['F']['prefix_hashes'] and len(am['prefix_hashes']) == len(model.model.layers)
                assert am['other_rows_unchanged'] and am['product_delta_installed'] == (arm in planned)
                if arm not in ('F', 'FZ', 'T'):
                    check(key, 'unchanged_source1', d[arm + '_source'][0], d['B_source'][0])
                if arm in planned:
                    product = (base_product.astype(float) + planned[arm]).astype(np.float32)
                elif arm in ('F', 'FZ'):
                    product = d['corner_F_finite_product']
                elif arm == 'C1':
                    product = d['corner_U_finite_product']
                elif arm == 'T':
                    product = None
                else:
                    product = base_product
                if product is not None:
                    assert digest(product) == am['product_hashes'][1]
                    ww = model.model.layers[1].mlp.down_proj(padded(product, length, s, model.device))
                    check(key, 'public_installed_down', array(ww[0, s]), d[arm + '_raw_write'])
                    public_down_maps += 1
                    if arm.startswith('G'):
                        i = int(arm[-1])
                        seed = int(hashlib.sha256((key + f':282-write:{i}').encode()).hexdigest()[:16], 16)
                        draw = np.random.default_rng(seed).normal(size=target_write.shape)
                        correction = diff(d, 'K', 'B', 'write')
                        expected = (draw*np.linalg.norm(correction)) / np.linalg.norm(draw)
                        check(key, 'random_direction_exact', d['draw_' + arm], expected)
                        check(key, 'random_norm', np.linalg.norm(expected), np.linalg.norm(correction), 1e-10)
                        expected_write = (d[arm + '_raw_write'].astype(float)+expected).astype(np.float32)
                    else:
                        expected_write = d[arm + '_raw_write']
                    check(key, 'installed_write', d[arm + '_write'], expected_write)
                    mid = d['F_node2_mid'] if arm in ('F', 'FZ') else d['B_node2_mid']
                    check(key, 'source2_addition', mid + d[arm + '_write'], d[arm + '_source'][1])
                if arm not in ('F', 'T', 'C1'):
                    assert am['actual_gate_hashes'] == meta['arms']['F']['actual_gate_hashes']
                    assert am['primal_hashes'] == meta['arms']['F']['primal_hashes']
                    assert am['current_MLP2_finite_coordinates'] == 0
                full = torch.zeros((1, length, config['hidden_size']), device=model.device)
                full[0, s:] = torch.tensor(d[arm + '_final'], device=model.device)
                actual = array(model.lm_head(model.model.norm(full)[:, -1:])[0, -1, [895, 830]])
                public_error = maximum(actual, d[arm + '_scores'])
                check(key, 'public_full_vocab_head', actual, d[arm + '_scores'])
                h = d[arm + '_final'][-1].astype(float)
                cpu = h/np.sqrt(np.mean(h*h)+head['final_eps'])*head['final_norm'] @ head['final_head'].T
                check(key, 'CPU64_head', cpu, d[arm + '_scores'], .005)
                heads.append(dict(key=key, arm=arm, public_error=public_error, cpu_error=maximum(cpu, d[arm + '_scores'])))
            target_final = diff(d, 'C1', 'B', 'final')
            native_target = diff(d, 'T', 'F', 'final')
            target_margin = margin(d['C1_scores'])-margin(d['B_scores'])
            for arm in EVAL:
                native.append(outcome(key, group, arm, diff(d, arm, 'F', 'final'), native_target, margin(d[arm + '_scores']), margin(d['T_scores']), .05, .99))
                v = metrics(diff(d, arm, 'B', 'write'), target_write)
                local.append(dict(key=key, **group, arm=arm, **v, passes=good(v)))
                effect.append(outcome(key, group, arm, diff(d, arm, 'B', 'final'), target_final, margin(d[arm + '_scores'])-margin(d['B_scores']), target_margin, .25, .95, True))
            for removed in ('NC', 'NW', 'S'):
                v = metrics(diff(d, 'U', removed, 'write'), target_write)
                row = outcome(key, group, removed, diff(d, 'U', removed, 'final'), target_final, margin(d['U_scores'])-margin(d[removed + '_scores']), target_margin, .25, .95, True)
                row.update({'local_' + n: x for n, x in v.items()})
                row['passes'] = bool(row['passes'] and good(v))
                removals.append(row)
            def partition(name, target, pieces):
                rounding = target-sum(pieces.values())
                pieces = {**pieces, 'rounding_or_downstream_nonadditivity': rounding}
                projections = []
                for component, value in pieces.items():
                    v = metrics(value, target)
                    projections.append(v['signed_projection'])
                    ledger.append(dict(key=key, **group, quantity=name, component=component, **v, component_norm=float(np.linalg.norm(value))))
                check(key, name + '_closure', sum(pieces.values()), target, 1e-10)
                if np.linalg.norm(target) >= .001:
                    check(key, name + '_signed_closure', sum(projections), 1., 1e-10)
            partition('product', rs['U'], dict(carry=rs['C'], attention=rs['A'], interaction=planned['X']))
            partition('write', target_write, {n: down @ value for n, value in dict(carry=rs['C'], attention=rs['A'], interaction=planned['X']).items()})
            for n in ('gate', 'up'):
                target = d['corner_U_' + n].astype(float)-d['corner_F_' + n].astype(float)
                c = d['corner_C_' + n].astype(float)-d['corner_F_' + n].astype(float)
                a = d['corner_A_' + n].astype(float)-d['corner_F_' + n].astype(float)
                partition(n, target, dict(carry=c, attention=a, interaction=target-c-a))
            for scope, pick in (('receiving', slice(1, None)), ('verdict', -1)):
                partition(scope, target_final[pick], {n: diff(d, a, 'B', 'final')[pick] for n, a in dict(carry='K', attention='W', interaction='X').items()})
            bank[key] = {n: d[n].astype(float).copy() for arm in EVAL for n in (arm + '_write', arm + '_final', arm + '_scores')}
        print('282 audited', len(bank), '/32', flush=True)
    # Pair the existing backgrounds exactly; no selection by observed effect sizes.
    grouped = {}
    for key, case in cases.items():
        group = case['group']
        index = tuple(group[n] for n in ('assignment', 'block', 'reverse', 'target'))
        assert group['schema'] not in grouped.setdefault(index, {})
        grouped[index][group['schema']] = key
    assert len(grouped) == 16 and all(set(v) == {'generic', 'descriptive'} for v in grouped.values())
    pair_pass = {}
    for index, names in grouped.items():
        g, desc = (bank[names[n]] for n in ('generic', 'descriptive'))
        def contrast(a, b, n):
            return (desc[a + '_' + n]-desc[b + '_' + n])-(g[a + '_' + n]-g[b + '_' + n])
        for arm in EVAL:
            v = metrics(contrast(arm, 'B', 'write'), contrast('C1', 'B', 'write'))
            row = dict(pair_key='|'.join(map(str, index)), generic_key=names['generic'], descriptive_key=names['descriptive'], arm=arm,
                **{'local_' + n: value for n, value in v.items()}, local_pass=good(v))
            target = contrast('C1', 'B', 'final')
            candidate = contrast(arm, 'B', 'final')
            oks = []
            for scope, pick in (('all', slice(1, None)), ('verdict', -1)):
                vv = metrics(candidate[pick], target[pick])
                row.update({scope + '_' + n: value for n, value in vv.items()})
                oks.append(good(vv))
            target_margin = (margin(desc['C1_scores'])-margin(desc['B_scores']))-(margin(g['C1_scores'])-margin(g['B_scores']))
            me = abs((margin(desc[arm + '_scores'])-margin(desc['B_scores']))-(margin(g[arm + '_scores'])-margin(g['B_scores']))-target_margin)
            row.update(target_margin=target_margin, margin_error=me, effect_pass=bool(all(oks) and abs(target_margin)>=.02 and me <= .005+.25*abs(target_margin)))
            pairs.append(row)
            for key in names.values():
                pair_pass[key, arm] = bool(row['local_pass'])
    ns, ls, es = ({(r['key'], r['arm']): r for r in data} for data in (native, local, effect))
    for key, case in cases.items():
        for arm in EVAL:
            ix = key, arm
            flags = dict(native_pass=ns[ix]['passes'], local_pass=ls[ix]['passes'], effect_pass=es[ix]['passes'], paired_local_pass=pair_pass[ix])
            joint.append(dict(key=key, **case['group'], arm=arm, **flags, passes=bool(all(flags.values()))))
        failed_random = sum(not (ls[key, a]['passes'] and es[key, a]['passes']) for a in ('G0', 'G1', 'G2'))
        restore = next(r['passes'] for r in joint if r['key']==key and r['arm']=='K') and failed_random>=2
        remove = next(r['passes'] for r in removals if r['key']==key and r['arm']=='NC')
        accounts.append(dict(key=key, **case['group'], random_failures=failed_random, selective_restoration_pass=bool(restore), removal_pass=bool(remove), baseline_damage=not ns[key, 'B']['passes']))
    datasets = dict(checks=checks, raw=raw, coefficients=coefficients, native=native, local=local, effect=effect,
        removal=removals, ledger=ledger, heads=heads, paired=pairs, joint=joint, accounts=accounts, up_bridges=up_bridges)
    for n, data in datasets.items():
        table(run / (n + '.csv'), data)
    def counts(data):
        return {a: sum(r['passes'] for r in data if r['arm']==a) for a in EVAL}
    def strata(data, field='passes', arm='K'):
        return {s+'/'+str(v):sum(r[field] for r in data if r['schema']==s and r['reverse']==v and (arm is None or r['arm']==arm)) for s in ('generic','descriptive') for v in (0,1)}
    def accepted(data, field='passes', arm='K'):
        return sum(r[field] for r in data if arm is None or r['arm']==arm)>=24 and min(strata(data, field, arm).values())>=6
    save_json(run / 'audit.json', dict(status='passed', experiment=282, retrospective=True, qualification=qualification['qualification'],
        qualification_sha256=sha(run/'audit-b-design.json'), original_audit_passed=False,
        original_chained_up_failures=sum(not r['original_passes'] for r in up_bridges),
        consumed_up_failures=sum(not r['consumed_original_passes'] for r in up_bridges),
        up_rounding_envelope_max_ratio=max(r['envelope_ratio'] for r in up_bridges), design_sha256=sha(run/'design.json'), status_sha256=sha(run/'status.json'),
        source_sha256=sha(__file__), native_counts=counts(native), local_counts=counts(local), effect_counts=counts(effect), joint_counts=counts(joint),
        paired_local_counts={a:sum(r['local_pass'] for r in pairs if r['arm']==a) for a in EVAL}, paired_effect_counts={a:sum(r['effect_pass'] for r in pairs if r['arm']==a) for a in EVAL},
        K_strata=strata(joint), primary_passes=bool(accepted(joint)), account_counts={n:sum(r[n] for r in accounts) for n in ('selective_restoration_pass','removal_pass','baseline_damage')},
        account_passes={n:bool(accepted(accounts,n,None)) for n in ('selective_restoration_pass','removal_pass','baseline_damage')}, numerical_checks=len(checks), numerical_max_error=max(r['error'] for r in checks),
        False_anchors=len(coefficients), coefficient_vectors=3*len(coefficients), public_corner_maps=public_corner_maps, public_down_maps=public_down_maps, public_primal_maps=primal_maps,
        raw_checks=dict(total=len(raw),failed=sum(not r['passes'] for r in raw),original_bound=.0001), csv_sha256={p.name:sha(p) for p in run.glob('*.csv')}, ended=stamp(),
        boundary='Qualified public Float32 up projections with dimension-derived rounding envelope; original absolute CPU64 up checks remain failed. Retrospective source-input/product interventions, rich finite budget. No learned feature/circuit, efficient/fresh predictor, priority or fullQB claim; preserve all failed accounts/older numerical failures.'))
    print(dict(native=counts(native), joint=counts(joint), primary=accepted(joint), account_counts={n:sum(r[n] for r in accounts) for n in ('selective_restoration_pass','removal_pass','baseline_damage')}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference-281d', type=Path, required=True)
    args = p.parse_args()
    try:
        main(args.run.resolve(), args.reference_281d.resolve())
    except Exception as exc:
        if not (args.run/'failure-audit-b.json').exists():
            save_json(args.run/'failure-audit-b.json',dict(error=repr(exc),time=stamp()))
        raise
