"""Standard-library verification of281d/e, including the failed primary and audits."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source2-products-281d'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(n):
    with (DEST / (n + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def boolean(r, key='passes'):
    assert r[key] in ('True', 'False')
    return r[key] == 'True'


def vec(r, prefix='', bound=.25, cosine=.95):
    return float(r[prefix + 'norm']) >= .001 and float(r[prefix + 'relative']) <= bound and float(r[prefix + 'cosine']) >= cosine


def effect(r):
    return vec(r, 'all_') and vec(r, 'verdict_') and abs(float(r['target_margin'])) >= .02 and float(r['margin_error']) <= .005 + .25 * abs(float(r['target_margin']))


def lookup(data):
    result = {(r['key'], r['arm']): r for r in data}
    assert len(result) == len(data)
    return result


def main():
    manifest, design, status, audit, qualified = (read(DEST / (n + '.json')) for n in ('manifest', 'design', 'status', 'audit', 'audit-e-design'))
    for name, h in manifest['files'].items():
        p = Path(name)
        assert not p.is_absolute() and not any(x in p.parts for x in ('..', '.context', 'drafts')) and p.suffix != '.npz'
        assert sha(ROOT / p) == h, name
    assert manifest['scientific_freeze'] == design['source_commit'] == 'db93c39edb190f274687ca2744a8cdf69f5ddb14'
    assert manifest['original_scientific_freeze'] == 'e96526f24fd8d02f1221867629ba733399ac9941'
    assert manifest['audit_qualification_freeze'] == qualified['source_commit'] == '6a5378ae6087b4cba15be131eef13a7e697dcd4e'
    for name, h in (design['source_hashes'] | qualified['source_hashes']).items():
        assert sha(ROOT / name) == h
    assert design['experiment'] == audit['experiment'] == 281 and design['retrospective'] and audit['retrospective']
    assert design['qualification'] == status['qualification'] == '281d-resume-path-binding'
    assert audit['qualification'] == qualified['qualification'] == '281e-generator-order' and not qualified['exact_bound_changed']
    assert audit['status'] == 'passed' and status['status'] == 'complete'
    assert audit['design_sha256'] == qualified['design_sha256'] == sha(DEST / 'design.json')
    assert audit['status_sha256'] == qualified['status_sha256'] == sha(DEST / 'status.json')
    assert audit['audit_qualification_sha256'] == sha(DEST / 'audit-e-design.json')
    assert qualified['original_failure_sha256'] == sha(DEST / 'failure-audit.json')
    assert audit['source_sha256'] == sha(ROOT / 'research/boolean_coupling/audit281e.py')
    assert 'random_direction' in read(DEST / 'failure-audit.json')['error'] and "'bound': 0.0" in read(DEST / 'failure-audit.json')['error']
    for name, h in audit['csv_sha256'].items():
        assert sha(DEST / name) == h
    failures = {label: read(DEST / label / 'design.json') for label in ('failed-original', 'failed-281b', 'failed-281c')}
    assert failures['failed-original']['source_commit'] == manifest['original_scientific_freeze']
    assert read(DEST / 'failed-original/failure.json')['error'] == "KeyError('B_source')"
    assert read(DEST / 'failed-281b/failure.json')['error'] == 'AssertionError()'
    assert 'NpzFile' in read(DEST / 'failed-281c/failure.json')['error']
    assert design['partial_design_sha256'] == sha(DEST / 'failed-281b/design.json')
    assert design['partial_failure_sha256'] == sha(DEST / 'failed-281b/failure.json')
    assert failures['failed-281c']['partial_design_sha256'] == design['partial_design_sha256']
    for d in failures.values():
        for name, h in d['source_hashes'].items():
            assert sha(ROOT / name) == h
    inventory = read(DEST / 'raw-array-inventory.json')['arrays']
    bypath = {r['path']: r for r in inventory}
    assert len(bypath) == len(status['cases']) == len(design['cases']) == 32
    assert sum(r['bytes'] for r in inventory) == status['raw_bytes'] == 71405398
    cases = {c['key']: c for c in design['cases']}
    assert set(cases) == {r['key'] for r in status['cases']}
    for entry in status['cases']:
        key = entry['key']
        p = DEST / 'cases' / (key + '.json')
        assert sha(p) == entry['metadata_sha256']
        meta = read(p)
        assert meta['arrays_sha256'] == bypath['cases/' + key + '.npz']['sha256'] == entry['arrays_sha256']
        assert all(meta[n] for n in ('native_source_input_not_used_for_masks', 'direct_product_restoration_exact', 'fixed_F_B_forecasts_before_full_source2_controls', 'archive_replays_exact'))
        assert set(meta['arms']) == set(design['arms'])
        for arm, d in meta['arms'].items():
            assert len(d['prefix_hashes']) == 28 and d['prefix_hashes'] == meta['arms']['F']['prefix_hashes']
            assert d['other_rows_unchanged'] and len(d['product_hashes']) == 2
            if arm in ('P', 'D', 'H', 'R0', 'R1', 'R2', 'FZ'):
                assert d['current_MLP2_finite_coordinates'] == 90
        assert meta['arms']['P']['product_hashes'] == meta['arms']['D']['product_hashes']
    assert len(status['reused_unchanged_cases']) == 30 and status['newly_executed_cases'] == 2
    for label in ('failed-281b', 'failed-281c'):
        inv = read(DEST / label / 'raw-array-inventory.json')['arrays']
        assert len(inv) == 31 and sum(r['metadata_present'] for r in inv) == 30
        assert all(r['sha256'] == bypath[r['path']]['sha256'] for r in inv)
    native, local, actual, joint, removal, accounts, selection = (rows(n) for n in ('native', 'local', 'effect', 'joint', 'removal', 'accounts', 'selection'))
    assert all(len(data) == 480 for data in (native, local, actual, joint))
    assert len(removal) == 128 and len(accounts) == 32 and len(selection) == 160
    for r in native:
        assert boolean(r) == (vec(r, 'all_', .05, .99) and vec(r, 'verdict_', .05, .99) and float(r['margin_error']) <= .1)
    for r in local:
        assert boolean(r) == vec(r)
    for r in actual:
        assert boolean(r) == effect(r)
    ns, ls, es, js, rs = (lookup(data) for data in (native, local, actual, joint, removal))
    for r in joint:
        key = r['key'], r['arm']
        expected = boolean(ns[key]) and boolean(ls[key]) and boolean(es[key])
        assert boolean(r) == expected
        assert boolean(r, 'native_pass') == boolean(ns[key]) and boolean(r, 'local_pass') == boolean(ls[key]) and boolean(r, 'effect_pass') == boolean(es[key])
    for r in removal:
        assert boolean(r) == (vec(r, 'local_') and effect(r))
    for r in accounts:
        key = r['key']
        mask_fail = sum(not (boolean(ls[key, a]) and boolean(es[key, a])) for a in ('R0', 'R1', 'R2'))
        direction_fail = sum(not (boolean(ls[key, a]) and boolean(es[key, a])) for a in ('G0', 'G1', 'G2'))
        removal_fail = sum(not boolean(rs[key, a]) for a in ('A0', 'A1', 'A2'))
        restore = boolean(js[key, 'P']) and mask_fail >= 2 and direction_fail >= 2
        remove = boolean(rs[key, 'A']) and removal_fail >= 2
        damage = not boolean(ns[key, 'B'])
        assert [int(r[n]) for n in ('mask_failures', 'direction_failures', 'removal_control_failures')] == [mask_fail, direction_fail, removal_fail]
        assert [boolean(r, n) for n in ('restoration_pass', 'removal_pass', 'baseline_damage', 'behavioral_account_pass')] == [restore, remove, damage, restore and remove and damage]
    for name, data in (('native', native), ('local', local), ('effect', actual), ('joint', joint)):
        assert {a: sum(boolean(r) for r in data if r['arm'] == a) for a in audit[name + '_counts']} == audit[name + '_counts']
    strata = {s + '/' + str(v): sum(boolean(r) for r in joint if r['arm'] == 'P' and r['schema'] == s and int(r['reverse']) == v) for s in ('generic', 'descriptive') for v in (0, 1)}
    assert audit['P_strata'] == strata and audit['primary_passes'] == (audit['joint_counts']['P'] >= 24 and min(strata.values()) >= 6)
    for name in audit['account_counts']:
        count = sum(boolean(r, name) for r in accounts)
        strata = {s + '/' + str(v): sum(boolean(r, name) for r in accounts if r['schema'] == s and int(r['reverse']) == v) for s in ('generic', 'descriptive') for v in (0, 1)}
        assert count == audit['account_counts'][name] and strata == audit['account_strata'][name]
        assert audit['account_passes'][name] == (count >= 24 and min(strata.values()) >= 6)
    assert not audit['primary_passes'] and audit['joint_counts']['P'] == 0 and audit['native_counts']['P'] == audit['native_counts']['H'] == 32
    assert audit['account_counts'] == dict(restoration_pass=0, removal_pass=0, baseline_damage=16, behavioral_account_pass=0)
    assert all(int(r['count']) == 90 and math.isfinite(float(r['signed_local_projection'])) for r in selection)
    checks, coefficients, raw, heads = (rows(n) for n in ('checks', 'coefficients', 'raw', 'heads'))
    assert len(checks) == audit['numerical_checks'] == 6816 and all(float(r['error']) <= float(r['bound']) for r in checks)
    assert max(float(r['error']) for r in checks) == audit['numerical_max_error']
    assert len(coefficients) == audit['False_anchors'] == 64 and audit['coefficient_vectors'] == 192
    for r in coefficients:
        assert float(r['bound']) == .0001
        assert all(float(v) <= .0001 for n, v in r.items() if n not in ('key', 'site', 'bound'))
    assert len(raw) == audit['raw_checks']['total'] == 2304
    assert sum(not boolean(r) for r in raw) == audit['raw_checks']['failed'] == 0
    assert all(float(r['original_bound']) == .0001 and boolean(r) == (float(r['error']) <= .0001) for r in raw)
    assert len(heads) == 576 and all(float(r['public_error']) == 0 and float(r['cpu_error']) <= .005 for r in heads)
    assert (audit['public_down_replays'], audit['public_input_replays'], audit['public_primal_replays']) == (768, 192, 1152)
    assert status['counts']['full_forwards'] == 576 and not status['counts']['forecast_full_current_source2_activation']
    notes = read(DEST / 'execution-notes.json')
    assert notes['total_checkpoint_producer_forwards'] == 614 and notes['failed_attempt_additional_forwards'] == 38
    assert not any(notes[n] for n in ('scientific_thresholds_changed', 'old_evidence_deleted', 'private_article_included', 'fresh_confirmation', 'learned_circuit', 'general_question_B_solved'))
    figure = ROOT / 'docs/figures/source2-products'
    prov = read(figure / 'provenance.json')
    assert prov['posthoc'] and prov['model_calls'] == 0 and not prov['predictions_modified']
    assert prov['script_sha256'] == sha(ROOT / 'research/boolean_coupling/describe_source2_products.py')
    for name, h in prov['source_csv_sha256'].items():
        assert sha(DEST / name) == h
    for name, h in prov['figure_sha256'].items():
        assert sha(figure / name) == h
    geometry = DEST / 'geometry-posthoc'
    post = read(geometry / 'provenance.json')
    assert post['posthoc'] and post['native_targets_used'] and post['model_forwards'] == 0
    assert post['source_sha256'] == sha(ROOT / 'research/boolean_coupling/diagnose_product_geometry.py')
    assert post['audit_sha256'] == sha(DEST / 'audit.json')
    assert post['source_arrays'] == {r['key']: r['arrays_sha256'] for r in status['cases']}
    for name, h in post['csv_sha256'].items():
        assert sha(geometry / name) == h
    with (geometry / 'geometry-cases.csv').open() as stream:
        gr = list(csv.DictReader(stream))
    with (geometry / 'rectifier-ledger.csv').open() as stream:
        ledger = list(csv.DictReader(stream))
    assert len(gr) == 32 and len(ledger) == 96 and {r['key'] for r in gr} == set(cases)
    assert all(float(r['identity_error']) <= 1e-12 and float(r['signed_closure_error']) <= 1e-10 for r in ledger)
    assert all(abs(sum(float(r[n + '_signed_projection']) for n in ('proxy', 'hump_change', 'anchor_rounding', 'remaining_rounding')) - 1) <= 1e-10 for r in ledger)
    print('281d/e verified: all32 old cases; primary0/32, nativeP/H32/32; all failed attempts/original audit retained; no fresh/circuit/fullQB claim.')


if __name__ == '__main__':
    main()
