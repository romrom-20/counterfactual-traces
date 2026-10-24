"""Standard-library280/280b source, failure, all-case and endpoint verification."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source-composition-280b'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(name):
    with (DEST / (name + '.csv')).open() as stream:
        return list(csv.DictReader(stream))


def boolean(r, key='passes'):
    assert r[key] in ('True', 'False')
    return r[key] == 'True'


def vec(r, prefix, bound, cosine):
    return float(r[prefix + 'norm']) >= .001 and float(r[prefix + 'relative']) <= bound and float(r[prefix + 'cosine']) >= cosine


def main():
    manifest, design, status, audit = (read(DEST / (n + '.json')) for n in ('manifest', 'design', 'status', 'audit'))
    for n, h in manifest['files'].items():
        p = Path(n)
        assert not p.is_absolute() and not any(x in p.parts for x in ('..', '.context', 'drafts')) and p.suffix != '.npz'
        assert sha(ROOT / p) == h, n
    assert manifest['scientific_freeze'] == design['source_commit'] == '8d2eebb26b57e655eaa7418c50d35e49fed7b1ca'
    for n, h in design['source_hashes'].items():
        assert sha(ROOT / n) == h
    assert design['experiment'] == audit['experiment'] == 280 and design['retrospective'] and audit['retrospective']
    assert design['qualification'] == status['qualification'] == '280b-storage-only'
    assert audit['status'] == 'passed' and status['status'] == 'complete'
    assert audit['design_sha256'] == sha(DEST / 'design.json') and audit['status_sha256'] == sha(DEST / 'status.json')
    for n, h in audit['csv_sha256'].items():
        assert sha(DEST / n) == h
    fail = DEST / 'failed-original'
    original = read(fail / 'design.json')
    assert original['source_commit'] == 'bf937b04b749d6732115174b9082d6a5703e9558'
    assert sha(fail / 'design.json') == design['preserved_partial_design_sha256']
    assert sha(fail / 'failure.json') == design['preserved_failure_sha256']
    assert read(fail / 'failure.json')['error'] == 'AssertionError()'
    inv = read(fail / 'raw-array-inventory.json')['arrays']
    assert len(inv) == len(status['reused_unchanged_cases']) == 28 and status['newly_executed_cases'] == 4
    complete_inv = read(DEST / 'raw-array-inventory.json')['arrays']
    assert len(complete_inv) == 32 and sum(r['bytes'] for r in complete_inv) == status['raw_bytes']
    bypath = {r['path']: r for r in complete_inv}
    assert all(r == bypath[r['path']] for r in inv)
    native, trajectory, joint = (rows(n) for n in ('native', 'trajectory', 'joint'))
    for records in (native, trajectory, joint):
        assert len(records) == 288
    for r in native:
        expected = vec(r, 'all_', .05, .99) and vec(r, 'verdict_', .05, .99) and float(r['margin_error']) <= .1
        assert boolean(r) == expected
    for r in trajectory:
        assert boolean(r) == (vec(r, 'source_', .1, .995) and vec(r, 'V_', .1, .995))
    def lookup(rs):
        return {(r['key'], r['arm']): r for r in rs}
    ns, ts = lookup(native), lookup(trajectory)
    for r in joint:
        key = (r['key'], r['arm'])
        assert boolean(r, 'native_pass') == boolean(ns[key]) and boolean(r, 'trajectory_pass') == boolean(ts[key])
        assert boolean(r) == (boolean(ns[key]) and boolean(ts[key]))
    for name, data in [('native', native), ('trajectory', trajectory), ('joint', joint)]:
        assert {a: sum(boolean(r) for r in data if r['arm'] == a) for a in audit[name + '_counts']} == audit[name + '_counts']
    strata = {s + '/' + str(v): sum(boolean(r) for r in joint if r['arm'] == 'C' and r['schema'] == s and int(r['reverse']) == v) for s in ('generic', 'descriptive') for v in (0, 1)}
    assert audit['C_strata'] == strata and audit['primary_passes'] == (audit['joint_counts']['C'] >= 24 and min(strata.values()) >= 6)
    correction = rows('correction')
    assert len(correction) == 128
    for r in correction:
        expected = vec(r, 'all_', .25, .95) and vec(r, 'verdict_', .25, .95) and abs(float(r['target_margin'])) >= .02 and float(r['margin_error']) <= .005 + .25 * abs(float(r['target_margin']))
        assert boolean(r) == expected and boolean(r, 'account_pass') == (expected and int(r['random_failures']) >= 2)
    rc = sum(boolean(r, 'account_pass') for r in correction if r['arm'] == 'R')
    assert rc == audit['correction_account_count']
    ledger, coefficients, checks, raw = (rows(n) for n in ('ledger', 'coefficients', 'checks', 'raw'))
    assert len(ledger) == len(coefficients) == audit['False_gate_anchors'] == 896
    assert audit['coefficient_vectors_checked'] == 2688 and len(checks) == audit['numerical_checks']
    assert all(float(r['error']) <= float(r['bound']) for r in checks)
    assert all(float(r['closure']) <= 1e-10 for r in ledger)
    for r in coefficients:
        assert float(r['bound']) == .0001
        assert all(float(v) <= .0001 for n, v in r.items() if n not in ('key', 'layer', 'bound'))
    assert len(raw) == audit['raw_checks']['total']
    assert sum(not boolean(r) for r in raw) == audit['raw_checks']['failed']
    assert all(float(r['original_bound']) == .0001 and boolean(r) == (float(r['error']) <= .0001) for r in raw)
    source2 = DEST / 'source2-posthoc'
    post = read(source2 / 'source2-provenance.json')
    assert post['model_calls'] == 0 and post['posthoc'] and post['native_target_used']
    assert post['source_sha256'] == sha(ROOT / 'research/boolean_coupling/diagnose_source_gates.py')
    assert post['audit_sha256'] == sha(DEST / 'audit.json')
    with (source2 / 'source2-cases.csv').open() as stream:
        saved_cases = list(csv.DictReader(stream))
    assert len(saved_cases) == 32 and all(float(r['signed_closure_error']) <= 1e-10 for r in saved_cases)
    assert audit['public_down_replays'] == 11648
    assert status['counts']['full_forwards'] == 352 and status['counts']['C_current_finite_source_sites_per_forward'] == 0
    prov = read(ROOT / 'docs/figures/source-composition/provenance.json')
    assert prov['posthoc'] and prov['model_calls'] == 0 and not prov['predictions_modified']
    assert prov['script_sha256'] == sha(ROOT / 'research/boolean_coupling/describe_source_composition.py')
    for n, h in prov['source_csv_sha256'].items():
        assert sha(DEST / n) == h
    for n, h in prov['figure_sha256'].items():
        assert sha(ROOT / 'docs/figures/source-composition' / n) == h
    print('280b verified: original storage failure retained, all32 cases and source seals, unchanged scientific/numerical rules.', audit['joint_counts'], 'raw failures', audit['raw_checks']['failed'])


if __name__ == '__main__':
    main()
