"""Standard-library282 verification: all endpoints, references and original bounds."""

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'results/source2-origin-282'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(n):
    with (DEST/(n+'.csv')).open() as stream:
        return list(csv.DictReader(stream))


def boolean(r,n='passes'):
    assert r[n] in ('True','False')
    return r[n]=='True'


def vec(r,p='',bound=.25,cosine=.95):
    return float(r[p+'norm'])>=.001 and float(r[p+'relative'])<=bound and float(r[p+'cosine'])>=cosine


def effect(r):
    return vec(r,'all_') and vec(r,'verdict_') and abs(float(r['target_margin']))>=.02 and float(r['margin_error'])<=.005+.25*abs(float(r['target_margin']))


def lookup(data):
    result={(r['key'],r['arm']):r for r in data}
    assert len(result)==len(data)
    return result


def main():
    manifest,design,status,audit=(read(DEST/(n+'.json')) for n in ('manifest','design','status','audit'))
    assert manifest['scientific_freeze']==design['source_commit']=='183e2083f5c5591efab5fd47b78fbc3114319e72'
    for n,h in manifest['files'].items():
        p=Path(n)
        assert not p.is_absolute() and not any(x in p.parts for x in ('..','.context','drafts')) and p.suffix!='.npz'
        assert sha(ROOT/p)==h,n
    for n,h in design['source_hashes'].items():
        assert sha(ROOT/n)==h
    reference=ROOT/'results/source2-products-281d'
    for n,h in design['reference_hashes'].items():
        assert sha(reference/n)==h
    assert design['experiment']==audit['experiment']==282 and design['retrospective'] and audit['retrospective']
    assert status['status']=='complete' and audit['status']=='passed'
    assert audit['design_sha256']==sha(DEST/'design.json') and audit['status_sha256']==sha(DEST/'status.json')
    assert audit['source_sha256']==sha(ROOT/'research/boolean_coupling/audit282c.py')
    qualification=read(DEST/'audit-b-design.json')
    assert qualification['qualification']==audit['qualification']=='282b-public-Float32-up-with-rounding-envelope'
    assert manifest['numerical_qualification_freeze']==qualification['source_commit']=='9332b5caa1b07783e0a146942172c1dd85bd22fb'
    assert audit['qualification_sha256']==sha(DEST/'audit-b-design.json')
    assert qualification['design_sha256']==sha(DEST/'design.json') and qualification['status_sha256']==sha(DEST/'status.json')
    assert qualification['original_failure_sha256']==sha(DEST/'failure-audit.json')
    assert qualification['numerical_validation_scope_changed'] and not qualification['original_audit_passed'] and not audit['original_audit_passed']
    assert not qualification['behavioral_predictions_or_thresholds_changed'] and not qualification['original_absolute_tolerance_changed']
    for n,h in qualification['source_hashes'].items():
        assert sha(ROOT/n)==h
    analysis=read(DEST/'audit-c-design.json')
    assert analysis['qualification']==audit['analysis_qualification']=='282c-paired-whole-secondary-unavailable'
    assert analysis['source_commit']==manifest['paired_analysis_qualification_freeze']=='cb9cfc8fbcc460976b000f52db711ed688c5433d'
    assert audit['analysis_qualification_sha256']==sha(DEST/'audit-c-design.json')
    assert analysis['design_sha256']==sha(DEST/'design.json') and analysis['status_sha256']==sha(DEST/'status.json')
    assert analysis['original_failure_b_sha256']==sha(DEST/'failure-audit-b.json')
    assert analysis['numerical_qualification_sha256']==sha(DEST/'audit-b-design.json')
    assert analysis['unequal_length_pairs']==16 and analysis['original_paired_whole_secondary_unavailable']
    assert not analysis['primary_criteria_changed'] and not analysis['forecasts_changed'] and not analysis['numerical_scope_changed_from_282b']
    for n,h in analysis['source_hashes'].items():
        assert sha(ROOT/n)==h
    diagnostic=DEST/'input-bridge-diagnostic'
    input_diagnostic=read(diagnostic/'diagnostic.json')
    assert qualification['diagnostic_sha256']==sha(diagnostic/'diagnostic.json')
    assert input_diagnostic['source_sha256']==sha(ROOT/'research/boolean_coupling/diagnose_input_bridges282.py') and input_diagnostic['csv_sha256']==sha(diagnostic/'bridges.csv')
    assert input_diagnostic['numerical_only'] and input_diagnostic['posthoc'] and input_diagnostic['model_forwards']==0 and not input_diagnostic['behavior_outcomes_inspected']
    assert input_diagnostic['design_sha256']==sha(DEST/'design.json') and input_diagnostic['status_sha256']==sha(DEST/'status.json') and input_diagnostic['failure_sha256']==sha(DEST/'failure-audit.json')
    assert design['storage']==dict(initial_free_MiB=500,per_case_free_MiB=448,new_array_cap_MiB=40,lossless_identical_array_aliases=True)
    for n,h in audit['csv_sha256'].items():
        assert sha(DEST/n)==h
    old={r['key']:r for r in read(reference/'status.json')['cases']}
    cases={c['key']:c for c in design['cases']}
    assert set(cases)==set(old) and len(status['cases'])==32
    inventory=read(DEST/'raw-array-inventory.json')['arrays']
    bypath={r['path']:r for r in inventory}
    assert len(inventory)==32 and sum(r['bytes'] for r in inventory)==status['raw_bytes']<=40*1024**2
    for entry in status['cases']:
        key=entry['key']
        p=DEST/'cases'/(key+'.json')
        assert sha(p)==entry['metadata_sha256']
        meta=read(p)
        assert meta['arrays_sha256']==entry['arrays_sha256']==bypath['cases/'+key+'.npz']['sha256']
        assert meta['reference_arrays_sha256']==old[key]['arrays_sha256']
        assert all(meta[n] for n in ('candidate_inputs_own_B','candidates_before_full_calibration','archive_replays_exact','only_source2_products_or_matched_writes_changed','full_control_exact','zero_exact'))
        assert set(meta['arms'])==set(design['arms'])
        for arm,data in meta['arms'].items():
            assert len(data['prefix_hashes'])==28 and data['prefix_hashes']==meta['arms']['F']['prefix_hashes']
            assert data['other_rows_unchanged']
            assert data['product_delta_installed']==(arm in ('K','W','S','U','X','NC','NW'))
            if arm not in ('F','T','C1'):
                assert data['actual_gate_hashes']==meta['arms']['F']['actual_gate_hashes']
                assert data['primal_hashes']==meta['arms']['F']['primal_hashes'] and data['current_MLP2_finite_coordinates']==0
        logical=meta['logical_arrays']
        assert all(len(r['sha256'])==64 and r['dtype'] in ('float32','float64') for r in logical.values())
        assert all(name in logical for name in meta['array_aliases'])
        for name,alias in meta['array_aliases'].items():
            assert set(alias) in ({'reference'},{'local'})
            if 'local' in alias:
                assert alias['local'] not in meta['array_aliases'] and logical[name]==logical[alias['local']]
    native,local,actual,paired,joint,removal,accounts=(rows(n) for n in ('native','local','effect','paired','joint','removal','accounts'))
    assert all(len(r)==416 for r in (native,local,actual,joint)) and len(paired)==208 and len(removal)==96 and len(accounts)==32
    for r in native:
        assert boolean(r)==(vec(r,'all_',.05,.99) and vec(r,'verdict_',.05,.99) and float(r['margin_error'])<=.1)
    for r in local:
        assert boolean(r)==vec(r)
    for r in actual:
        assert boolean(r)==effect(r)
    pair_pass={}
    for r in paired:
        assert boolean(r,'local_pass')==vec(r,'local_')
        assert not boolean(r,'effect_pass') and not boolean(r,'original_secondary_effect_evaluable')
        assert int(r['generic_receiving_rows'])!=int(r['descriptive_receiving_rows'])
        verdict=vec(r,'verdict_') and abs(float(r['target_margin']))>=.02 and float(r['margin_error'])<=.005+.25*abs(float(r['target_margin']))
        assert boolean(r,'verdict_only_pass')==verdict
        g,d=(r[n] for n in ('generic_key','descriptive_key'))
        assert cases[g]['group']['schema']=='generic' and cases[d]['group']['schema']=='descriptive'
        assert all(cases[g]['group'][n]==cases[d]['group'][n] for n in ('assignment','block','reverse','target'))
        for key in (g,d):
            pair_pass[key,r['arm']]=boolean(r,'local_pass')
    ns,ls,es,js=(lookup(data) for data in (native,local,actual,joint))
    for r in joint:
        ix=r['key'],r['arm']
        flags=[boolean(ns[ix]),boolean(ls[ix]),boolean(es[ix]),pair_pass[ix]]
        assert [boolean(r,n) for n in ('native_pass','local_pass','effect_pass','paired_local_pass')]==flags
        assert boolean(r)==all(flags)
    for r in removal:
        assert boolean(r)==(vec(r,'local_') and effect(r))
    rs=lookup(removal)
    for r in accounts:
        key=r['key']
        failures=sum(not (boolean(ls[key,a]) and boolean(es[key,a])) for a in ('G0','G1','G2'))
        assert int(r['random_failures'])==failures
        assert boolean(r,'selective_restoration_pass')==(boolean(js[key,'K']) and failures>=2)
        assert boolean(r,'removal_pass')==boolean(rs[key,'NC'])
        assert boolean(r,'baseline_damage')==(not boolean(ns[key,'B']))
    for name,data in (('native',native),('local',local),('effect',actual),('joint',joint)):
        assert {a:sum(boolean(r) for r in data if r['arm']==a) for a in audit[name+'_counts']}==audit[name+'_counts']
    for field in ('local','effect'):
        assert {a:sum(boolean(r,field+'_pass') for r in paired if r['arm']==a) for a in audit['paired_'+field+'_counts']}==audit['paired_'+field+'_counts']
    assert audit['paired_whole_secondary_evaluable']==0 and not audit['primary_criteria_changed']
    assert {a:sum(boolean(r,'verdict_only_pass') for r in paired if r['arm']==a) for a in audit['paired_verdict_only_counts']}==audit['paired_verdict_only_counts']
    def strata(data,field,arm=None):
        return {s+'/'+str(v):sum(boolean(r,field) for r in data if r['schema']==s and int(r['reverse'])==v and (arm is None or r['arm']==arm)) for s in ('generic','descriptive') for v in (0,1)}
    expected=strata(joint,'passes','K')
    assert audit['K_strata']==expected and audit['primary_passes']==(audit['joint_counts']['K']>=24 and min(expected.values())>=6)
    assert audit['native_counts']['K']==audit['local_counts']['K']==audit['effect_counts']['K']==32
    assert audit['joint_counts']['K']==audit['paired_local_counts']['K']==0 and not audit['primary_passes']
    assert audit['joint_counts']['NW']==32 and audit['paired_local_counts']['NW']==16
    assert audit['account_counts']['removal_pass']==32 and audit['account_counts']['baseline_damage']==16
    for name in audit['account_counts']:
        count=sum(boolean(r,name) for r in accounts)
        assert count==audit['account_counts'][name]
        assert audit['account_passes'][name]==(count>=24 and min(strata(accounts,name).values())>=6)
    checks,raw,coeff,heads,ledger=(rows(n) for n in ('checks','raw','coefficients','heads','ledger'))
    assert len(checks)==audit['numerical_checks'] and all(float(r['error'])<=float(r['bound']) for r in checks)
    assert max(float(r['error']) for r in checks)==audit['numerical_max_error']
    assert len(coeff)==audit['False_anchors']==64 and audit['coefficient_vectors']==192
    for r in coeff:
        assert float(r['bound'])==.0001 and all(float(v)<=.0001 for n,v in r.items() if n not in ('key','site','bound'))
    assert len(raw)==audit['raw_checks']['total']==896 and sum(not boolean(r) for r in raw)==audit['raw_checks']['failed']
    assert all(float(r['original_bound'])==.0001 and boolean(r)==(float(r['error'])<=.0001) for r in raw)
    assert all(boolean(r) for r in raw if boolean(r,'required_bridge'))
    assert all(boolean(r,'original_required_bridge')==(r['node'] in ('norm','gate','up','phi')) for r in raw)
    assert all(boolean(r,'required_bridge')==(r['node'] in ('norm','gate','phi')) for r in raw)
    up=rows('up_bridges')
    assert len(up)==128
    assert sum(not boolean(r,'original_passes') for r in up)==audit['original_chained_up_failures']==input_diagnostic['original_chained_failures']
    assert sum(not boolean(r,'consumed_original_passes') for r in up)==audit['consumed_up_failures']
    raw_up={(r['key'],r['corner']):r for r in raw if r['node']=='up'}
    for r in up:
        assert float(r['original_bound'])==.0001 and float(r['original_error'])==float(raw_up[r['key'],r['corner']]['error'])
        assert boolean(r,'original_passes')==(float(r['original_error'])<=.0001)
        assert boolean(r,'consumed_original_passes')==(float(r['consumed_input_error'])<=.0001)
        assert float(r['signed_closure'])<=1e-10 and 0<=float(r['envelope_ratio'])<=1
        assert int(r['operations'])==3075
        # NumPy Float32 scalars were serialized with their short decimal form.
        assert abs(float(r['Float32_unit_roundoff'])-2**-24)<=2**-47
        k,u=int(r['operations']),2**-24
        expected_gamma=k*u/(1-k*u)
        assert abs(float(r['gamma'])-expected_gamma)<=expected_gamma*2**-23
    assert max(float(r['envelope_ratio']) for r in up)==audit['up_rounding_envelope_max_ratio']
    assert len(heads)==480 and all(float(r['public_error'])==0 and float(r['cpu_error'])<=.005 for r in heads)
    assert len(ledger)==768 and audit['public_corner_maps']==128 and audit['public_down_maps']==768 and audit['public_primal_maps']==256
    for key in cases:
        for quantity in ('product','write','gate','up','receiving','verdict'):
            selected=[r for r in ledger if r['key']==key and r['quantity']==quantity]
            assert len(selected)==4
            if float(selected[0]['norm'])>=.001:
                assert abs(sum(float(r['signed_projection']) for r in selected)-1)<=1e-10
    assert status['counts']['full_forwards']==480 and status['counts']['local_finite_corner_maps']==128 and status['counts']['local_down_maps']==256
    assert status['counts']['other_source_sites_finite']==26 and status['counts']['all_receiving_operations_finite']
    failure=DEST/'preserved-storage-stop'
    partial=read(failure/'inventory.json')
    prior=read(failure/'design.json')
    assert prior['source_hashes']==design['source_hashes'] and prior['source_commit']==design['source_commit']
    assert prior['storage']==design['storage'] and prior['cases']==design['cases'] and prior['arms']==design['arms']
    assert read(failure/'failure.json')['error']=='AssertionError()'
    assert partial['status']=='incomplete_storage_guard' and len(partial['cases'])==23
    entries={r['key']:r for r in status['cases']}
    for entry in partial['cases']:
        assert entry==entries[entry['key']]
    assert partial['raw_bytes']==sum(r['bytes'] for r in partial['cases'])==24294205
    assert partial['full_forwards']==345 and partial['local_corner_maps']==92 and partial['local_down_maps']==184
    assert partial['no_orphan_case24'] and partial['cold_retry_all_shared_arrays_and_metadata_identical']
    assert not partial['numerical_outcomes_interpreted_before_audit']
    notes=read(DEST/'execution-notes.json')
    assert notes['prefreeze_initial_CPU_guard_rejection'] and notes['checkpoint_started_after_full_CPU_producer_and_audit']
    assert notes['numerical_validation_scope_changed'] and not notes['original_audit_passed']
    assert notes['paired_whole_secondary_unavailable'] and notes['paired_secondary_scope_changed'] and not notes['primary_criteria_changed']
    assert notes['original_equal_length_synthetic_fixture_missed_shape_bug'] and notes['unequal_length_full_synthetic_producer_and_audit_passed_before_282c_freeze']
    assert notes['unsealed_qualified_CPU_fixture_name_shadowing_failure'] and notes['qualified_CPU_fixture_passed_before_freeze_and_real_audit']
    assert not any(notes[n] for n in ('numerical_thresholds_changed','reference_arrays_modified','independent_research_team','fresh_confirmation','learned_circuit','general_question_B_solved','private_article_included'))
    assert notes['original_checkpoint_stopped_at_free_space_guard'] and notes['original_completed_cases']==23
    assert notes['original_failed_attempt_full_forwards']==345 and notes['cold_retry_full_forwards']==480 and notes['total_checkpoint_producer_full_forwards']==825
    assert notes['source_runtime_storage_policies_unchanged_for_retry'] and not notes['partial_reuse']
    assert notes['original_partial_shared_arrays_and_metadata_identical']
    logs=read(DEST/'log-provenance.json')['logs']
    assert len(logs)==10
    for r in logs:
        assert sha(DEST/(r['stage']+'.log'))==r['published_sha256']
    geometry=DEST/'geometry-posthoc'
    gp=read(geometry/'provenance.json')
    assert gp['posthoc'] and gp['model_forwards']==0 and not gp['predictions_modified']
    assert gp['uses_saved_full_calibration'] and not gp['direct_primitive_interventions']
    assert gp['source_sha256']==sha(ROOT/'research/boolean_coupling/diagnose_origin_geometry.py') and gp['audit_sha256']==sha(DEST/'audit.json')
    assert gp['cases']==32 and gp['dependent_pairs']==16 and gp['checkpoint']==design['checkpoint']
    for n,h in gp['csv_sha256'].items():
        assert sha(geometry/n)==h
    with (geometry/'signed-ledger.csv').open() as stream:
        gl=list(csv.DictReader(stream))
    assert len(gl)==176 and gp['signed_closure_max']<=1e-10
    groups={}
    for r in gl:
        assert float(r['closure'])<=1e-10
        ratio=float(r['component_norm'])/max(float(r['target_norm']),1e-30)
        assert abs(float(r['relative_norm'])-ratio)<=1e-12
        assert abs(float(r['signed_projection'])-float(r['cosine'])*ratio)<=1e-10
        groups.setdefault((r['kind'],r['key']),[]).append(r)
    assert sum(kind=='local_mixed_write' for kind,key in groups)==32
    assert sum(kind=='paired_missing_write' for kind,key in groups)==16
    for (kind,key),group in groups.items():
        assert len(group)==(3 if kind=='local_mixed_write' else 5)
        assert abs(sum(float(r['signed_projection']) for r in group)-1)<=1e-10
    fig=ROOT/'docs/figures/source2-origin'
    prov=read(fig/'provenance.json')
    assert prov['posthoc'] and prov['model_calls']==0 and not prov['predictions_modified']
    assert prov['script_sha256']==sha(ROOT/'research/boolean_coupling/describe_source_origin.py') and prov['audit_sha256']==sha(DEST/'audit.json')
    assert prov['geometry_provenance_sha256']==sha(geometry/'provenance.json')
    for n,h in prov['source_csv_sha256'].items():
        assert sha(DEST/n)==h
    for n,h in prov['figure_sha256'].items():
        assert sha(fig/n)==h
    print('282 verified: all32 old contexts,16 paired backgrounds, frozen endpoint decisions and source seals; primary',audit['primary_passes'],'joint',audit['joint_counts']['K'],'raw failures',audit['raw_checks']['failed'],'— not fresh/circuit/fullQB.')


if __name__=='__main__':
    main()
