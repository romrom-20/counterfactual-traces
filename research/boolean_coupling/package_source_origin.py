"""Publish audited282 metadata and all outcomes, with private arrays excluded."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source2-origin-282'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    assert not p.exists(), p
    p.write_text(json.dumps(d, indent=2) + '\n')


def main(run, logs, partial, geometry, seal):
    if not seal:
        assert read(run/'audit.json')['status']=='passed' and not DEST.exists()
        DEST.mkdir()
        arrays=[]
        for p in sorted(run.rglob('*')):
            if not p.is_file() or any(n in p.relative_to(run).parts for n in ('sources', 'audit-b-sources', 'audit-c-sources')):
                continue
            rel=p.relative_to(run)
            if p.suffix=='.npz':
                arrays.append(dict(path=str(rel),sha256=sha(p),bytes=p.stat().st_size))
                continue
            assert p.suffix in ('.json','.csv')
            q=DEST/rel
            q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,q)
        save(DEST/'raw-array-inventory.json',dict(omitted=True,arrays=arrays,
            reference='source2-products-281d',lossless_aliases=True,regeneration='research/README.md#source2-origin-282'))
        shutil.copytree(geometry,DEST/'geometry-posthoc')
        # Preserve the incomplete run and prove that the cold retry reproduced
        # its sealed observations. This does not make that partial run complete.
        assert not (partial/'status.json').exists()
        failure=DEST/'preserved-storage-stop'
        failure.mkdir()
        for n in ('design.json','runtime.json','failure.json'):
            shutil.copy2(partial/n,failure/n)
        partial_cases=[]
        for p in sorted((partial/'cases').glob('*.npz')):
            m=p.with_suffix('.json')
            assert sha(p)==read(m)['arrays_sha256']
            q=run/'cases'/p.name
            assert sha(p)==sha(q) and sha(m)==sha(q.with_suffix('.json'))
            partial_cases.append(dict(key=p.stem,arrays_sha256=sha(p),metadata_sha256=sha(m),bytes=p.stat().st_size))
        assert len(partial_cases)==23
        save(failure/'inventory.json',dict(status='incomplete_storage_guard',cases=partial_cases,
            raw_bytes=sum(r['bytes'] for r in partial_cases),no_orphan_case24=True,
            full_forwards=345,local_corner_maps=92,local_down_maps=184,
            cold_retry_all_shared_arrays_and_metadata_identical=True,
            numerical_outcomes_interpreted_before_audit=False))
        records=[]
        for stage in ('282-synthetic-producer-audit','282-synthetic-producer-audit-v2','282-producer','282-retry-producer','282-retry-audit',
                      '282b-synthetic-audit','282b-synthetic-audit-v2','282b-qualified-audit',
                      '282c-synthetic-producer-audit','282c-qualified-audit'):
            p=logs/(stage+'.log')
            lines=[line for line in p.read_text().splitlines() if line.startswith(('282 ',"{'passed':","{'native':",'Traceback','AssertionError','KeyError','AttributeError','ValueError','real ','user ','sys '))]
            q=DEST/(stage+'.log')
            q.write_text('\n'.join(line.rstrip() for line in lines)+'\n')
            records.append(dict(stage=stage,raw_sha256=sha(p),published_sha256=sha(q),representation='Progress/outcomes/error/timing, local paths omitted and trailing whitespace removed; original retained.'))
        save(DEST/'log-provenance.json',dict(logs=records))
        save(DEST/'execution-notes.json',dict(prefreeze_initial_CPU_guard_rejection=True,checkpoint_started_after_full_CPU_producer_and_audit=True,
            numerical_thresholds_changed=False,reference_arrays_modified=False,reference_arrays_aliased_by_exact_bytes=True,
            numerical_validation_scope_changed=True,original_audit_passed=False,
            paired_whole_secondary_unavailable=True,paired_secondary_scope_changed=True,primary_criteria_changed=False,
            original_equal_length_synthetic_fixture_missed_shape_bug=True,
            unequal_length_full_synthetic_producer_and_audit_passed_before_282c_freeze=True,
            unsealed_qualified_CPU_fixture_name_shadowing_failure=True,
            qualified_CPU_fixture_passed_before_freeze_and_real_audit=True,
            original_checkpoint_stopped_at_free_space_guard=True,original_completed_cases=23,
            original_failed_attempt_full_forwards=345,cold_retry_full_forwards=480,total_checkpoint_producer_full_forwards=825,
            source_runtime_storage_policies_unchanged_for_retry=True,partial_reuse=False,
            original_partial_shared_arrays_and_metadata_identical=True,
            synthetic_reference_full_forwards=128,synthetic_new_full_forwards=480,synthetic_local_corner_maps=128,
            independent_research_team=False,fresh_confirmation=False,learned_circuit=False,general_question_B_solved=False,private_article_included=False))
        broken=logs/'282b-synthetic-audit/audit-repetition/audit-b-sources'
        save(DEST/'unsealed-CPU-fixture-failure.json', dict(real_checkpoint=False,
            error="AttributeError: 'numpy.ndarray' object has no attribute 'append'",
            cause='Unsealed audit used local for a projection array, shadowing its local-outcomes list. Renamed cpu_local before the passing fixture and freeze; no actual qualified model audit ran before that pass.',
            original_source_hashes={p.name:sha(p) for p in sorted(broken.iterdir())},
            failed_log_raw_sha256=sha(logs/'282b-synthetic-audit.log')))
        return
    design=read(DEST/'design.json')
    qualification=read(DEST/'audit-b-design.json')
    analysis=read(DEST/'audit-c-design.json')
    source_hashes={**design['source_hashes'],**qualification['source_hashes'],**analysis['source_hashes']}
    for name,h in source_hashes.items():
        assert sha(ROOT/name)==h
    paths=[p for p in DEST.rglob('*') if p.is_file()]
    paths += [ROOT/n for n in source_hashes]
    paths += [ROOT/n for n in ('docs/analysis/282-source2-gate-origin.md','research/boolean_coupling/package_source_origin.py',
        'research/boolean_coupling/verify_source_origin.py','research/boolean_coupling/describe_source_origin.py',
        'research/boolean_coupling/diagnose_origin_geometry.py')]
    paths += list((ROOT/'docs/figures/source2-origin').iterdir())
    assert all('.context' not in p.relative_to(ROOT).parts and 'drafts' not in p.parts and p.suffix!='.npz' for p in paths)
    save(DEST/'manifest.json',dict(release=DEST.name,scientific_freeze=design['source_commit'],
        numerical_qualification_freeze=qualification['source_commit'],
        paired_analysis_qualification_freeze=analysis['source_commit'],
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},scope='All-case source, alias, qualification and endpoint integrity; not independent model replication.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--logs',type=Path,required=True)
    p.add_argument('--partial',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--seal',action='store_true')
    a=p.parse_args()
    main(a.run.resolve(),a.logs.resolve(),a.partial.resolve(),a.geometry.resolve(),a.seal)
