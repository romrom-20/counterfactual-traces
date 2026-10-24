"""Publish qualified281d evidence, preserving the original failed producer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source2-products-281d'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    assert not p.exists(), p
    p.write_text(json.dumps(d, indent=2) + '\n')


def main(run, failed, failed_b, failed_c, logs, seal):
    if not seal:
        assert read(run / 'audit.json')['status'] == 'passed' and not DEST.exists()
        DEST.mkdir()
        arrays = []
        for p in sorted(run.rglob('*')):
            if not p.is_file() or any(x.endswith('sources') for x in p.relative_to(run).parts):
                continue
            rel = p.relative_to(run)
            if p.suffix == '.npz':
                arrays.append(dict(path=str(rel), sha256=sha(p), bytes=p.stat().st_size))
                continue
            assert p.suffix in ('.json', '.csv')
            q = DEST / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
        old = DEST / 'failed-original'
        old.mkdir()
        for name in ('design.json', 'runtime.json', 'failure.json'):
            shutil.copy2(failed / name, old / name)
        save(old / 'boundary.json', dict(completed_cases=0, model_forwards_before_failure=2,
            reason='B_source read before stored; fixed storage order only in separately frozen281b.', scientific_outcomes_issued=False))
        for label, directory in (('failed-281b', failed_b), ('failed-281c', failed_c)):
            dst = DEST / label
            dst.mkdir()
            for name in ('design.json', 'runtime.json', 'failure.json'):
                shutil.copy2(directory / name, dst / name)
            arrays = [dict(path='cases/' + p.name, sha256=sha(p), bytes=p.stat().st_size, metadata_present=p.with_suffix('.json').exists()) for p in sorted((directory / 'cases').glob('*.npz'))]
            save(dst / 'raw-array-inventory.json', dict(omitted=True, arrays=arrays))
        # Restore the completed inventory, not the failed-attempt local variable.
        arrays = [dict(path='cases/' + p.name, sha256=sha(p), bytes=p.stat().st_size) for p in sorted((run / 'cases').glob('*.npz'))]
        save(DEST / 'execution-notes.json', dict(original_bookkeeping_failure=True, original_cap_failure=True, resume_binding_failure=True,
            completed_case_forwards=576, failed_attempt_additional_forwards=38, total_checkpoint_producer_forwards=614,
            reused_exact_cases=30, orphan31_regenerated_exact=True, scientific_thresholds_changed=False, old_evidence_deleted=False,
            preflight281c_failed=True, preflight281c_gating_was_missed=True, preflight281d_cold_and_resume_passed=True,
            private_article_included=False, fresh_confirmation=False, learned_circuit=False, general_question_B_solved=False))
        records = []
        for name in ('281-producer', '281b-producer', '281c-producer', '281d-producer', '281d-audit', '281e-audit'):
            p = logs / (name + '.log')
            lines = [line for line in p.read_text().splitlines() if line.startswith(('281 ', '281b ', '281c ', '281d ', 'Traceback', 'KeyError', 'AssertionError', 'AttributeError', "{'native':", 'real ', 'user ', 'sys '))]
            q = DEST / (name + '.log')
            q.write_text('\n'.join(lines) + '\n')
            records.append(dict(stage=name, raw_sha256=sha(p), published_sha256=sha(q), representation='Scientific progress/results/error/timing, local paths omitted; original retained.'))
        save(DEST / 'log-provenance.json', dict(logs=records))
        save(DEST / 'raw-array-inventory.json', dict(omitted=True, arrays=arrays, regeneration='research/README.md#source2-products-281d'))
        return
    design = read(DEST / 'design.json')
    qualified = read(DEST / 'audit-e-design.json')
    hashes = design['source_hashes'] | qualified['source_hashes']
    for n, h in hashes.items():
        assert sha(ROOT / n) == h
    paths = [p for p in DEST.rglob('*') if p.is_file()]
    paths += [ROOT / n for n in hashes]
    paths += [ROOT / n for n in ('docs/analysis/281-source2-products.md',
        'research/boolean_coupling/package_source2_products.py', 'research/boolean_coupling/describe_source2_products.py',
        'research/boolean_coupling/verify_source2_products.py', 'research/boolean_coupling/diagnose_product_geometry.py',
        'research/boolean_coupling/qualify281e.py')]
    paths += list((ROOT / 'docs/figures/source2-products').iterdir())
    assert all(p.suffix != '.npz' and 'drafts' not in p.parts and '.context' not in p.relative_to(ROOT).parts for p in paths)
    save(DEST / 'manifest.json', dict(release=DEST.name, scientific_freeze=design['source_commit'],
        original_scientific_freeze='e96526f24fd8d02f1221867629ba733399ac9941', audit_qualification_freeze=qualified['source_commit'],
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}, scope='All-case/source/failure integrity, not independent model replication.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--failed-281', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--failed-281b', type=Path, required=True)
    parser.add_argument('--failed-281c', type=Path, required=True)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    main(args.run.resolve(), args.failed_281.resolve(), args.failed_281b.resolve(), args.failed_281c.resolve(), args.logs.resolve(), args.seal)
