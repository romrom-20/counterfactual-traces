"""Package completed280 metadata; omit arrays, preserve scientific source seals."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'results/source-composition-280b'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def save(p, value):
    assert not p.exists(), p
    p.write_text(json.dumps(value, indent=2) + '\n')


def main(run, logs, partial, seal):
    if not seal:
        assert read(run / 'audit.json')['status'] == 'passed' and not DEST.exists()
        DEST.mkdir()
        arrays = []
        for p in sorted(run.rglob('*')):
            if not p.is_file() or 'sources' in p.relative_to(run).parts:
                continue
            rel = p.relative_to(run)
            if p.suffix == '.npz':
                arrays.append(dict(path=str(rel), sha256=sha(p), bytes=p.stat().st_size))
            else:
                assert p.suffix in ('.json', '.csv')
                q = DEST / rel
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
        save(DEST / 'raw-array-inventory.json', dict(omitted=True, arrays=arrays, regeneration='research/README.md#source-composition-280'))
        failed = DEST / 'failed-original'
        failed.mkdir()
        for name in ('design.json', 'runtime.json', 'failure.json'):
            shutil.copy2(partial / name, failed / name)
        original_arrays = [dict(path='cases/' + p.name, sha256=sha(p), bytes=p.stat().st_size) for p in sorted((partial / 'cases').glob('*.npz'))]
        save(failed / 'raw-array-inventory.json', dict(omitted=True, arrays=original_arrays))
        save(failed / 'boundary.json', dict(reason='Original280 stopped at768MiB per-case disk reserve before29; raw cap1200MiB also too small for32.',
            original_case_count=28, scientific_conclusions_issued=False, arrays_reused_unchanged=True,
            qualification='280b increases storage cap/reserve only; sourceEngine and original audit280/numerical/scientific criteria unchanged.'))
        records = []
        for stage, logname in (('failed-producer', '280-producer'), ('producer', '280b-producer'), ('audit', '280b-audit')):
            p = logs / (logname + '.log')
            lines = [line for line in p.read_text().splitlines() if line.startswith(('280 ', '280b ', 'Traceback', 'AssertionError', "{'native':", 'real ', 'user ', 'sys '))]
            q = DEST / (stage + '.log')
            q.write_text('\n'.join(lines) + '\n')
            records.append(dict(stage=stage, raw_sha256=sha(p), published_sha256=sha(q), representation='Progress/results/timing; path warnings omitted; original retained.'))
        save(DEST / 'log-provenance.json', dict(logs=records))
        return
    design = read(DEST / 'design.json')
    for n, h in design['source_hashes'].items():
        assert sha(ROOT / n) == h
    paths = [p for p in DEST.rglob('*') if p.is_file()]
    paths += [ROOT / n for n in design['source_hashes']]
    paths += [ROOT / n for n in ('docs/analysis/280-source-composition.md',
        'research/boolean_coupling/package_source_composition.py', 'research/boolean_coupling/describe_source_composition.py',
        'research/boolean_coupling/verify_source_composition.py', 'research/boolean_coupling/diagnose_source_gates.py')]
    paths += list((ROOT / 'docs/figures/source-composition').iterdir())
    assert all('.context' not in p.relative_to(ROOT).parts and 'drafts' not in p.parts and p.suffix != '.npz' for p in paths)
    save(DEST / 'manifest.json', dict(release=DEST.name, scientific_freeze=design['source_commit'],
        files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}, scope='Integrity of frozen source and all-case evidence, not independent model replication.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--partial-280', type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.logs.resolve(), args.partial_280.resolve(), args.seal)
