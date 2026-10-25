"""Frozen282 producer: own-input carry/read factorial, then direct interventions."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .composition280 import digest
from .core225b import load_model
from .origin282 import (
    ARMS,
    CANDIDATES,
    CONTROLS,
    ROOT,
    SOURCES,
    Engine,
    deltas,
    local_pair,
    pack_observations,
    random_writes,
    split_inputs,
)
from .run272 import runtime, stamp

FIRST = ('mid', 'gate', 'up', 'phi', 'raw_write')


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 500 * 1024**2
    old, complete = (read(reference / (n + '.json')) for n in ('design', 'status'))
    assert old['experiment'] == 281 and complete['status'] == 'complete' and read(reference / 'audit.json')['status'] == 'passed'
    assert len(old['cases']) == len(complete['cases']) == 32
    for name, h in old['source_hashes'].items():
        assert sha(ROOT / name) == h
    out.mkdir()
    (out / 'cases').mkdir()
    (out / 'sources').mkdir()
    source_hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in source_hashes:
        shutil.copy2(ROOT / n, out / 'sources' / Path(n).name)
    save_json(out / 'design.json', dict(experiment=282, retrospective=True, cases=old['cases'], arms=ARMS,
        checkpoint=locate(ROOT)[1], installed_sources=installed_sources(), source_hashes=source_hashes,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), began=stamp(),
        reference_hashes={n: sha(reference / n) for n in ('design.json', 'status.json', 'audit.json')},
        storage=dict(initial_free_MiB=500, per_case_free_MiB=448, new_array_cap_MiB=40, lossless_identical_array_aliases=True),
        primary='K original native5%/.99/.1nat AND local/actual C1-B write/effect25%/.95/material/.02nat/.005+.25abs(target) AND paired local missing-write contrast25%/.95/material; >=24/32 and>=6/8schema/order.',
        boundary='Old dependent contexts, own first-source cubic state and finite attention2, additional finite local corner maps; later26source/all receivers finite. No compact/fresh/circuit/priority/fullQB claim.'))
    model = load_model(True)
    save_json(out / 'runtime.json', runtime(model))
    original = {r['key']: r for r in complete['cases']}
    entries, raw_bytes = [], 0
    for case in old['cases']:
        assert shutil.disk_usage(out).free >= 448 * 1024**2
        key, values, metadata = case['key'], {}, {}
        engine = Engine(model, case)

        def execute(arm):
            with engine:
                packed, meta, nodes = engine.forward(arm)
            if arm != 'F':
                assert meta['prefix_hashes'] == metadata['F']['prefix_hashes']
            if arm not in ('F', 'FZ', 'T'):
                for n in ('mid', 'gate', 'up'):
                    assert np.array_equal(nodes[1][n], values['B_node2_' + n]), (arm, n)
                assert np.array_equal(packed['source'][0], values['B_source'][0])
            if arm == 'FZ':
                for n in ('source', 'final', 'scores', 'write'):
                    assert np.array_equal(packed[n], values['F_' + n]), ('zero', n)
            if arm in ('F', 'B'):
                for n in FIRST:
                    values[arm + '_node1_' + n] = nodes[0][n]
                for n in ('mid', 'gate', 'up', 'phi', 'write'):
                    values[arm + '_node2_' + n] = nodes[1][n]
            values.update({arm + '_' + n: v for n, v in packed.items() if n != 'phi'})
            metadata[arm] = meta
            return packed, nodes

        f, fn = execute('F')
        # B's incoming rows must be stored before the generic shared-input comparison.
        with engine:
            b, bm, bn = engine.forward('B')
        values.update({'B_' + n: v for n, v in b.items() if n != 'phi'})
        metadata['B'] = bm
        assert bm['prefix_hashes'] == metadata['F']['prefix_hashes']
        for n in FIRST:
            values['B_node1_' + n] = bn[0][n]
        for n in ('mid', 'gate', 'up', 'phi', 'write'):
            values['B_node2_' + n] = bn[1][n]
        points, components = split_inputs(dict(mid=fn[1]['mid'], before=f['source'][0]), dict(mid=bn[1]['mid'], before=b['source'][0]))
        values.update({'point_' + n: v for n, v in points.items()})
        values.update({'component_' + n: v for n, v in components.items()})
        corners = {n: local_pair(model, case, engine.anchors[1], points[n]) for n in ('F', 'C', 'A')}
        assert np.array_equal(corners['F']['finite_product'], corners['F']['cubic_product'])
        assert np.array_equal(corners['F']['finite_write'], values['F_write'])
        engine.base_product = (bn[1]['phi'] * bn[1]['up']).astype(np.float32)
        assert bm['product_hashes'][1] == digest(engine.base_product)
        engine.product_deltas = deltas(corners)
        for arm in CANDIDATES:
            packed, _ = execute(arm)
            if arm == 'K':
                engine.draws = random_writes(case, packed['write'].astype(float) - b['write'].astype(float))
                values.update({'draw_' + n: v for n, v in engine.draws.items()})
        # Complete finite-input calibration only after all restricted candidates.
        corners['U'] = local_pair(model, case, engine.anchors[1], points['U'])
        assert np.array_equal(corners['U']['cubic_product'], engine.base_product)
        assert np.array_equal(corners['U']['cubic_write'], values['B_write'])
        for n in ('mid', 'gate', 'up'):
            assert np.array_equal(corners['U'][n], values['B_node2_' + n])
        engine.product_deltas = deltas(corners)
        for arm in CONTROLS:
            execute(arm)
        for n in ('source', 'final', 'scores', 'write'):
            assert np.array_equal(values['U_' + n], values['C1_' + n]), ('full', n)
        values.update({'corner_' + n + '_' + node: row for n, c in corners.items() for node, row in c.items()})
        source = reference / 'cases' / (key + '.npz')
        assert sha(source) == original[key]['arrays_sha256']
        path = out / 'cases' / (key + '.npz')
        with np.load(source) as archived:
            for arm in ('F', 'B', 'C1', 'T'):
                for n in ('source', 'final', 'scores', 'write'):
                    assert np.array_equal(values[arm + '_' + n], archived[arm + '_' + n]), ('archive', arm, n)
            aliases, logical = pack_observations(path, values, archived)
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 40 * 1024**2
        save_json(path.with_suffix('.json'), dict(key=key, arrays_sha256=sha(path), arms=metadata,
            array_aliases=aliases, logical_arrays=logical,
            candidate_inputs_own_B=True, candidates_before_full_calibration=True, archive_replays_exact=True,
            only_source2_products_or_matched_writes_changed=True, full_control_exact=True, zero_exact=True,
            reference_arrays_sha256=sha(source)))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix('.json')), bytes=path.stat().st_size))
        print('282', len(entries), '/32 complete', flush=True)
    save_json(out / 'status.json', dict(status='complete', cases=entries, raw_bytes=raw_bytes, ended=stamp(),
        counts=dict(full_forwards=len(ARMS) * len(old['cases']), local_finite_corner_maps=4 * len(old['cases']), local_down_maps=8 * len(old['cases']),
            source_cubic_sites=2, other_source_sites_finite=len(model.model.layers)-2, all_receiving_operations_finite=True, preparatory_B_forward_per_case=1)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-281d', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    try:
        main(args.reference_281d.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / 'failure.json').exists():
            save_json(args.out / 'failure.json', dict(error=repr(exc), time=stamp()))
        raise
