"""Frozen old-context source2 product selection, restoration and removal281."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .products281 import ARMS, ROOT, SOURCES, Engine, masks, random_writes
from .run272 import runtime, stamp

NODE = ('mid', 'gate', 'up', 'phi', 'raw_write', 'write')


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 600 * 1024**2
    old = read(reference / 'design.json')
    assert old['experiment'] == 280 and read(reference / 'audit.json')['status'] == 'passed'
    out.mkdir()
    (out / 'cases').mkdir()
    (out / 'sources').mkdir()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in hashes:
        shutil.copy2(ROOT / n, out / 'sources' / Path(n).name)
    save_json(out / 'design.json', dict(experiment=281, retrospective=True, cases=old['cases'], arms=ARMS,
        checkpoint=locate(ROOT)[1], installed_sources=installed_sources(), source_hashes=hashes,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), began=stamp(),
        reference_hashes={n: sha(reference / n) for n in ('design.json', 'status.json', 'audit.json')},
        primary='P original native5%/.99/.10nat AND local missing C1-B write25%/.95/material AND actual C1-B receiving/verdict effect25%/.95/.02nat/.005+.25|target|; >=24/32 and>=6/8strata. Matched control separation and removal account separate.',
        boundary='Old dependent contexts, firstsourcecubic and source2selected90; all later operations finite. No learned features/circuit, novel priority, efficiency or completeQB.'))
    model = load_model(True)
    save_json(out / 'runtime.json', runtime(model))
    column = np.linalg.norm(array(model.model.layers[1].mlp.down_proj.weight).astype(float), axis=0)
    entries, raw_bytes = [], 0
    for case in old['cases']:
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        key = case['key']
        values, metadata, phis = {'column_norm': column}, {}, {}
        engine = Engine(model, case)
        with engine:
            for arm in ARMS:
                packed, meta, nodes = engine.forward(arm)
                if arm in ('F', 'B', 'T'):
                    for i, node in enumerate(nodes):
                        values.update({f'{arm}_node{i + 1}_' + n: node[n] for n in NODE})
                if arm == 'F':
                    false = nodes
                if arm == 'B':
                    engine.plan, scores = masks(case, false[1], nodes[1], column)
                    values.update({'mask_' + a: v for a, v in engine.plan.items()})
                    values.update({'score_' + a: v for a, v in scores.items()})
                    base_input = {n: nodes[1][n].copy() for n in ('mid', 'gate', 'up')}
                if arm not in ('F', 'FZ', 'T'):
                    for n in ('mid', 'gate', 'up'):
                        assert np.array_equal(nodes[1][n], base_input[n]), (arm, n)
                if arm == 'P':
                    delta = packed['write'].astype(float) - values['B_write'].astype(float)
                    engine.draws = random_writes(case, delta)
                    values.update({'draw_' + a: v for a, v in engine.draws.items()})
                if arm != 'F':
                    assert meta['prefix_hashes'] == metadata['F']['prefix_hashes']
                if arm not in ('F', 'FZ', 'T'):
                    assert np.array_equal(packed['source'][0], values['B_source'][0])
                if arm == 'D':
                    for n in ('source', 'final', 'scores', 'write'):
                        assert np.array_equal(packed[n], values['P_' + n]), ('direct', n)
                    assert meta['product_hashes'][1] == metadata['P']['product_hashes'][1]
                if arm == 'FZ':
                    for n in ('source', 'final', 'scores', 'write'):
                        assert np.array_equal(packed[n], values['F_' + n]), ('zero', n)
                # Store shared phi rows once; every actual row stays bound by a reference/hash.
                phi = packed.pop('phi')
                previous = next((a for a, v in phis.items() if v.tobytes() == phi.tobytes()), None)
                if previous is None:
                    phis[arm] = phi
                    values[arm + '_phi'] = phi
                    previous = arm
                meta['phi_ref'] = previous + '_phi'
                values.update({arm + '_' + n: v for n, v in packed.items()})
                metadata[arm] = meta
        # Archive checks only AFTER all target-blind candidates and calibration controls.
        source = reference / 'cases' / (key + '.npz')
        with np.load(source) as prior:
            for arm in ('F', 'C1', 'T'):
                for n in ('source', 'final', 'scores'):
                    expected = prior[arm + '_' + n][:2] if n == 'source' else prior[arm + '_' + n]
                    assert np.array_equal(values[arm + '_' + n], expected), ('archive', arm, n)
            for i in range(2):
                for n in ('mid', 'gate', 'up', 'phi', 'write'):
                    assert np.array_equal(values[f'B_node{i + 1}_' + n], prior['C_' + n][i]), ('prepared', i, n)
        p = out / 'cases' / (key + '.npz')
        np.savez_compressed(p, **values)
        raw_bytes += p.stat().st_size
        assert raw_bytes <= 64 * 1024**2
        save_json(p.with_suffix('.json'), dict(key=key, arrays_sha256=sha(p), arms=metadata,
            native_source_input_not_used_for_masks=True, direct_product_restoration_exact=True,
            fixed_F_B_forecasts_before_full_source2_controls=True, archive_replays_exact=True,
            reference_arrays_sha256=sha(source)))
        entries.append(dict(key=key, arrays_sha256=sha(p), metadata_sha256=sha(p.with_suffix('.json')), bytes=p.stat().st_size))
        print('281', len(entries), '/32 complete', flush=True)
    save_json(out / 'status.json', dict(status='complete', cases=entries, raw_bytes=raw_bytes, ended=stamp(),
        counts=dict(full_forwards=576, model_shape='full sequence', selected_current_source2_coordinates=90,
                    forecast_full_current_source2_activation=False, other26source_and_all_receiving_activations_finite=True,
                    extra_local_maps=0, shared_preparatory_baseline_forward_per_case=1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-280b', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference_280b.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / 'failure.json').exists():
            save_json(args.out / 'failure.json', dict(error=repr(exc), time=stamp()))
        raise
