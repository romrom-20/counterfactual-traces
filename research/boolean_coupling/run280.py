"""Execute frozen retrospective source-composition test and one causal repair."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .composition280 import (
    BASE,
    REPAIRS,
    ROOT,
    SOURCES,
    Engine,
    local,
    polynomial,
    repair_choice,
)
from .core225b import load_model
from .run272 import runtime, stamp


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 1536 * 1024**2
    old = read(reference / 'design.json')
    assert read(reference / 'audit.json')['status'] == 'passed' and old['experiment'] == 279
    out.mkdir()
    (out / 'cases').mkdir()
    (out / 'sources').mkdir()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in hashes:
        shutil.copy2(ROOT / n, out / 'sources' / Path(n).name)
    save_json(out / 'design.json', dict(experiment=280, retrospective=True, cases=old['cases'],
        partners=old['partners'], checkpoint=locate(ROOT)[1], installed_sources=installed_sources(), source_hashes=hashes,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), began=stamp(),
        reference_hashes={n: sha(reference / n) for n in ('design.json', 'forecast-status.json', 'native-status.json', 'audit.json')},
        arms=['F', *BASE, 'T', *REPAIRS], primary='C joint original native5%/.99/.1nat + all28 source trajectory10%/.995 + emittedV10%/.995; >=24/32 and>=6/8strata.',
        boundary='Old contexts. One own-input source rule; all receiving/attention/norm/projections/product finite. No efficient whole-model forecast, learned circuit or complete Question B.'))
    model = load_model(True)
    save_json(out / 'runtime.json', runtime(model))
    entries, raw_bytes = [], 0
    for case in old['cases']:
        assert shutil.disk_usage(out).free >= 768 * 1024**2
        values, metadata = {}, {}
        key, start, length = case['key'], case['position'], len(case['false_ids'])
        engine = Engine(model, case)
        with engine:
            for arm in ('F', *BASE, 'T'):
                packed, meta, _ = engine.forward(arm)
                values.update({arm + '_' + n: v for n, v in packed.items()})
                metadata[arm] = meta
                if arm != 'F':
                    for n in ('prefix_hashes', 'K_prefix', 'V_prefix'):
                        assert meta[n] == metadata['F'][n]
                if arm == 'FZ':
                    for n, v in packed.items():
                        assert np.array_equal(v, values['F_' + n]), ('zero', n)
            with np.load(reference / 'forecasts' / (key + '.npz')) as previous, np.load(reference / 'observations' / (key + '.npz')) as native:
                for arm, archive, label in (('F', previous, 'F'), ('T', native, 'T'), ('C1', previous, 'C')):
                    for n in ('source', 'final', 'scores'):
                        actual = values[arm + '_' + n][0] if n == 'source' else values[arm + '_' + n]
                        assert np.array_equal(actual, archive[label + '_' + n]), (arm, n)
        # All base predictions have been made before any local finite diagnostics.
        # Old contexts: this ordering is not presented as a fresh prospective seal.
        diagnostic = {p + '_' + n: [] for p in ('C_finite', 'T_cubic') for n in ('phi', 'write')}
        missing = []
        with torch.no_grad():
            for i, layer in enumerate(model.model.layers):
                cf = local(layer, values['C_gate'][i], values['C_up'][i], length, start)
                anchor = {'gate': values['F_gate'][i], 'phi': values['F_phi'][i]}
                tc = local(layer, values['T_gate'][i], values['T_up'][i], length, start,
                           polynomial(anchor, values['T_gate'][i], 3))
                for label, node in (('C_finite', cf), ('T_cubic', tc)):
                    for n in ('phi', 'write'):
                        diagnostic[label + '_' + n].append(node[n])
                missing.append(cf['write'].astype(float) - values['C_write'][i].astype(float))
        values.update({n: np.stack(v) for n, v in diagnostic.items()})
        selected, draws = repair_choice(case, {'write': values['C_write']}, np.stack(missing))
        engine.selected, engine.draws = selected, draws
        values.update(selected=np.array(selected), **{'draw_' + n: v for n, v in draws.items()})
        with engine:
            for arm in REPAIRS:
                packed, meta, _ = engine.forward(arm)
                for n in ('prefix_hashes', 'K_prefix', 'V_prefix'):
                    assert meta[n] == metadata['F'][n]
                for n in ('before', 'mid', 'gate', 'up'):
                    assert np.array_equal(packed[n][selected], values['C_' + n][selected]), (arm, n)
                assert np.array_equal(packed['source'][:selected], values['C_source'][:selected])
                if arm == 'R':
                    assert np.array_equal(packed['phi'][selected], values['C_finite_phi'][selected])
                    assert np.array_equal(packed['write'][selected], values['C_finite_write'][selected])
                else:
                    expected = (values['C_write'][selected].astype(float) + draws[arm]).astype(np.float32)
                    assert np.array_equal(packed['write'][selected], expected)
                values.update({arm + '_' + n: v for n, v in packed.items()})
                metadata[arm] = meta
        x = (values['C_mid'] + values['C_finite_write']).astype(np.float32)
        error = values['T_source'].astype(float) - values['C_source'].astype(float)
        inherited = values['T_source'].astype(float) - x.astype(float)
        new = x.astype(float) - values['C_source'].astype(float)
        assert np.max(abs(error - inherited - new)) <= 1e-10
        values.update(finite_C_source=x, inherited_error=inherited, new_error=new)
        for arm in ('F', *BASE, 'T', *REPAIRS):
            assert np.array_equal((values[arm + '_mid'] + values[arm + '_write']).astype(np.float32), values[arm + '_source'])
            assert np.isfinite(values[arm + '_final']).all()
        p = out / 'cases' / (key + '.npz')
        np.savez_compressed(p, **values)
        raw_bytes += p.stat().st_size
        assert raw_bytes <= 1200 * 1024**2
        save_json(p.with_suffix('.json'), dict(key=key, arrays_sha256=sha(p), arms=metadata,
            selected_layer=selected + 1, archive_first_source_final_score_replays_exact=True,
            only_source_intervened=True, ordinary_true_is_retrospective=True,
            reference_forecast_sha256=sha(reference / 'forecasts' / (key + '.npz')),
            reference_native_sha256=sha(reference / 'observations' / (key + '.npz'))))
        entries.append(dict(key=key, arrays_sha256=sha(p), metadata_sha256=sha(p.with_suffix('.json')), bytes=p.stat().st_size))
        print('280', len(entries), '/32 complete', 'selected-layer', selected + 1, flush=True)
    save_json(out / 'status.json', dict(status='complete', ended=stamp(), cases=entries, raw_bytes=raw_bytes,
        counts=dict(full_forwards=352, extra_local_diagnostic_product_down_maps=1792, extra_local_current_finite_source_SiLU_maps=896,
                    C_current_finite_source_sites_per_forward=0, C1_current_finite_source_sites=27,
                    R_current_finite_source_sites=1, R_selection_local_finite_maps_per_case=28)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-279', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference_279.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / 'failure.json').exists():
            save_json(args.out / 'failure.json', dict(error=repr(exc), time=stamp()))
        raise
