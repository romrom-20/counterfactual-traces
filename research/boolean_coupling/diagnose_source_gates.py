"""Posthoc source-MLP2 gate-error ledger; zero model calls, no feature attribution."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read, table
from .checkpoint238 import locate

ROOT = Path(__file__).resolve().parents[2]


def main(run, out):
    assert not out.exists() and read(run / 'audit.json')['status'] == 'passed'
    out.mkdir()
    folder, checkpoint = locate(ROOT)
    with safe_open(folder / 'model.safetensors', framework='pt', device='cpu') as handle:
        down = handle.get_tensor('model.layers.1.mlp.down_proj.weight').float().numpy().astype(float)
    design, status = read(run / 'design.json'), read(run / 'status.json')
    cases = {c['key']: c for c in design['cases']}
    case_rows, groups = [], []
    for entry in status['cases']:
        key = entry['key']
        p = run / 'cases' / (key + '.npz')
        assert sha(p) == entry['arrays_sha256']
        with np.load(p) as d:
            gf, gt = d['F_gate'][1].astype(float), d['T_gate'][1].astype(float)
            up = d['T_up'][1].astype(float)
            product = (d['T_phi'][1].astype(float) - d['T_cubic_phi'][1].astype(float)) * up
            target = d['T_write'][1].astype(float) - d['T_cubic_write'][1].astype(float)
            vectors = down * product[None, :]
            total = vectors.sum(axis=1)
            rounding = target - total
            norm2 = float(target @ target)
            projections = (vectors.T @ target) / norm2
            energy = np.sum(vectors * vectors, axis=0)
            order = np.argsort(-energy, kind='stable')
            top90 = order[:90]
            reconstructed = vectors[:, top90].sum(axis=1)
            crossing = (gf >= 0) != (gt >= 0)
            verror = np.linalg.norm(reconstructed - target) / np.linalg.norm(target)
            case_rows.append(dict(key=key, **cases[key]['group'],
                gate_change_norm=float(np.linalg.norm(gt - gf)), gate_change_max=float(np.max(abs(gt - gf))),
                sign_crossing_count=int(crossing.sum()),
                crossing_energy_fraction=float(energy[crossing].sum() / energy.sum()),
                top90_energy_fraction=float(energy[top90].sum() / energy.sum()), top90_target_relative=float(verror),
                top90_signed_projection=float(projections[top90].sum()),
                rounding_relative=float(np.linalg.norm(rounding) / np.linalg.norm(target)),
                signed_closure_error=float(abs(projections.sum() + rounding @ target / norm2 - 1))))
            for sf, st in ((False, False), (False, True), (True, False), (True, True)):
                mask = ((gf >= 0) == sf) & ((gt >= 0) == st)
                vector = vectors[:, mask].sum(axis=1)
                groups.append(dict(key=key, **cases[key]['group'], false_nonnegative=sf, true_nonnegative=st,
                    neurons=int(mask.sum()), individual_energy_fraction=float(energy[mask].sum() / energy.sum()),
                    signed_projection=float(projections[mask].sum()), vector_norm=float(np.linalg.norm(vector))))
    table(out / 'source2-cases.csv', case_rows)
    table(out / 'source2-regimes.csv', groups)
    fields = ['gate_change_norm', 'gate_change_max', 'sign_crossing_count', 'crossing_energy_fraction', 'top90_energy_fraction', 'top90_target_relative', 'top90_signed_projection', 'rounding_relative']
    summary = {f: float(np.median([r[f] for r in case_rows])) for f in fields}
    (out / 'source2-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'source2-provenance.json').write_text(json.dumps(dict(posthoc=True, model_calls=0, native_target_used=True,
        source_sha256=sha(__file__), audit_sha256=sha(run / 'audit.json'), checkpoint=checkpoint,
        source_arrays={e['key']: e['arrays_sha256'] for e in status['cases']},
        bound='Exact Float64 accounting with explicit rounding residual; target-aware concentration is not a prospective mask, causal neuron test or learned feature.'), indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    main(args.run.resolve(), args.out.resolve())
