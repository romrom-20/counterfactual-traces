"""Target-blind source-MLP2 product selection and direct product interventions."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .composition280 import SOURCES as SOURCES280
from .composition280 import digest, polynomial
from .core225b import array, observe

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ('P', 'D', 'H', 'R0', 'R1', 'R2', 'G0', 'G1', 'G2', 'FZ')
CONTROLS = ('C1', 'A', 'A0', 'A1', 'A2', 'T')
ARMS = ('F', 'B', *FORECAST, *CONTROLS)
EVAL = ('B', 'P', 'H', 'R0', 'R1', 'R2', 'G0', 'G1', 'G2', 'C1', 'A', 'A0', 'A1', 'A2', 'T')
SOURCES = (*SOURCES280, 'research/boolean_coupling/products281.py',
    'research/boolean_coupling/run281.py', 'research/boolean_coupling/audit281.py',
    'research/boolean_coupling/preflight281.py', 'docs/experiments/281-source2-products.md')


def masks(case, false, current, column):
    gate, up, ph = (current[n].astype(float) for n in ('gate', 'up', 'phi'))
    gf, pf = false['gate'].astype(float), false['phi'].astype(float)
    proxy = pf + np.maximum(gate, 0.) - np.maximum(gf, 0.) - ph
    scores = {'P': abs(proxy) * abs(up) * column,
              'H': abs(gate - gf)**4 * abs(up) * column}
    count = int(np.ceil(.01 * len(gate)))
    selected = {a: np.sort(np.lexsort((np.arange(len(gate)), -s))[:count]) for a, s in scores.items()}
    for i in range(3):
        seed = int(hashlib.sha256((case['key'] + f':281-mask:{i}').encode()).hexdigest()[:16], 16)
        selected['R' + str(i)] = np.sort(np.random.default_rng(seed).choice(len(gate), count, replace=False))
    return selected, scores


def random_writes(case, correction):
    result = {}
    for i in range(3):
        seed = int(hashlib.sha256((case['key'] + f':281-write:{i}').encode()).hexdigest()[:16], 16)
        draw = np.random.default_rng(seed).normal(size=correction.shape)
        result['G' + str(i)] = draw * np.linalg.norm(correction) / np.linalg.norm(draw)
    return result


class Engine:
    """MLP1 cubic; MLP2 selected activations/products; every later operation finite."""

    def __init__(self, model, case):
        self.model, self.case = model, case
        self.anchors, self.plan, self.draws = None, None, None
        self.arm, self.record, self.handles = None, None, []
        self.raw_selected = None

    def replace(self, tensor, row):
        s = self.case['position']
        changed = tensor.clone()
        changed[0, s] = torch.tensor(row, dtype=torch.float32, device=tensor.device)
        assert torch.equal(changed[:, :s], tensor[:, :s])
        assert torch.equal(changed[:, s + 1:], tensor[:, s + 1:])
        return changed

    def selection(self):
        if self.plan is None:
            return None
        if self.arm == 'D' or self.arm == 'A':
            return self.plan['P']
        if self.arm.startswith('A'):
            return self.plan['R' + self.arm[-1]]
        if self.arm == 'FZ':
            return np.arange(len(self.plan['P']))
        return self.plan.get(self.arm)

    def finite(self, i):
        return self.arm in ('F', 'T') or (i == 1 and self.arm in CONTROLS)

    def __enter__(self):
        s = self.case['position']
        for i, layer in enumerate(self.model.model.layers[:2]):
            def mid(module, args, i=i):
                self.record[i]['mid'] = array(args[0][0, s])

            def gate(module, args, output, i=i):
                self.record[i]['gate'] = array(output[0, s])

            def up(module, args, output, i=i):
                self.record[i]['up'] = array(output[0, s])
                self.record[i]['up_tensor'] = output[0, s].detach().clone()

            def phi_before(module, args, i=i):
                if self.finite(i):
                    return None
                rec, anchor = self.record[i], self.anchors[i]
                rec['cubic'] = polynomial(anchor, rec['gate'], 3)
                row = anchor['gate'].copy()
                selected = self.selection() if i == 1 else None
                if selected is not None:
                    row[selected] = rec['gate'][selected]
                rec['actual_gate'] = row
                return (self.replace(args[0], row),)

            def phi_after(module, args, output, i=i):
                rec = self.record[i]
                if not self.finite(i):
                    rec['primal'] = array(output[0, s])
                    selected = self.selection() if i == 1 else None
                    unselected = np.ones(len(rec['gate']), dtype=bool)
                    if selected is not None:
                        unselected[selected] = False
                    assert np.array_equal(rec['primal'][unselected], self.anchors[i]['phi'][unselected])
                    ph = rec['cubic'].copy()
                    if selected is not None:
                        self.raw_selected = output[0, s].detach().clone()
                        if self.arm != 'D':
                            ph[selected] = rec['primal'][selected]
                    output = self.replace(output, ph)
                rec['phi'] = array(output[0, s])
                return output

            def product(module, args, i=i):
                value = args[0]
                if i == 1 and (self.arm == 'D' or self.arm.startswith('A')):
                    indices = self.selection()
                    row = value[0, s].detach().clone()
                    if self.arm == 'D':
                        row[indices] = self.raw_selected[indices] * self.record[i]['up_tensor'][indices]
                    else:
                        cubic = polynomial(self.anchors[i], self.record[i]['gate'], 3)
                        c = torch.tensor(cubic, device=value.device)
                        row[indices] = c[indices] * self.record[i]['up_tensor'][indices]
                    value = self.replace(value, array(row))
                self.record[i]['product_hash'] = digest(array(value[0, s]))
                return (value,)

            def write(module, args, output, i=i):
                self.record[i]['raw_write'] = array(output[0, s])
                if i == 1 and self.arm.startswith('G'):
                    row = (array(output[0, s]).astype(float) + self.draws[self.arm]).astype(np.float32)
                    output = self.replace(output, row)
                self.record[i]['write'] = array(output[0, s])
                return output

            self.handles += [layer.post_attention_layernorm.register_forward_pre_hook(mid),
                layer.mlp.gate_proj.register_forward_hook(gate), layer.mlp.up_proj.register_forward_hook(up),
                layer.mlp.act_fn.register_forward_pre_hook(phi_before), layer.mlp.act_fn.register_forward_hook(phi_after),
                layer.mlp.down_proj.register_forward_pre_hook(product), layer.mlp.register_forward_hook(write)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def forward(self, arm):
        self.arm, self.record = arm, [dict(), dict()]
        ids = self.case['false_ids' if arm in ('F', 'FZ') else 'true_ids']
        states, _, scores = observe(self.model, torch.tensor([ids], device=self.model.device))
        metadata = dict(prefix_hashes=[digest(array(st[0, :self.case['position']])) for st in states],
            product_hashes=[r['product_hash'] for r in self.record],
            actual_gate_hashes=[digest(r.get('actual_gate', r['gate'])) for r in self.record],
            primal_hashes=[digest(r.get('primal', r['phi'])) for r in self.record],
            changed_source_activation_sites=0 if arm in ('F', 'T') else 1 if arm in CONTROLS else 2,
            current_MLP2_finite_coordinates=len(self.record[1]['gate']) if arm in ('F', *CONTROLS) else len(self.selection()) if self.selection() is not None else 0,
            other_rows_unchanged=True)
        packed = dict(source=np.stack([array(st[0, self.case['position']]) for st in states[:2]]),
            final=array(states[-1][0, self.case['position']:]), scores=array(scores),
            write=self.record[1]['write'], raw_write=self.record[1]['raw_write'], phi=self.record[1]['phi'])
        if arm == 'F':
            self.anchors = [{k: r[k].copy() for k in ('gate', 'phi')} for r in self.record]
        return packed, metadata, self.record
