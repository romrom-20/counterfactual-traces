"""Source-only activation expansions, driven by their own changing inputs."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .core225b import array, observe
from .source279 import SOURCES as SOURCES279
from .source279 import derivatives

SOURCES = (*SOURCES279, 'research/boolean_coupling/composition280.py',
           'research/boolean_coupling/run280.py', 'research/boolean_coupling/audit280.py',
           'research/boolean_coupling/preflight280.py', 'docs/experiments/280-source-composition.md')
ROOT = Path(__file__).resolve().parents[2]
BASE = ('C', 'Q', 'L', 'C1', 'FZ')
REPAIRS = ('R', 'G0', 'G1', 'G2')
EVAL = ('C', 'Q', 'L', 'C1', 'T', *REPAIRS)
FIELDS = ('before', 'mid', 'gate', 'up', 'phi', 'raw_write', 'write', 'K', 'V', 'q')


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def polynomial(anchor, gate, degree):
    dg = gate.astype(float) - anchor['gate'].astype(float)
    d = derivatives(anchor['gate'])
    value = anchor['phi'].astype(float) + d['d1'] * dg
    if degree >= 2:
        value += .5 * d['d2'] * dg**2
    if degree >= 3:
        value += d['d3'] * dg**3 / 6
    return value.astype(np.float32)


def padded(row, length, start, device):
    value = torch.zeros((1, length, len(row)), dtype=torch.float32, device=device)
    value[0, start] = torch.tensor(row, dtype=torch.float32, device=device)
    return value


def local(layer, gate, up, length, start, phi=None):
    device = layer.mlp.down_proj.weight.device
    g = padded(gate, length, start, device)
    ph = layer.mlp.act_fn(g) if phi is None else padded(phi, length, start, device)
    product = ph * padded(up, length, start, device)
    write = layer.mlp.down_proj(product)
    return dict(phi=array(ph[0, start]), write=array(write[0, start]),
                product_hash=digest(array(product[0, start])))


class Engine:
    """Only source SiLU/write changes. No native True trace is a forecast input."""

    def __init__(self, model, case):
        self.model, self.case = model, case
        self.original = qwen.eager_attention_forward
        self.handles, self.anchors = [], None
        self.arm, self.record = None, None
        self.selected, self.draws = None, None
        self.current_source_finite = 0

    def replace(self, value, row):
        start = self.case['position']
        changed = value.clone()
        changed[0, start] = torch.tensor(row, dtype=torch.float32, device=value.device)
        assert torch.equal(changed[:, :start], value[:, :start])
        assert torch.equal(changed[:, start + 1:], value[:, start + 1:])
        return changed

    def degree(self, index):
        if self.arm in ('F', 'T') or (self.arm == 'C1' and index != 0) or (self.arm == 'R' and index == self.selected):
            return 0
        return 1 if self.arm == 'L' else 2 if self.arm == 'Q' else 3

    def __enter__(self):
        start = self.case['position']
        qwen.eager_attention_forward = self.attention
        for i, layer in enumerate(self.model.model.layers):
            def before(module, args, i=i):
                self.record[i]['before'] = array(args[0][0, start])

            def mid(module, args, i=i):
                self.record[i]['mid'] = array(args[0][0, start])

            def gate(module, args, output, i=i):
                self.record[i]['gate'] = array(output[0, start])

            def up(module, args, output, i=i):
                self.record[i]['up'] = array(output[0, start])

            def phi_before(module, args, i=i):
                degree = self.degree(i)
                if degree:
                    rec, anchor = self.record[i], self.anchors[i]
                    rec['proposal'] = polynomial(anchor, rec['gate'], degree)
                    return (self.replace(args[0], anchor['gate']),)
                self.current_source_finite += 1

            def phi_after(module, args, output, i=i):
                rec = self.record[i]
                if self.degree(i):
                    assert np.array_equal(array(output[0, start]), self.anchors[i]['phi'])
                    rec['False_primal_exact'] = True
                    output = self.replace(output, rec.pop('proposal'))
                rec['phi'] = array(output[0, start])
                return output

            def product(module, args, i=i):
                self.record[i]['product_hash'] = digest(array(args[0][0, start]))

            def write(module, args, output, i=i):
                self.record[i]['raw_write'] = array(output[0, start])
                if self.arm.startswith('G') and i == self.selected:
                    row = (array(output[0, start]).astype(float) + self.draws[self.arm]).astype(np.float32)
                    output = self.replace(output, row)
                self.record[i]['write'] = array(output[0, start])
                return output

            self.handles += [layer.register_forward_pre_hook(before),
                layer.post_attention_layernorm.register_forward_pre_hook(mid),
                layer.mlp.gate_proj.register_forward_hook(gate), layer.mlp.up_proj.register_forward_hook(up),
                layer.mlp.act_fn.register_forward_pre_hook(phi_before), layer.mlp.act_fn.register_forward_hook(phi_after),
                layer.mlp.down_proj.register_forward_pre_hook(product), layer.mlp.register_forward_hook(write)]
        return self

    def __exit__(self, *_):
        qwen.eager_attention_forward = self.original
        for handle in self.handles:
            handle.remove()

    def attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        assert dropout == 0 and not module.training
        output, p = self.original(module, query, key, value, mask, scaling=scaling, dropout=dropout, **kwargs)
        i, start = module.layer_idx, self.case['position']
        self.record[i].update(q=array(query[0, :, start]), K=array(key[0, :, start]), V=array(value[0, :, start]),
            K_prefix=digest(array(key[0, :, :start])), V_prefix=digest(array(value[0, :, :start])))
        return output, p

    def forward(self, arm):
        self.arm, self.record = arm, [dict() for _ in self.model.model.layers]
        self.current_source_finite = 0
        ids = self.case['false_ids' if arm in ('F', 'FZ') else 'true_ids']
        states, context, scores = observe(self.model, torch.tensor([ids], device=self.model.device))
        source = np.stack([array(s[0, self.case['position']]) for s in states])
        packed = {n: np.stack([r[n] for r in self.record]) for n in FIELDS}
        packed.update(source=source, final=array(states[-1][0, self.case['position']:]), scores=array(scores))
        metadata = dict(prefix_hashes=[digest(array(s[0, :self.case['position']])) for s in states],
                        K_prefix=[r['K_prefix'] for r in self.record], V_prefix=[r['V_prefix'] for r in self.record],
                        product_hashes=[r['product_hash'] for r in self.record],
                        false_primal_checks=sum(r.get('False_primal_exact', False) for r in self.record),
                        current_source_finite_sites=self.current_source_finite)
        if arm == 'F':
            self.anchors = self.record
        return packed, metadata, context


def repair_choice(case, base, missing):
    norms = np.linalg.norm(missing.astype(float), axis=-1)
    selected = int(np.argmax(norms))
    draws = {}
    for number in range(3):
        seed = int(hashlib.sha256((case['key'] + f':280-write:{number}').encode()).hexdigest()[:16], 16)
        draw = np.random.default_rng(seed).normal(size=base['write'][selected].shape)
        draws['G' + str(number)] = draw * norms[selected] / np.linalg.norm(draw)
    return selected, draws
