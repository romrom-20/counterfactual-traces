"""Finite source2 carry/read input factorial and direct product interventions."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .composition280 import digest, padded, polynomial
from .core225b import array
from .products281 import SOURCES as SOURCES281
from .products281 import Engine as Parent

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ('K', 'W', 'S', 'G0', 'G1', 'G2', 'FZ')
CONTROLS = ('U', 'X', 'NC', 'NW', 'C1', 'T')
ARMS = ('F', 'B', *CANDIDATES, *CONTROLS)
EVAL = ('B', 'K', 'W', 'S', 'G0', 'G1', 'G2', 'U', 'X', 'NC', 'NW', 'C1', 'T')
SOURCES = (*SOURCES281, 'research/boolean_coupling/origin282.py',
    'research/boolean_coupling/run282.py', 'research/boolean_coupling/audit282.py',
    'research/boolean_coupling/preflight282.py', 'docs/experiments/282-source2-gate-origin.md')


def split_inputs(false, base):
    x = false['mid'].astype(float)
    c = base['before'].astype(float) - false['before'].astype(float)
    a = (base['mid'].astype(float) - base['before'].astype(float)) - (x - false['before'].astype(float))
    assert np.max(abs(x + c + a - base['mid'].astype(float))) <= 1e-10
    return dict(F=false['mid'].copy(), C=(x + c).astype(np.float32),
                A=(x + a).astype(np.float32), U=base['mid'].copy()), dict(carry=c, attention=a)


def local_pair(model, case, anchor, mid):
    """One finite norm/gate/up/activation map, two full-shaped down projections."""
    layer, s, length = model.model.layers[1], case['position'], len(case['false_ids'])
    with torch.no_grad():
        x = padded(mid, length, s, model.device)
        z = layer.post_attention_layernorm(x)
        gate, up = layer.mlp.gate_proj(z), layer.mlp.up_proj(z)
        phi = layer.mlp.act_fn(gate)
        cubic = padded(polynomial(anchor, array(gate[0, s]), 3), length, s, model.device)
        pf, pc = phi * up, cubic * up
        wf, wc = layer.mlp.down_proj(pf), layer.mlp.down_proj(pc)
    return {n: array(v[0, s]) for n, v in dict(mid=x, norm=z, gate=gate, up=up,
        phi=phi, cubic=cubic, finite_product=pf, cubic_product=pc,
        finite_write=wf, cubic_write=wc).items()}


def remainder(node):
    return node['finite_product'].astype(float) - node['cubic_product'].astype(float)


def deltas(corners):
    c, a = (remainder(corners[n]) for n in ('C', 'A'))
    result = dict(K=c, W=a, S=c + a)
    if 'U' in corners:
        u = remainder(corners['U'])
        result.update(U=u, X=u - c - a, NC=u - c, NW=u - a)
    return result


def random_writes(case, correction):
    result = {}
    for i in range(3):
        seed = int(hashlib.sha256((case['key'] + f':282-write:{i}').encode()).hexdigest()[:16], 16)
        draw = np.random.default_rng(seed).normal(size=correction.shape)
        result['G' + str(i)] = (draw * np.linalg.norm(correction)) / np.linalg.norm(draw)
    return result


def pack_observations(path, values, reference):
    """Lossless aliases to already sealed identical arrays; never alter reference."""
    def signature(value):
        return str(value.dtype), tuple(value.shape), digest(value)
    old = {signature(reference[n]): n for n in reference.files}
    unique, aliases, seen, inventory = {}, {}, {}, {}
    for name, value in values.items():
        sig = signature(value)
        inventory[name] = dict(dtype=sig[0], shape=list(sig[1]), sha256=sig[2])
        if sig in old:
            assert value.tobytes() == reference[old[sig]].tobytes()
            aliases[name] = dict(reference=old[sig])
        elif sig in seen:
            assert value.tobytes() == unique[seen[sig]].tobytes()
            aliases[name] = dict(local=seen[sig])
        else:
            unique[name] = value
            seen[sig] = name
    np.savez_compressed(path, **unique)
    return aliases, inventory


class Engine(Parent):
    """Source1/2 cubic; insert defined product differences only at source MLP2."""

    def __init__(self, model, case):
        super().__init__(model, case)
        self.base_product, self.product_deltas = None, {}

    def finite(self, i):
        return self.arm in ('F', 'T') or (i == 1 and self.arm == 'C1')

    def selection(self):
        return None

    def __enter__(self):
        super().__enter__()

        def product(module, args):
            tensor, s = args[0], self.case['position']
            if self.arm in self.product_deltas:
                assert np.array_equal(array(tensor[0, s]), self.base_product)
                row = (self.base_product.astype(float) + self.product_deltas[self.arm]).astype(np.float32)
                tensor = self.replace(tensor, row)
            self.record[1]['installed_product'] = array(tensor[0, s])
            self.record[1]['product_hash'] = digest(self.record[1]['installed_product'])
            return (tensor,)

        self.handles.append(self.model.model.layers[1].mlp.down_proj.register_forward_pre_hook(product))
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self.handles = []

    def forward(self, arm):
        packed, meta, nodes = super().forward(arm)
        meta.update(changed_source_activation_sites=0 if arm in ('F', 'T') else 1 if arm == 'C1' else 2,
            current_MLP2_finite_coordinates=len(nodes[1]['gate']) if self.finite(1) else 0,
            product_delta_installed=arm in self.product_deltas)
        return packed, meta, nodes
