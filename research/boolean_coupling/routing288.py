"""Finite recentered source/receiver updates and their genuine derivatives.

The source is the first suffix row. Causality makes its local update independent
of later rows. 0=ordinary False anchor; 1=source-only edited anchor (MF in 287).
An arm names source-block, receiving-block, and final-head anchors, in that order.
These are constructed finite networks, not straight-through backward overrides.
"""

import itertools

import numpy as np
import torch

from .cache272 import prefix
from .core225b import array
from .qualify287b import SOURCES as PREVIOUS
from .readout286 import Suffix, margin, release

ARMS = tuple("".join(bits) for bits in itertools.product("01", repeat=3))
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/routing288.py",
    "research/boolean_coupling/run288.py",
    "research/boolean_coupling/audit288.py",
    "research/boolean_coupling/preflight288.py",
    "docs/experiments/288-source-receiver-tangent-routing.md",
)
BOUNDS = dict(
    boundary_adjoint=1e-4,
    matched_old_cached_gradient=1e-4,
    matched_old_full_gradient=1e-4,
    old_cached_primal=1e-4,
    native_finite_coordinate=0.002,
    native_finite_score=0.001,
    CPU64_head=0.005,
    projected_AD_FD_absolute=0.001,
    projected_AD_FD_relative=0.02,
    FD_halving_absolute=0.002,
    FD_halving_relative=0.03,
)
PRIMARY = dict(
    target="g111-g000 and finite E111-E000",
    hypotheses={"SOURCE": "100", "RECEIVING_AND_HEAD": "011"},
    vector_norm_min=0.001,
    relative_max=0.25,
    cosine_min=0.95,
    finite_error_absolute=0.005,
    finite_error_relative=0.25,
    finite_material_min=0.02,
    population_min=24,
    per_layout_min=6,
)


class Routing(Suffix):
    def __init__(self, model, case, write):
        super().__init__(model, case)
        forecast = self.first(write)
        edited = torch.cat((forecast[:, :1], self.false_tail[:, 1:]), dim=1).detach()
        self.counts = dict(
            full_F=1, first_block=1, anchor_suffixes=0, anchor_blocks=0,
            modified_forwards=0, modified_blocks=0, modified_heads=0,
            tape_backwards=0, triangular_blocks=0, triangular_block_VJPs=0,
            triangular_head_VJPs=0, native_suffixes=0, native_blocks=0,
        )
        self.tapes, self.scores = [], []
        for base in (self.false_tail, edited):
            tape, scores = self.native(base)
            self.tapes.append(tape)
            self.scores.append(scores)
            self.counts["anchor_suffixes"] += 1
            self.counts["anchor_blocks"] += len(tape) - 1

    def fresh(self):
        return prefix(self.cache, self.start, self.model.config)

    def block(self, index, value, cache):
        return self.model.model.layers[index](
            value,
            attention_mask=self.mask,
            position_ids=self.positions[None],
            past_key_values=cache,
            use_cache=True,
            cache_position=self.positions,
            position_embeddings=self.rope,
        )

    def head(self, value):
        return self.model.lm_head(self.model.model.norm(value)[:, -1:])[0, -1, [895, 830]]

    def native(self, base):
        fresh, tape = self.fresh(), [base.detach().clone()]
        with torch.no_grad():
            for index in range(1, len(self.model.model.layers)):
                tape.append(self.block(index, tape[-1], fresh).detach().clone())
            scores = self.head(tape[-1])
        return tape, array(scores)

    def native_bump(self, anchor, displacement):
        base = self.tapes[anchor][0].clone()
        base[:, :1] += torch.tensor(
            displacement[None, None], device=self.model.device, dtype=self.model.dtype
        )
        tape, scores = self.native(base)
        self.counts["native_suffixes"] += 1
        self.counts["native_blocks"] += len(tape) - 1
        return array(tape[-1][0]), scores

    def _modified(self, arm, point, trace=False):
        a, r, h = map(int, arm)
        ca, cr = self.fresh(), self.fresh() if a != r else None
        value = torch.cat((point[None, None], torch.zeros_like(self.false_tail[:, 1:])), dim=1)
        if trace:
            self.trace = {}
        for j in range(len(self.tapes[0]) - 1):
            if trace:

                def record(g, j=j):
                    self.trace[j] = array(g[0, 0])

                value.register_hook(record)
            sa = self.block(j + 1, self.tapes[a][j] + value, ca)
            rr = sa if a == r else self.block(j + 1, self.tapes[r][j] + value, cr)
            value = torch.cat(
                (sa[:, :1] - self.tapes[a][j + 1][:, :1],
                 rr[:, 1:] - self.tapes[r][j + 1][:, 1:]), dim=1
            )
            self.counts["modified_blocks"] += 1 if a == r else 2
        head_state = self.tapes[h][-1] + value
        scores = self.head(head_state)
        self.counts["modified_forwards"] += 1
        self.counts["modified_heads"] += 1
        return value, head_state, scores

    def finite(self, arm, displacement):
        point = torch.tensor(displacement, device=self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            offset, head_state, raw = self._modified(arm, point)
        h = int(arm[-1])
        # Float64 lifting avoids adding another Float32 rounding at the score node.
        lifted = self.scores[1].astype(float) + array(raw).astype(float) - self.scores[h]
        return dict(offset=array(offset[0]), head_state=array(head_state[0]),
                    raw_scores=array(raw), lifted_scores=lifted)

    def gradient(self, arm):
        with torch.enable_grad():
            point = torch.zeros_like(self.false_tail[0, 0]).requires_grad_(True)
            offset, _, scores = self._modified(arm, point, trace=True)
            gradient = torch.autograd.grad(scores[1] - scores[0], point)[0]
        self.counts["tape_backwards"] += 1
        return dict(gradient=array(gradient), zero_offset=array(offset[0]),
                    zero_raw_scores=array(scores),
                    source_trace=np.stack([self.trace[j] for j in sorted(self.trace)]))

    def triangular(self, arm):
        """Separate block-VJP implementation of the triangular derivative algebra."""
        a, r, h = map(int, arm)
        ca, cr = self.fresh(), self.fresh() if a != r else None
        with torch.enable_grad():
            point = self.tapes[h][-1].clone().requires_grad_(True)
            scores = self.head(point)
            adjoint = torch.autograd.grad(scores[1] - scores[0], point)[0].detach()
        self.counts["triangular_head_VJPs"] += 1
        traces, noncausal = [], []
        for j in reversed(range(len(self.tapes[0]) - 1)):
            with torch.enable_grad():
                point = self.tapes[a][j].clone().requires_grad_(True)
                output = self.block(j + 1, point, ca)
                if a == r:
                    adjoint = torch.autograd.grad(output, point, grad_outputs=adjoint)[0].detach()
                    self.counts["triangular_blocks"] += 1
                    self.counts["triangular_block_VJPs"] += 1
                else:
                    source_cotangent = torch.zeros_like(adjoint)
                    source_cotangent[:, :1] = adjoint[:, :1]
                    left = torch.autograd.grad(output, point, grad_outputs=source_cotangent)[0]
                    noncausal.append(float(left[:, 1:].abs().max().detach().cpu()))
                    other = self.tapes[r][j].clone().requires_grad_(True)
                    output_r = self.block(j + 1, other, cr)
                    receiver_cotangent = adjoint.clone()
                    receiver_cotangent[:, :1] = 0
                    right = torch.autograd.grad(output_r, other, grad_outputs=receiver_cotangent)[0]
                    adjoint = (left + right).detach()
                    self.counts["triangular_blocks"] += 2
                    self.counts["triangular_block_VJPs"] += 2
            traces.append(array(adjoint[0, 0]))
            del point, output
        return dict(gradient=array(adjoint[0, 0]), source_trace=np.stack(traces[::-1]),
                    noncausal=np.asarray(noncausal))


def finite_differences(engine, arm, gradient, direction, name):
    ad, rows, estimates = float(np.asarray(gradient, float) @ direction), [], []
    for step in (0.05, 0.025):
        plus = engine.finite(arm, step * direction)
        release(engine.model)
        minus = engine.finite(arm, -step * direction)
        release(engine.model)
        fd = (margin(plus["lifted_scores"]) - margin(minus["lifted_scores"])) / (2 * step)
        rows.append(dict(arm=arm, direction=name, step=step, AD=ad, FD=fd,
                         plus_scores=plus["lifted_scores"].tolist(),
                         minus_scores=minus["lifted_scores"].tolist(),
                         error=abs(ad - fd), bound=0.001 + 0.02 * abs(ad)))
        estimates.append(fd)
    rows.append(dict(arm=arm, direction=name, step="halving", AD=ad, FD=estimates[-1],
                     error=abs(estimates[0] - estimates[1]), bound=0.002 + 0.03 * abs(ad)))
    return rows
