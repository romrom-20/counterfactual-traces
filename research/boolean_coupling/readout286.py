"""Source-residual sensitivities through a finite, False-prefix cached suffix.

The variable is the edited token's residual AFTER block1, not a learned feature.
Every affected later operation and the full vocabulary head is evaluated. Prefix
keys/values strictly before the source come from one ordinary False observation.
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .cache272 import SOURCES as CACHE_SOURCES
from .cache272 import cache_seals, prefix
from .composition280 import digest
from .core225b import array
from .structure285 import SOURCES as PREVIOUS

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *dict.fromkeys((*CACHE_SOURCES, *PREVIOUS)),
    "research/boolean_coupling/readout286.py",
    "research/boolean_coupling/run286.py",
    "research/boolean_coupling/audit286.py",
    "research/boolean_coupling/preflight286.py",
    "docs/experiments/286-false-readout.md",
)


def release(model):
    if model.device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
        torch.mps.synchronize()


@contextmanager
def source_write(model, position, gate, write, expected_phi=None):
    """Prevent current source SiLU before installing a known forecast write."""
    layer, handles = model.model.layers[0], []
    record = {"activation_calls": 0, "False_primal_exact": False}

    def before(module, args):
        value = args[0].clone()
        value[0, position] = torch.tensor(gate, dtype=model.dtype, device=model.device)
        return (value,)

    def observe_input(module, args):
        actual = array(args[0][0, position])
        assert np.array_equal(actual, gate)
        record["activation_calls"] += 1
        record["input_gate_sha256"] = digest(actual)

    def observe_phi(module, args, output):
        actual = array(output[0, position])
        record["output_phi_sha256"] = digest(actual)
        if expected_phi is not None:
            assert np.array_equal(actual, expected_phi)
            record["False_primal_exact"] = True

    def after(module, args, output):
        value = output.clone()
        value[0, position] = torch.tensor(write, dtype=model.dtype, device=model.device)
        return value

    handles += [
        layer.mlp.act_fn.register_forward_pre_hook(before),
        layer.mlp.act_fn.register_forward_pre_hook(observe_input),
        layer.mlp.act_fn.register_forward_hook(observe_phi),
        layer.mlp.register_forward_hook(after),
    ]
    try:
        yield record
    finally:
        for handle in handles:
            handle.remove()


class Suffix:
    """One False observation; fresh cropped cache for each later-suffix call."""

    def __init__(self, model, case):
        self.model, self.case = model, case
        self.start, self.total = case["position"], len(case["false_ids"])
        assert 0 < self.start < self.total - 1
        assert not model.training
        self.calls, self.backwards, self.first_blocks = 0, 0, 0
        captured, nodes, handles = {}, {}, []
        first, last = model.model.layers[0], model.model.layers[-1]

        def context(module, args, kwargs):
            captured["mask"] = kwargs["attention_mask"].detach().clone()
            captured["rope"] = tuple(v.detach().clone() for v in kwargs["position_embeddings"])

        def state(name):
            def hook(module, args, output):
                captured[name] = output.detach().clone()

            return hook

        def node(name):
            def hook(module, args, output):
                nodes[name] = array(output[0, self.start])

            return hook

        handles += [
            first.register_forward_pre_hook(context, with_kwargs=True),
            first.register_forward_hook(state("first")),
            last.register_forward_hook(state("last")),
        ]
        handles += [
            m.register_forward_hook(node(n))
            for m, n in (
                (first.mlp.gate_proj, "gate"),
                (first.mlp.act_fn, "phi"),
                (first.mlp, "write"),
            )
        ]
        ids = torch.tensor([case["false_ids"]], device=model.device)
        try:
            with torch.no_grad():
                result = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=True,
                    logits_to_keep=1,
                    position_ids=torch.arange(self.total, device=model.device)[None],
                    cache_position=torch.arange(self.total, device=model.device),
                )
        finally:
            for handle in handles:
                handle.remove()
        self.cache, self.context, self.false_nodes = result.past_key_values, captured, nodes
        self.false_tail = captured["first"][:, self.start :].detach().clone()
        self.false_final = array(captured["last"][0, self.start :])
        self.false_scores = array(result.logits[0, -1, [895, 830]])
        self.seals = cache_seals(self.cache, self.start)
        self.positions = torch.arange(self.start, self.total, device=model.device)
        self.mask = captured["mask"][:, :, self.start :, :]
        self.rope = tuple(v[:, self.start :] for v in captured["rope"])

    def first(self, write):
        ids = torch.tensor([self.case["true_ids"]], device=self.model.device)
        with (
            torch.no_grad(),
            source_write(
                self.model,
                self.start,
                self.false_nodes["gate"],
                write,
                expected_phi=self.false_nodes["phi"],
            ) as policy,
        ):
            value = self.model.model.layers[0](
                self.model.model.embed_tokens(ids),
                attention_mask=self.context["mask"],
                position_ids=torch.arange(self.total, device=self.model.device)[None],
                cache_position=torch.arange(self.total, device=self.model.device),
                position_embeddings=self.context["rope"],
                use_cache=False,
            )
        assert policy["activation_calls"] == 1 and policy["False_primal_exact"]
        self.first_policy = policy
        self.first_blocks += 1
        return value[:, self.start :].detach().clone()

    def _forward(self, base, source, trace=False):
        assert base.shape == (1, self.total - self.start, self.model.config.hidden_size)
        fresh = prefix(self.cache, self.start, self.model.config)
        value = torch.cat((source[None, None, :], base[:, 1:]), dim=1)
        if trace:
            self._trace_source, self._trace_norms = {}, {}
        for index, layer in enumerate(self.model.model.layers[1:]):
            if trace:

                def record(gradient, index=index):
                    self._trace_source[index] = array(gradient[0, 0])
                    self._trace_norms[index] = float(
                        torch.linalg.vector_norm(gradient).detach().cpu()
                    )

                value.register_hook(record)
            value = layer(
                value,
                attention_mask=self.mask,
                position_ids=self.positions[None],
                past_key_values=fresh,
                use_cache=True,
                cache_position=self.positions,
                position_embeddings=self.rope,
            )
        scores = self.model.lm_head(self.model.model.norm(value)[:, -1:])[0, -1, [895, 830]]
        self.calls += 1
        return value, scores

    def finite(self, base, source=None):
        if source is None:
            point = base[0, 0].detach()
        else:
            point = torch.tensor(
                np.asarray(source, dtype=np.float32),
                dtype=self.model.dtype,
                device=self.model.device,
            )
        with torch.no_grad():
            final, scores = self._forward(base, point)
        return array(final[0]), array(scores)

    def gradient(self, base):
        with torch.enable_grad():
            point = base[0, 0].detach().clone().requires_grad_(True)
            final, scores = self._forward(base, point, trace=True)
            margin = scores[1] - scores[0]
            result = torch.autograd.grad(margin, point)[0]
        self.backwards += 1
        source_trace = np.stack([self._trace_source[i] for i in sorted(self._trace_source)])
        norms = np.asarray([self._trace_norms[i] for i in sorted(self._trace_norms)])
        assert np.array_equal(source_trace[0], array(result))
        return array(result), array(scores), array(final[0]), source_trace, norms

    def unchanged(self):
        assert cache_seals(self.cache, self.start) == self.seals


def margin(scores):
    return float(scores[1]) - float(scores[0])


def split(gradient, delta):
    g, d = gradient.astype(float), delta.astype(float)
    square = float(np.dot(g, g))
    if square == 0.0:
        return np.zeros_like(d), d.copy()
    projection = float(np.dot(g, d)) / square
    parallel = projection * g
    return parallel, d - parallel


def directions(key, delta, parallel, orthogonal):
    width = len(delta)
    seed = int(key[:16], 16)
    rng = np.random.default_rng(seed)
    choices = rng.choice(width, 2, replace=False)
    result = {"delta": delta, "parallel_F": parallel, "orthogonal_F": orthogonal}
    for index in choices:
        basis = np.zeros(width, dtype=float)
        basis[index] = 1.0
        result["coordinate_" + str(index)] = basis
    return result


def finite_difference(engine, base, gradient, direction, label):
    """Projected derivative checks; no claim to validate every Jacobian coordinate."""
    actual, rows, estimates = float(np.dot(gradient.astype(float), direction)), [], []
    source = array(base[0, 0]).astype(float)
    for step in (0.05, 0.025):
        _, plus = engine.finite(base, source + step * direction)
        release(engine.model)
        _, minus = engine.finite(base, source - step * direction)
        release(engine.model)
        estimate = (margin(plus) - margin(minus)) / (2 * step)
        bound = 0.001 + 0.02 * abs(actual)
        error = abs(actual - estimate)
        rows.append(
            dict(
                direction=label,
                step=step,
                AD=actual,
                FD=estimate,
                plus_margin=margin(plus),
                minus_margin=margin(minus),
                plus_scores=plus.tolist(),
                minus_scores=minus.tolist(),
                error=error,
                bound=bound,
                passes=bool(np.isfinite(error) and error <= bound),
            )
        )
        estimates.append(estimate)
    stability = abs(estimates[0] - estimates[1])
    stability_bound = 0.002 + 0.03 * abs(actual)
    rows.append(
        dict(
            direction=label,
            step="halving",
            AD=actual,
            FD=estimates[1],
            error=stability,
            bound=stability_bound,
            passes=bool(np.isfinite(stability) and stability <= stability_bound),
        )
    )
    return rows
