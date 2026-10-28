"""Receiving RMSNorm versus gated-MLP response, conditional on289's edited attention."""

from contextlib import contextmanager

import numpy as np
import torch

from .core225b import array
from .interface289 import SOURCES as PREVIOUS
from .interface289 import Interface

ARMS = ("00", "01", "10", "11")  # receiving normalizers, gated MLP
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/receiver290.py",
    "research/boolean_coupling/run290.py",
    "research/boolean_coupling/audit290.py",
    "research/boolean_coupling/preflight290.py",
    "docs/experiments/290-receiving-nonlinearity.md",
)
PRIMARY = dict(
    hypotheses={"NORMALIZATION": "10", "GATED_MLP": "01"},
    norm_min=0.001,
    relative_max=0.25,
    cosine_min=0.95,
    finite_absolute=0.005,
    finite_relative=0.25,
    material_min=0.02,
    population_min=24,
    per_layout_min=6,
)


class Receiver(Interface):
    def __init__(self, model, case, write):
        self.receiving_anchors = []
        super().__init__(model, case, write)
        assert len(self.receiving_anchors) == 2
        self.counts.update(
            receiving_norm_calls=0, receiving_MLP_reanchors=0, receiver_norm_probe_VJPs=0
        )

    def native(self, base):
        if not self.collect_anchors:
            return super().native(base)
        saved, handles = {}, []
        for j, layer in enumerate(self.model.model.layers[1:]):
            saved[j] = {}

            def node(name, j=j):
                def hook(module, args, output):
                    saved[j][name] = output.detach().clone()
                    if name == "post_norm":
                        saved[j]["mid"] = args[0].detach().clone()

                return hook

            for module, name in (
                (layer.input_layernorm, "input_norm"),
                (layer.post_attention_layernorm, "post_norm"),
                (layer.mlp, "write"),
            ):
                handles.append(module.register_forward_hook(node(name)))
        try:
            result = super().native(base)
        finally:
            for h in handles:
                h.remove()
        self.receiving_anchors.append(saved)
        return result

    @contextmanager
    def receiving(self, j, arm):
        normalization, mlp = map(int, arm)
        layer = self.model.model.layers[j + 1]
        f, e = (self.receiving_anchors[a][j] for a in (0, 1))
        handles = []
        original_mlp = layer.mlp.forward
        prior_attribute = layer.mlp.__dict__.get("forward")

        def input_norm(module, args, output):
            if not normalization:
                return output
            shifted = self.tapes[1][j] + (args[0] - self.tapes[0][j])
            new = type(module).forward(module, shifted)
            self.counts["receiving_norm_calls"] += 1
            return f["input_norm"] + (new - e["input_norm"])

        def post_norm(module, args, output):
            if not normalization:
                return output
            new = type(module).forward(module, e["mid"] + (args[0] - f["mid"]))
            self.counts["receiving_norm_calls"] += 1
            return f["post_norm"] + (new - e["post_norm"])

        def gated(value):
            self.counts["receiving_MLP_reanchors"] += 1
            return f["write"] + (
                original_mlp(e["post_norm"] + (value - f["post_norm"])) - e["write"]
            )

        with super().receiving(j, "111"):
            handles.append(layer.input_layernorm.register_forward_hook(input_norm))
            handles.append(layer.post_attention_layernorm.register_forward_hook(post_norm))
            if mlp:
                layer.mlp.forward = gated
            try:
                yield
            finally:
                if mlp:
                    if prior_attribute is None:
                        del layer.mlp.__dict__["forward"]
                    else:
                        layer.mlp.forward = prior_attribute
                for h in handles:
                    h.remove()


def norm_probes(engine):
    """One receiving-row cotangent, both native cached norm shapes and both anchors."""
    data = {}
    width = engine.model.config.hidden_size
    eta = float(np.float32(1 / np.sqrt(width)))
    data["receiver_norm_eta"] = np.asarray(eta)
    for kind in ("input", "post"):
        weights = [
            array(
                getattr(
                    layer, "input_layernorm" if kind == "input" else "post_attention_layernorm"
                ).weight
            )
            for layer in engine.model.model.layers[1:]
        ]
        data[kind + "_weights"] = np.stack(weights)
        for anchor in (0, 1):
            xs, ys, grads = [], [], []
            for j, layer in enumerate(engine.model.model.layers[1:]):
                norm = layer.input_layernorm if kind == "input" else layer.post_attention_layernorm
                base = (
                    engine.tapes[anchor][j]
                    if kind == "input"
                    else engine.receiving_anchors[anchor][j]["mid"]
                )
                with torch.enable_grad():
                    point = base.clone().requires_grad_(True)
                    out = norm(point)
                    grad = torch.autograd.grad(out[0, -1].sum() * eta, point)[0]
                xs.append(array(point[0, -1]))
                ys.append(array(out[0, -1]))
                grads.append(array(grad[0, -1]))
                engine.counts["receiver_norm_probe_VJPs"] += 1
            data[f"{kind}{anchor}_x"] = np.stack(xs)
            data[f"{kind}{anchor}_y"] = np.stack(ys)
            data[f"{kind}{anchor}_AD"] = np.stack(grads)
    return data


def norm_records(data, eps):
    rows = []
    eta = float(data["receiver_norm_eta"])
    for kind in ("input", "post"):
        weights = data[kind + "_weights"].astype(float)
        for anchor in (0, 1):
            for j, w in enumerate(weights):
                x = data[f"{kind}{anchor}_x"][j].astype(float)
                scale = np.sqrt(np.mean(x * x) + eps)
                y = w * x / scale
                g = w * eta / scale - x * np.dot(w * eta, x) / (len(x) * scale**3)
                for mode, actual, expected in (
                    ("primal", data[f"{kind}{anchor}_y"][j], y),
                    ("VJP", data[f"{kind}{anchor}_AD"][j], g),
                ):
                    err = float(np.max(abs(actual.astype(float) - expected)))
                    rows.append(
                        dict(
                            kind=kind,
                            anchor=anchor,
                            layer=j + 2,
                            mode=mode,
                            error=err,
                            bound=1e-4,
                            passes=bool(np.isfinite(err) and err <= 1e-4),
                        )
                    )
    return rows
