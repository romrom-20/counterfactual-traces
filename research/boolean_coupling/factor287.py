"""Post-block1 source/receiving factorial with cached and full-shape adjoints."""

import torch

from .composition280 import digest
from .core225b import array
from .qualify286b import SOURCES as PREVIOUS
from .qualify286b import FullShape
from .readout286 import Suffix

SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/factor287.py",
    "research/boolean_coupling/run287.py",
    "research/boolean_coupling/audit287.py",
    "research/boolean_coupling/preflight287.py",
    "docs/experiments/287-source-receiver-readout.md",
)
CORNERS = ("FF", "MF", "FM", "MM")  # source first, other receiving rows second
BOUNDS = dict(
    cached_coordinate=0.002,
    cached_relative=1e-4,
    cached_score=0.001,
    full_replay=1e-4,
    source_adjoint=1e-4,
    projected_response=1e-4,
    CPU64_head=0.005,
    projected_AD_FD_absolute=0.001,
    projected_AD_FD_relative=0.02,
    FD_halving_absolute=0.002,
    FD_halving_relative=0.03,
)


class Crossed(Suffix):
    """Observe False once, retaining full prefix inputs for numerical checking."""

    def __init__(self, model, case):
        inputs, last, handles = [], {}, []

        def before(module, args):
            inputs.append(args[0].detach().clone())

        def after(module, args, output):
            last["value"] = output.detach().clone()

        handles.extend(layer.register_forward_pre_hook(before) for layer in model.model.layers)
        handles.append(model.model.layers[-1].register_forward_hook(after))
        try:
            super().__init__(model, case)
        finally:
            for handle in handles:
                handle.remove()
        assert len(inputs) == len(model.model.layers)
        full = object.__new__(FullShape)
        full.model, full.case, full.start, full.total = model, case, self.start, self.total
        full.inputs, full.false_last = inputs, last["value"]
        positions = torch.arange(self.total, device=model.device)
        full.context = dict(
            attention_mask=self.context["mask"],
            position_ids=positions[None],
            cache_position=positions,
            position_embeddings=self.context["rope"],
            use_cache=False,
        )
        full.forward_calls = full.reverse_calls = full.head_reverse_calls = 0
        self.full_shape = full

    def _forward(self, base, source, trace=False):
        source_hashes, handles = [], []

        def source_state(module, args, output):
            source_hashes.append(digest(array(output[0, 0])))

        handles.extend(
            layer.register_forward_hook(source_state) for layer in self.model.model.layers[1:]
        )
        try:
            result = super()._forward(base, source, trace)
        finally:
            for handle in handles:
                handle.remove()
        assert len(source_hashes) == len(self.model.model.layers) - 1
        self.last_source_path = source_hashes
        return result


def corners(false, forecast):
    assert false.shape == forecast.shape
    return dict(
        FF=false.detach().clone(),
        MF=torch.cat((forecast[:, :1], false[:, 1:]), dim=1).detach(),
        FM=torch.cat((false[:, :1], forecast[:, 1:]), dim=1).detach(),
        MM=forecast.detach().clone(),
    )
