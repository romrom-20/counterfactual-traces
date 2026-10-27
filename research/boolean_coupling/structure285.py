"""First-source polynomial forecasts and frozen historical-direction controls.

All source/receiving operations except the first source SiLU remain finite.
The inherited pre-hook substitutes the observed False gate before the actual
SiLU, then installs the polynomial. Donor writes never require current SiLU.
"""

from pathlib import Path

import numpy as np

from .composition280 import SOURCES as PREVIOUS_SOURCES
from .composition280 import Engine as PreviousEngine
from .core225b import array

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ("C", "L", "D", "M", "Z", "FZ")
ARMS = ("F", *FORECAST)
CANDIDATES = FORECAST
SOURCES = (
    *dict.fromkeys(PREVIOUS_SOURCES),
    "research/boolean_coupling/structure285.py",
    "research/boolean_coupling/run285.py",
    "research/boolean_coupling/audit285.py",
    "research/boolean_coupling/preflight285.py",
    "research/boolean_coupling/fixtures285.py",
    "research/boolean_coupling/cases285.json",
    "research/boolean_coupling/history285.json",
    "research/boolean_coupling/bank285.json",
    "docs/experiments/285-native-structure.md",
)


class Engine(PreviousEngine):
    """Reuse the qualified gate-before-SiLU substitution only at source MLP1."""

    def __init__(self, model, case, donor):
        super().__init__(model, case)
        self.donor = np.asarray(donor, dtype=np.float64)
        assert self.donor.shape == (model.config.hidden_size,)
        assert np.isfinite(self.donor).all() and np.linalg.norm(self.donor) > 0
        self.c_write = None

    def degree(self, index):
        if index != 0 or self.arm in ("F", "T"):
            return 0
        return 1 if self.arm == "L" else 3

    def __enter__(self):
        super().__enter__()
        layer, start = self.model.model.layers[0], self.case["position"]

        def capture(name):
            def hook(module, args, output):
                self.record[0][name] = array(output[0, start])

            return hook

        self.handles += [
            layer.input_layernorm.register_forward_hook(capture("norm")),
            layer.post_attention_layernorm.register_forward_hook(capture("z")),
            layer.self_attn.o_proj.register_forward_hook(capture("attwrite")),
        ]

        def install(module, args, output):
            if self.arm not in ("D", "M", "Z", "FZ"):
                return output
            false_write = self.anchors[0]["write"]
            if self.arm in ("Z", "FZ"):
                row = false_write.copy()
                factor = 0.0
            else:
                factor = 1.0
                if self.arm == "M":
                    assert self.c_write is not None, "C must precede M"
                    own = self.c_write.astype(float) - false_write.astype(float)
                    factor = float(np.linalg.norm(own) / np.linalg.norm(self.donor))
                row = (false_write.astype(float) + factor * self.donor).astype(np.float32)
            self.record[0]["write"] = row.copy()
            self.record[0]["donor_factor"] = factor
            return self.replace(output, row)

        # Registered after the inherited MLP hook: raw_write remains the actual
        # polynomial branch, while write records the installed source output.
        self.handles.append(layer.mlp.register_forward_hook(install))
        return self

    def attention(self, module, query, key, value, mask, scaling, dropout=0.0, **kwargs):
        output, p = super().attention(
            module, query, key, value, mask, scaling, dropout=dropout, **kwargs
        )
        if module.layer_idx == 0:
            start = self.case["position"]
            self.record[0].update(
                full_k=array(key[0]),
                full_v=array(value[0]),
                p=array(p[0, :, start]),
                y=array(output[0, start]),
                mask=array(mask[0, 0, start]),
                scaling=np.asarray(scaling),
            )
        return output, p

    def forward(self, arm):
        assert arm in (*ARMS, "T")
        packed, metadata, context = super().forward(arm)
        first = {
            name: packed[name][0]
            for name in ("before", "mid", "gate", "up", "phi", "raw_write", "write", "source")
        }
        first.update(final=packed["final"], scores=packed["scores"])
        for name in (
            "norm",
            "z",
            "attwrite",
            "q",
            "K",
            "V",
            "full_k",
            "full_v",
            "p",
            "y",
            "mask",
            "scaling",
        ):
            first[name] = self.record[0][name]
        first.update(rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
        if arm == "C":
            self.c_write = first["write"].copy()
        metadata.update(
            full_current_first_source_activation=int(arm in ("F", "T")),
            ordinary_true=int(arm == "T"),
            only_first_source_policy=True,
            donor_factor=float(self.record[0].get("donor_factor", 0.0)),
        )
        return first, metadata, context
