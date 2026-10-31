"""Retrospective fixed-block query interaction, with finite downstream interventions."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .cache272 import digest
from .core225b import array, observe
from .generator299 import SOURCES as PREVIOUS
from .query296 import record

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/interaction300.py",
    "research/boolean_coupling/run300.py",
    "research/boolean_coupling/audit300.py",
    "research/boolean_coupling/preflight300.py",
    "research/boolean_coupling/parents300.json",
    "docs/experiments/300-query-interaction.md",
)
ARMS = ("A", "R", "J", "O")
CRITERIA = dict(
    norm=0.001,
    relative=0.25,
    cosine=0.95,
    material=0.02,
    margin_abs=0.1,
    margin_fraction=0.25,
    population=6,
    per_schema=3,
    per_layout=1,
    control_max=2,
)
BOUNDS = dict(
    composed=1e-4,
    projection_abs=1e-3,
    projection_relative=1e-5,
    contrast_relative=1e-3,
    head=1e-4,
    raw_diagnostic=1e-4,
    random_norm_relative=1e-4,
)


class CalibrationFailure(RuntimeError):
    def __init__(self, data, checks, counts):
        super().__init__("Native replay failed; stop before conditional interventions")
        self.data, self.checks, self.counts = data, checks, dict(counts)


def execute(model, case, parent, reference, parent_meta):
    """Uses known native parent queries. This is NOT a False-only forecast."""
    start, j = case["position"], len(model.model.layers) // 2
    prefix = f"T_{j}_"
    device, original = model.device, qwen.eager_attention_forward
    data, cache, checks, counts, fingerprints = {}, {}, [], Counter(), {}
    state, arm = {}, "T0"

    def attention(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        y, p = original(
            module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
        )
        if module.layer_idx != j:
            return y, p
        for n, value in (
            ("N_q", array(query[0, :, start + 1 :])),
            ("k", array(keys[0])),
            ("v", array(values[0])),
            ("mask", array(mask[0, :, start + 1 :, : keys.shape[-2]])),
            ("N_output", array(y[0, start + 1 :]).transpose(1, 0, 2)),
            ("N_prob", array(p[0, :, start + 1 :])),
        ):
            record(checks, arm + "/native_parent_" + n, value, parent[prefix + n])
        if arm == "T0":
            qa = query.clone()
            q = {
                c: torch.tensor(parent[prefix + c + "_q"], device=device, dtype=query.dtype)
                for c in ("C", "M", "B")
            }
            qa[0, :, start + 1 :] = (q["C"] + q["M"]) - q["B"]
            ya, pa = original(
                module, qa, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
            )
            counts["extra_full_QK"] += 1
            counts["extra_full_softmax"] += 1
            counts["extra_full_PV"] += 1
            state.update(head=y, A_head=ya)
            data.update(
                A_query=array(qa[0, :, start + 1 :]),
                A_head=array(ya[0, start + 1 :]).transpose(1, 0, 2),
                A_prob=array(pa[0, :, start + 1 :]),
            )
        return y, p

    def output_hook(module, args, output):
        if arm == "T0":
            head = state["head"]
            cache["N"] = output[0, start + 1 :].clone()
            for corner in ("C", "M", "B", "A"):
                yy = state["A_head"] if corner == "A" else head.clone()
                if corner != "A":
                    yy[0, start + 1 :] = torch.tensor(
                        parent[prefix + corner + "_output"], device=device, dtype=head.dtype
                    ).transpose(0, 1)
                write = functional.linear(
                    yy.reshape(1, yy.shape[1], -1), module.weight, module.bias
                )
                cache[corner] = write[0, start + 1 :].clone()
                counts["extra_full_output_projection"] += 1
            cache["L"] = (cache["C"] + cache["M"]) - cache["B"]
            cache["R"] = cache["L"] + (cache["N"] - cache["A"])
            delta = array(cache["N"] - cache["L"]).astype(float)
            seed = int.from_bytes(
                hashlib.sha256((case["key"] + "/300").encode()).digest()[:8], "big"
            )
            rng = np.random.default_rng(seed)
            rnd = rng.normal(size=delta.shape)
            rnd *= np.linalg.norm(delta, axis=-1, keepdims=True) / np.linalg.norm(
                rnd, axis=-1, keepdims=True
            )
            data["random_delta"] = rnd.astype(np.float32)
            cache["J"] = cache["L"] + torch.tensor(
                data["random_delta"], device=device, dtype=output.dtype
            )
            data.update({"write_" + c: array(w) for c, w in cache.items()})
            return None
        chosen = cache["N" if arm == "O" else arm]
        changed = output.clone()
        changed[0, start + 1 :] = chosen
        record(checks, arm + "/write_install", array(changed[0, start + 1 :]), array(chosen))
        record(
            checks,
            arm + "/protected_output",
            array(changed[0, : start + 1]),
            array(output[0, : start + 1]),
        )
        return changed

    qwen.eager_attention_forward = attention
    handle = model.model.layers[j].self_attn.o_proj.register_forward_hook(output_hook)
    try:
        for arm in ("T0", "L", "A", "R", "J", "O"):
            states, _, scores = observe(model, torch.tensor([case["true_ids"]], device=device))
            counts["full_forwards"] += 1
            data[arm + "_final"] = array(states[-1][0, start:])
            data[arm + "_trace"] = np.stack([array(s[0, start + 1 :]) for s in states])
            data[arm + "_scores"] = array(scores)
            source = [digest(array(s[0, start])) for s in states]
            protected = [digest(array(s[0, : start + 1])) for s in states]
            fingerprints[arm] = dict(source=source, protected=protected)
            record(
                checks,
                arm + "/source_matches_299",
                float(source != parent_meta["source_hashes"]["T"]),
                0.0,
            )
            if arm == "T0":
                for suffix in ("final", "trace", "scores"):
                    record(
                        checks,
                        "T0/native_299_" + suffix,
                        data["T0_" + suffix],
                        reference["T_" + suffix],
                    )
                if not all(r["passes"] for r in checks):
                    raise CalibrationFailure(data, checks, counts)
            else:
                record(
                    checks,
                    arm + "/protected_trajectory",
                    float(protected != fingerprints["T0"]["protected"]),
                    0.0,
                )
                record(
                    checks,
                    arm + "/pre_intervention_trajectory",
                    data[arm + "_trace"][:j],
                    data["T0_trace"][:j],
                )
            if arm == "O":
                for suffix in ("final", "trace", "scores"):
                    record(
                        checks,
                        "O/native_replay_" + suffix,
                        data["O_" + suffix],
                        data["T0_" + suffix],
                    )
            assert all(np.isfinite(v).all() for v in data.values()), case["key"]
            del states
    finally:
        qwen.eager_attention_forward = original
        handle.remove()
    return (
        data,
        checks,
        dict(counts),
        dict(
            fingerprints=fingerprints, retrospective=True, native_parents_supplied=True, block=j + 1
        ),
    )
