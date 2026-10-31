"""Own-state source attention mass versus conditional context selection."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .cache272 import digest
from .core225b import array, observe
from .query296 import SOURCES as PREVIOUS
from .query296 import record

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/routes297.py",
    "research/boolean_coupling/run297.py",
    "research/boolean_coupling/audit297.py",
    "research/boolean_coupling/fixtures297.py",
    "research/boolean_coupling/preflight297.py",
    "research/boolean_coupling/cases297.json",
    "research/boolean_coupling/history297.json",
    "docs/experiments/297-query-routes.md",
)
ARMS = ("S", "C", "B")
PRIMARY = dict(
    norm=0.001, relative=0.05, cosine=0.99, margin=0.1, population=24, per_layout=6, control_max=8
)
EFFECT = dict(norm=0.001, relative=0.25, cosine=0.95, margin=0.1, material=0.02)
BOUNDS = dict(map=2e-7, probability_sum=1e-6, composed=1e-4, head=0.005, raw_diagnostic=1e-4)


def probabilities(current_raw, false_raw, mask, current_prob, equal, source):
    """Separate source-vs-rest mass from the conditional non-source distribution."""
    fp = torch.softmax(false_raw + mask, dim=-1, dtype=torch.float32).to(current_prob.dtype)
    cr, fr = (current_raw + mask).clone(), (false_raw + mask).clone()
    cr[..., source], fr[..., source] = -torch.inf, -torch.inf
    bc = torch.softmax(cr, dim=-1, dtype=torch.float32).to(current_prob.dtype)
    bf = torch.softmax(fr, dim=-1, dtype=torch.float32).to(current_prob.dtype)
    ac, af = current_prob[..., source : source + 1], fp[..., source : source + 1]
    sp, cp = (1 - ac) * bf, (1 - af) * bc
    sp[..., source : source + 1], cp[..., source : source + 1] = ac, af
    # Exact equality bypass avoids perturbing the unchanged computation by a
    # different rounding route through the hierarchical probability identity.
    sp, cp = torch.where(equal, current_prob, sp), torch.where(equal, current_prob, cp)
    return dict(N=current_prob, B=fp, S=sp, C=cp), dict(current_context=bc, false_context=bf)


def sites(model):
    return sorted({1, len(model.model.layers) // 2, len(model.model.layers) - 1})


def execute(model, case, phase, parent=None, parent_meta=None):
    start, key = case["position"], case["key"]
    device = model.device
    data, witness, checks, counts, source = {}, {}, [], Counter(), {}
    banks = (
        {} if parent is None else {j: parent[f"F_q{j + 1}"] for j in range(len(model.model.layers))}
    )
    original = qwen.eager_attention_forward
    is_witness = case["group"]["reverse"] == case["group"]["target"] == 0
    arm = "F" if phase == "forecast" else "T"

    def attention(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        output, prob = original(
            module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
        )
        native_prob = prob
        assert dropout == 0 and not module.training
        j = module.layer_idx
        current = array(query[0, :, start + 1 :])
        if arm == "F":
            banks[j] = current.copy()
            data[f"F_q{j + 1}"] = current.copy()
        if j == 0:
            record(checks, arm + "/first_block_query_unchanged", current, banks[j])
        intervene = arm not in ("F", "T")
        witness_here = is_witness and j in sites(model) and not arm.startswith("Z")
        if not (intervene or witness_here):
            return output, prob
        fquery = query.clone()
        fquery[0, :, start + 1 :] = torch.tensor(banks[j], device=device, dtype=query.dtype)
        record(checks, f"{arm}/{j}/false_query_bank", array(fquery[0, :, start + 1 :]), banks[j])
        kk = qwen.repeat_kv(keys, module.num_key_value_groups)
        vv = qwen.repeat_kv(values, module.num_key_value_groups)
        cr = (query @ kk.transpose(2, 3)) * scaling
        fr = (fquery @ kk.transpose(2, 3)) * scaling
        counts["extra_full_QK"] += 2
        selected = "N" if arm in ("F", "T", "O") else arm[-1]
        corners, context = probabilities(
            cr,
            fr,
            mask[:, :, :, : keys.shape[-2]],
            native_prob,
            torch.all(query == fquery, dim=-1, keepdim=True),
            start,
        )
        counts["extra_full_softmax"] += 3
        if intervene:
            pp = corners[selected]
            yy = (pp @ vv).transpose(1, 2).contiguous()
            changed, cp = output.clone(), prob.clone()
            changed[:, start + 1 :] = yy[:, start + 1 :]
            cp[:, :, start + 1 :] = pp[:, :, start + 1 :]
            record(
                checks,
                f"{arm}/{j}/output_install",
                array(changed[:, start + 1 :]),
                array(yy[:, start + 1 :]),
            )
            record(
                checks,
                f"{arm}/{j}/protected_output",
                array(changed[:, : start + 1]),
                array(output[:, : start + 1]),
            )
            record(
                checks,
                f"{arm}/{j}/protected_prob",
                array(cp[:, :, : start + 1]),
                array(prob[:, :, : start + 1]),
            )
            output, prob = changed, cp
            counts["extra_full_PV"] += 1
        if witness_here:
            payload = dict(
                current_q=current,
                false_q=banks[j],
                k=array(keys[0]),
                v=array(values[0]),
                groups=np.array(module.num_key_value_groups),
                scaling=np.array(scaling),
                mask=array(mask[0, :, start + 1 :, : keys.shape[-2]]),
                current_raw=array(cr[0, :, start + 1 :]),
                false_raw=array(fr[0, :, start + 1 :]),
                current_prob=array(native_prob[0, :, start + 1 :]),
                false_prob=array(corners["B"][0, :, start + 1 :]),
                selected_prob=array(prob[0, :, start + 1 :]),
                selected_output=array(output[0, start + 1 :]).transpose(1, 0, 2),
            )
            payload.update({n: array(v[0, :, start + 1 :]) for n, v in context.items()})
            for corner in ("B", "S", "C", "N"):
                if corner == selected:
                    pc, yc = payload["selected_prob"], payload["selected_output"]
                else:
                    p = corners[corner]
                    y = p @ vv
                    pc, yc = array(p[0, :, start + 1 :]), array(y[0, :, start + 1 :])
                    counts["witness_extra_full_PV"] += 1
                payload[corner + "_prob"], payload[corner + "_output"] = pc, yc
            witness.update({f"{arm}_{j}_" + n: v for n, v in payload.items()})
        return output, prob

    qwen.eager_attention_forward = attention
    prefix = None if parent_meta is None else parent_meta["false_prefix_hashes"]
    try:
        for arm in ("F", *ARMS, "ZS", "ZC", "ZB") if phase == "forecast" else ("T", "O"):
            ids = case["false_ids"] if arm == "F" or arm.startswith("Z") else case["true_ids"]
            states, _, scores = observe(model, torch.tensor([ids], device=device))
            counts["full_forwards"] += 1
            final, scores = array(states[-1][0, start:]), array(scores)
            trace = np.stack([array(s[0, start + 1 :]) for s in states])
            assert all(np.isfinite(x).all() for x in [final, scores, trace]), key
            if arm == "F":
                prefix = [digest(array(s[0, :start])) for s in states]
            for j, state in enumerate(states):
                record(
                    checks,
                    f"{arm}/{j}/prefix_exact",
                    float(digest(array(state[0, :start])) != prefix[j]),
                    0.0,
                )
            source[arm] = [digest(array(s[0, start])) for s in states]
            data[arm + "_final"], data[arm + "_scores"] = final, scores
            if not arm.startswith("Z"):
                data[arm + "_trace"] = trace
            if arm.startswith("Z"):
                record(checks, arm + "/zero_final", final, data["F_final"])
                record(checks, arm + "/zero_scores", scores, data["F_scores"])
                record(checks, arm + "/zero_trace", trace, data["F_trace"])
            if arm == "O":
                record(checks, "O/native_final", final, data["T_final"])
                record(checks, "O/native_scores", scores, data["T_scores"])
                record(checks, "O/native_trace", trace, data["T_trace"])
            if arm == "T":
                for a in ARMS:
                    record(
                        checks,
                        a + "/native_source_exact",
                        float(source["T"] != parent_meta["source_hashes"][a]),
                        0.0,
                    )
            del states
    finally:
        qwen.eager_attention_forward = original
    return (
        data,
        witness,
        checks,
        dict(counts),
        dict(false_prefix_hashes=prefix, source_hashes=source),
    )
