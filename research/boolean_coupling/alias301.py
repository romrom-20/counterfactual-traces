"""Route a controlled alias edit to the Boolean-source generator or its readers."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .cache272 import digest
from .core225b import array, observe
from .interaction300 import SOURCES as PREVIOUS
from .query296 import record

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/alias301.py",
    "research/boolean_coupling/run301.py",
    "research/boolean_coupling/audit301.py",
    "research/boolean_coupling/fixtures301.py",
    "research/boolean_coupling/preflight301.py",
    "research/boolean_coupling/cases301.json",
    "research/boolean_coupling/cases301-flat.json",
    "research/boolean_coupling/history301.json",
    "docs/experiments/301-equivalent-alias-source-routes.md",
)
PRIMARY = dict(
    norm=0.001,
    relative=0.25,
    cosine=0.95,
    material=0.25,
    margin_abs=0.1,
    margin_fraction=0.25,
    population=24,
    per_layout=6,
    per_mention_order=12,
    control_max=8,
)
NATIVE = dict(norm=0.001, relative=0.05, cosine=0.99, margin=0.1)
BOUNDS = dict(
    composed=1e-4,
    projection_abs=1e-3,
    projection_relative=1e-5,
    head=1e-4,
    raw=1e-4,
    random_norm_relative=1e-4,
    random_norm_floor=0.001,
)
FORECAST_ARMS = ("A_F", "B_F", "AB_F", "AB_T", "BA_F", "BA_T", "RA_F", "RA_T", "AZ_F", "BZ_F")
NATIVE_ARMS = ("A_T", "B_T", "AO_T", "BO_T")


def definitions(arm):
    receiver = arm[0] if arm[0] in "AB" else "A"
    donor = (
        "B"
        if arm.startswith("AB_")
        else "A"
        if arm.startswith("BA_")
        else "R"
        if arm.startswith("RA_")
        else receiver
        if arm.startswith(("AZ_", "BZ_", "AO_", "BO_"))
        else None
    )
    return receiver, donor, arm.endswith("_T")


def sites(model):
    return sorted({0, len(model.model.layers) // 2, len(model.model.layers) - 1})


def random_banks(data, case, layers):
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256((case["key"] + "/301").encode()).digest()[:8], "big")
    )
    for j in range(layers):
        for kind in ("k", "v"):
            a, b = [data[f"bank_{alias}_{kind}{j}"].astype(float) for alias in ("A", "B")]
            rnd = rng.normal(size=a.shape)
            rnd *= np.linalg.norm(b - a, axis=-1, keepdims=True) / np.linalg.norm(
                rnd, axis=-1, keepdims=True
            )
            data[f"bank_R_{kind}{j}"] = (a + rnd).astype(np.float32)


def final_scores(model, value, length, start):
    x = torch.zeros(
        (1, length, value.shape[-1]), device=model.device, dtype=next(model.parameters()).dtype
    )
    x[0, start:] = torch.tensor(value, device=model.device, dtype=x.dtype)
    with torch.no_grad():
        scores = model.lm_head(model.model.norm(x)[:, -1:])[0, -1, [895, 830]]
    return array(scores)


def execute(model, case, phase, parent=None, parent_meta=None):
    start, begin = case["position"], case["alias_start"]
    assert 0 < begin < start
    data, witness, checks, counts = {}, {}, [], Counter()
    banks = {} if parent is None else parent
    prefix = {} if parent_meta is None else parent_meta["prefix_hashes"]
    common = {} if parent_meta is None else parent_meta["common_kv_hashes"]
    sources = {}
    original = qwen.eager_attention_forward
    witness_case = case["group"]["reverse"] == case["group"]["target"] == 0
    arm = "A_F"

    def attention(module, query, keys, values, mask, scaling, dropout=0.0, **kwargs):
        output, p = original(
            module, query, keys, values, mask, scaling=scaling, dropout=dropout, **kwargs
        )
        assert dropout == 0 and not module.training
        j = module.layer_idx
        receiver, donor, _ = definitions(arm)
        if arm in ("A_F", "B_F"):
            for kind, tensor in (("k", keys), ("v", values)):
                name = f"bank_{receiver}_{kind}{j}"
                data[name] = array(tensor[0, :, begin:start])
                banks[name] = data[name]
                hh = digest(array(tensor[0, :, :begin]))
                if receiver == "A":
                    common[f"{kind}{j}"] = hh
                else:
                    record(
                        checks,
                        f"{arm}/{j}/before_alias_{kind}_equal",
                        float(hh != common[f"{kind}{j}"]),
                        0.0,
                    )
        used_k, used_v = keys, values
        if donor is not None:
            used_k, used_v = keys.clone(), values.clone()
            for kind, tensor in (("k", used_k), ("v", used_v)):
                patch = banks[f"bank_{donor}_{kind}{j}"]
                tensor[0, :, begin:start] = torch.tensor(
                    patch, device=model.device, dtype=tensor.dtype
                )
                record(
                    checks,
                    f"{arm}/{j}/{kind}_prefix_install",
                    array(tensor[0, :, begin:start]),
                    patch,
                )
            changed, cp = original(
                module, query, used_k, used_v, mask, scaling=scaling, dropout=dropout, **kwargs
            )
            counts["extra_full_QK_softmax_PV_each"] += 1
            out, prob = output.clone(), p.clone()
            out[:, start : start + 1] = changed[:, start : start + 1]
            prob[:, :, start : start + 1] = cp[:, :, start : start + 1]
            record(
                checks,
                f"{arm}/{j}/source_output_install",
                array(out[:, start : start + 1]),
                array(changed[:, start : start + 1]),
            )
            record(
                checks,
                f"{arm}/{j}/other_query_outputs_protected",
                array(torch.cat((out[:, :start], out[:, start + 1 :]), dim=1)),
                array(torch.cat((output[:, :start], output[:, start + 1 :]), dim=1)),
            )
            output, p = out, prob
        if not arm.startswith(("AZ_", "BZ_")):
            data[f"{arm}_emit{j}"] = array(values[0, :, start])
        if witness_case and j in sites(model) and not arm.startswith(("AZ_", "BZ_")):
            raw = (
                query @ qwen.repeat_kv(used_k, module.num_key_value_groups).transpose(2, 3)
            ) * scaling
            counts["witness_extra_full_QK"] += 1
            payload = dict(
                query=array(query[0, :, start : start + 1]),
                recipient_k=array(keys[0, :, : start + 1]),
                recipient_v=array(values[0, :, : start + 1]),
                used_k=array(used_k[0, :, : start + 1]),
                used_v=array(used_v[0, :, : start + 1]),
                mask=array(mask[0, :, start : start + 1, : start + 1]),
                raw=array(raw[0, :, start : start + 1, : start + 1]),
                prob=array(p[0, :, start : start + 1, : start + 1]),
                head_output=array(output[0, start : start + 1]).transpose(1, 0, 2),
                groups=np.array(module.num_key_value_groups),
                scaling=np.array(scaling),
            )
            witness.update({f"{arm}_{j}_" + n: v for n, v in payload.items()})
        return output, p

    def output_hook(j):
        def hook(module, args, output):
            name = f"{arm}_{j}_head_output"
            if name in witness:
                witness[f"{arm}_{j}_source_write"] = array(output[0, start])

        return hook

    handles = [
        model.model.layers[j].self_attn.o_proj.register_forward_hook(output_hook(j))
        for j in sites(model)
    ]
    qwen.eager_attention_forward = attention
    try:
        for arm in FORECAST_ARMS if phase == "forecast" else NATIVE_ARMS:
            receiver, _, bit = definitions(arm)
            ids = case["aliases"][receiver]["true_ids" if bit else "false_ids"]
            states, _, scores = observe(model, torch.tensor([ids], device=model.device))
            counts["full_forwards"] += 1
            data[arm + "_final"] = array(states[-1][0, start:])
            data[arm + "_scores"] = array(scores)
            trace = np.stack([array(s[0, start:]) for s in states])
            if arm == receiver + "_F":
                prefix[receiver] = [digest(array(s[0, :start])) for s in states]
            for j, s in enumerate(states):
                record(
                    checks,
                    f"{arm}/{j}/receiver_prefix",
                    float(digest(array(s[0, :start])) != prefix[receiver][j]),
                    0.0,
                )
            sources[arm] = [digest(array(s[0, start])) for s in states]
            if arm.startswith(("AZ_", "BZ_")):
                for suffix, actual in (
                    ("final", data[arm + "_final"]),
                    ("scores", data[arm + "_scores"]),
                    ("trace", trace),
                ):
                    record(checks, arm + "/zero_" + suffix, actual, data[receiver + "_F_" + suffix])
            else:
                data[arm + "_trace"] = trace
            if arm == "B_F":
                random_banks(data, case, len(model.model.layers))
                banks.update({k: v for k, v in data.items() if k.startswith("bank_R_")})
            if arm in ("AB_F", "BA_F"):
                donor = "B" if arm == "AB_F" else "A"
                record(
                    checks,
                    arm + "/False_source_matches_donor",
                    float(sources[arm] != sources[donor + "_F"]),
                    0.0,
                )
            if arm in ("AO_T", "BO_T"):
                for suffix, actual in (
                    ("final", data[arm + "_final"]),
                    ("scores", data[arm + "_scores"]),
                    ("trace", trace),
                ):
                    record(
                        checks, arm + "/native_" + suffix, actual, data[receiver + "_T_" + suffix]
                    )
            del states
        if phase == "forecast":
            for alias, hybrid in (("A", "BA"), ("B", "AB")):
                value = (
                    data[alias + "_F_final"].astype(float)
                    + data[hybrid + "_T_final"].astype(float)
                    - data[hybrid + "_F_final"].astype(float)
                ).astype(np.float32)
                data["P_" + alias + "_final"] = value
                data["P_" + alias + "_scores"] = final_scores(
                    model, value, len(case["aliases"][alias]["false_ids"]), start
                )
                counts["extra_full_final_norm"] += 1
                counts["extra_full_vocabulary_head"] += 1
        else:
            for alias, hybrid in (("A", "BA"), ("B", "AB")):
                record(
                    checks,
                    hybrid + "/True_source_matches_donor",
                    float(parent_meta["source_hashes"][hybrid + "_T"] != sources[alias + "_T"]),
                    0.0,
                )
        assert all(np.isfinite(v).all() for v in data.values()), case["key"]
    finally:
        qwen.eager_attention_forward = original
        for handle in handles:
            handle.remove()
    return (
        data,
        witness,
        checks,
        dict(counts),
        dict(prefix_hashes=prefix, common_kv_hashes=common, source_hashes=sources),
    )
