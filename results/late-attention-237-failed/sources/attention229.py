"""Fixed attention-2 mixed-term graph, with distinct public and smooth arithmetic."""

import torch
from torch.nn import functional as F

from .smooth228 import linear, norm, parameters, promote, rotate

PRIMITIVES = ("QK", "SM", "AVs", "AVd")


def graph(layer, state, context, start, double=False):
    attn = layer.self_attn
    x = promote(state) if double else state
    z = (norm(x, promote(layer.input_layernorm.weight),
              layer.input_layernorm.variance_epsilon) if double else layer.input_layernorm(x))
    width = attn.head_dim

    def heads(value):
        return value.view(*value.shape[:-1], -1, width).transpose(1, 2)

    def project(name):
        module = getattr(attn, name + "_proj")
        return heads(linear(z, parameters(module)) if double else module(z))

    cosine, sine = ((promote(v) if double else v).unsqueeze(1) for v in context["rope"])
    query, key = (rotate(project(name), cosine, sine) for name in ("q", "k"))
    key = key.repeat_interleave(attn.num_key_value_groups, dim=1)
    value = project("v").repeat_interleave(attn.num_key_value_groups, dim=1)
    # Full rows preserve the public projection/matmul shapes. Mask is added only once.
    logits = query @ key.transpose(-1, -2) * attn.scaling
    mask = promote(context["mask"]) if double else context["mask"]
    probabilities = torch.softmax(logits + mask, dim=-1)
    weighted = probabilities @ value
    flat = weighted.transpose(1, 2).contiguous().reshape_as(x)
    write = linear(flat, parameters(attn.o_proj)) if double else attn.o_proj(flat)
    return dict(query=query[:, :, start:], key=key, value=value,
                logits=logits[:, :, start:], probabilities=probabilities[:, :, start:],
                weighted=weighted[:, :, start:], write=write, norm=z)


def output(layer, weighted, double=False):
    flat = weighted.transpose(1, 2).contiguous()
    flat = flat.reshape(flat.shape[0], flat.shape[1], -1)
    module = layer.self_attn.o_proj
    # All components are increments, so the output projection bias cancels.
    return F.linear(flat, promote(module.weight) if double else module.weight)


def mixed(layer, corners, context, start, double=False):
    zero, source, receiver, both = (corners[k] for k in ("zero", "source", "receiver", "both"))
    mask = context["mask"][:, :, start:]
    if double:
        mask = promote(mask)
    added_logits = source["logits"] + receiver["logits"] - zero["logits"]
    added_prob = torch.softmax(added_logits + mask, dim=-1)
    cross_logit = ((receiver["query"] - zero["query"])
                   @ (source["key"] - zero["key"]).transpose(-1, -2)
                   * layer.self_attn.scaling)
    soft_mix = added_prob - source["probabilities"] - receiver["probabilities"] + zero["probabilities"]
    avs = ((receiver["probabilities"] - zero["probabilities"])
           @ (source["value"] - zero["value"]))
    avd = ((source["probabilities"] - zero["probabilities"])
           @ (receiver["value"] - zero["value"]))
    contexts = dict(QK=(both["probabilities"] - added_prob) @ both["value"],
                    SM=soft_mix @ both["value"], AVs=avs, AVd=avd)
    vectors = {k: output(layer, v, double) for k, v in contexts.items()}
    return dict(added_logits=added_logits, added_prob=added_prob, cross_logit=cross_logit,
                soft_mix=soft_mix, **{"context_"+k: v for k, v in contexts.items()},
                **{"vector_"+k: v for k, v in vectors.items()})


def primitive_removed(layer, corners, terms, context, start, name):
    """Change a named primitive in the finite native public-arithmetic graph."""
    native = corners["both"]
    if name == "QK":
        probabilities = torch.softmax(native["logits"] - terms["cross_logit"]
                                      + context["mask"][:, :, start:], dim=-1)
        weighted = probabilities @ native["value"]
    elif name == "SM":
        probabilities = native["probabilities"] - terms["soft_mix"]
        weighted = probabilities @ native["value"]
    else:
        weighted = native["weighted"] - terms["context_"+name]
    flat = weighted.transpose(1, 2).contiguous()
    flat = flat.reshape(flat.shape[0], flat.shape[1], -1)
    return layer.self_attn.o_proj(flat)
