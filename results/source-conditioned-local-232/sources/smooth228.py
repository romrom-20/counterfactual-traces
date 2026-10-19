"""Declared CPU64 receiving maps. No public Float32 whole-block AD is used."""

import torch
from torch.nn import functional as F


def promote(value):
    return value.detach().cpu().to(torch.float64)


def norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True)+eps) * weight


def parameters(module):
    return promote(module.weight), promote(module.bias) if module.bias is not None else None


def linear(x, pair):
    return F.linear(x, *pair)


def rotate(x, cosine, sine):
    a, b = x.chunk(2, dim=-1)
    return x*cosine + torch.cat((-b,a), dim=-1)*sine


def attention_map(layer, false_input, context, start):
    """Only receiving rows vary; all False prefix keys/values are fixed."""
    attn = layer.self_attn
    weight, eps = promote(layer.input_layernorm.weight), layer.input_layernorm.variance_epsilon
    q, k, v, o = (parameters(getattr(attn, name+"_proj")) for name in ("q","k","v","o"))
    width = attn.head_dim
    groups = attn.num_key_value_groups
    cos, sin = (promote(value).unsqueeze(1) for value in context["rope"])
    mask = promote(context["mask"])[:, :, start:, :]
    prefix = norm(promote(false_input[:, :start]), weight, eps)

    def heads(x):
        return x.view(x.shape[0], x.shape[1], -1, width).transpose(1,2)

    prefix_key = rotate(heads(linear(prefix,k)),cos[:, :, :start],sin[:, :, :start])
    prefix_value = heads(linear(prefix,v))

    def function(receiving):
        z = norm(receiving, weight, eps)
        query = rotate(heads(linear(z,q)),cos[:, :, start:],sin[:, :, start:])
        key = rotate(heads(linear(z,k)),cos[:, :, start:],sin[:, :, start:])
        key = torch.cat((prefix_key,key),dim=2).repeat_interleave(groups,dim=1)
        value = torch.cat((prefix_value,heads(linear(z,v))),dim=2).repeat_interleave(groups,dim=1)
        probabilities = torch.softmax(query@key.transpose(-1,-2)*attn.scaling+mask,dim=-1)
        out = (probabilities@value).transpose(1,2).contiguous().reshape(*receiving.shape)
        return receiving+linear(out,o)

    return function


def mlp_map(layer):
    weight, eps = promote(layer.post_attention_layernorm.weight), layer.post_attention_layernorm.variance_epsilon
    gate, up, down = (parameters(getattr(layer.mlp,name+"_proj")) for name in ("gate","up","down"))

    def function(receiving):
        z = norm(receiving,weight,eps)
        return receiving+linear(F.silu(linear(z,gate))*linear(z,up),down)

    return function


def qualified_jvp(function, anchor, direction):
    primal, tangent = torch.func.jvp(function,(anchor,),(direction,))
    step = 1e-5
    plus, minus = function(anchor+step*direction), function(anchor-step*direction)
    error = float((tangent-(plus-minus)/(2*step)).abs().max())
    assert torch.isfinite(tangent).all() and error<=1e-4, ("smooth228_jvp_fd",error)
    return primal.detach(), tangent.detach(), plus.detach(), minus.detach(), error
