"""Four finite/False-affine attention primitives, with explicit arithmetic."""

import hashlib

import torch

from .smooth228 import linear, norm, parameters, promote, rotate

BITS = tuple(f"{i:04b}" for i in range(16))


def norm_direction(x, direction, weight, eps):
    scale = torch.sqrt(x.square().mean(-1, keepdim=True)+eps)
    return weight*(direction/scale-x*(x*direction).mean(-1, keepdim=True)/scale**3)


def normalized_paths(layer, anchor, dz, context):
    """False-linear output increments from a normalized-input edit, not residual."""
    attn = layer.self_attn
    width = attn.head_dim
    cosine, sine = (promote(v).unsqueeze(1) for v in context["rope"])

    def projected(name):
        value = torch.nn.functional.linear(dz, promote(getattr(attn, name+"_proj").weight))
        value = value.view(*value.shape[:-1], -1, width).transpose(1, 2)
        if name != "v":
            value = rotate(value, cosine, sine)
        if name != "q":
            value = value.repeat_interleave(attn.num_key_value_groups, dim=1)
        return value

    q, k, v = (projected(name) for name in ("q", "k", "v"))
    p, original_v = promote(anchor["probabilities"]), promote(anchor["value"])
    logits = dict(query=q@promote(anchor["key"]).transpose(-1, -2)*attn.scaling,
                  key=promote(anchor["query"])@k.transpose(-1, -2)*attn.scaling)
    contexts = {name:(p*(dl-(p*dl).sum(-1, keepdim=True)))@original_v for name, dl in logits.items()}
    contexts["value"] = p@v
    outputs = {}
    for name, value in contexts.items():
        flat = value.transpose(1, 2).contiguous().reshape_as(dz)
        outputs[name] = torch.nn.functional.linear(flat, promote(attn.o_proj.weight))
    return outputs


def affine(name, anchor_value, parents, directions, function, analytic, checks, cache):
    """Qualify each distinct actual smooth direction, with no MPS AD."""
    digest = hashlib.sha256(name.encode())
    for d in directions:
        digest.update(d.detach().contiguous().numpy().tobytes())
    key = digest.hexdigest()
    if key not in cache:
        primal, tangent = torch.func.jvp(function, tuple(parents), tuple(directions))
        plus = function(*(p+1e-5*d for p, d in zip(parents, directions)))
        minus = function(*(p-1e-5*d for p, d in zip(parents, directions)))
        ad_error = float((tangent-analytic).abs().max())
        fd_error = float((tangent-(plus-minus)/2e-5).abs().max())
        assert torch.isfinite(tangent).all() and ad_error <= 1e-9 and fd_error <= 1e-4, (name, ad_error, fd_error)
        checks.append(dict(name=name, witness_key=key, ad_error=ad_error, fd_error=fd_error,
                           primal_error=float((primal-anchor_value).abs().max())))
        start = cache["start"]
        def receiving(value):
            return (value[0, start:] if name == "norm" else value[0, :, start:]).detach().clone()
        cache[key] = {label:receiving(value) for label, value in
                      (("tangent", tangent), ("plus", plus), ("minus", minus))}
    return anchor_value+analytic


def corner(layer, false_input, receiving_input, context, start, anchor, bits, double,
           checks=None, cache=None, norm_rows=None):
    attn = layer.self_attn
    x = promote(false_input)
    operand = promote(receiving_input) if double else receiving_input
    direction = promote(receiving_input)-x
    weight, eps = promote(layer.input_layernorm.weight), layer.input_layernorm.variance_epsilon
    norm_anchor = promote(anchor["norm"])
    tangent = norm_direction(x, direction, weight, eps)
    if bits[0] == "0":
        z64 = (affine("norm", norm_anchor, [x], [direction], lambda v:norm(v, weight, eps),
                      tangent, checks, cache) if double else norm_anchor+tangent)
        z = z64 if double else z64.to("mps", torch.float32)
    else:
        z = norm(operand, weight, eps) if double else layer.input_layernorm(operand)
        if norm_rows is not None:
            selected = (norm_anchor+tangent) if double else (norm_anchor+tangent).to("mps", torch.float32)
            selected = selected.clone()
            selected[:, norm_rows] = z[:, norm_rows]
            z = selected

    def heads(value):
        return value.view(*value.shape[:-1], -1, attn.head_dim).transpose(1, 2)

    def projected(name):
        module = getattr(attn, name+"_proj")
        return heads(linear(z, parameters(module)) if double else module(z))

    cosine, sine = ((promote(v) if double else v).unsqueeze(1) for v in context["rope"])
    q, k = (rotate(projected(name), cosine, sine) for name in ("q", "k"))
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
    v = projected("v").repeat_interleave(attn.num_key_value_groups, dim=1)
    q0, k0, v0 = (promote(anchor[name]) for name in ("query", "key", "value"))
    p0, l0 = promote(anchor["probabilities"]), promote(anchor["logits"])
    if bits[1] == "0":
        dq, dk = promote(q)-q0, promote(k)-k0
        dl = (dq@k0.transpose(-1, -2)+q0@dk.transpose(-1, -2))*attn.scaling
        logits = (affine("qk", l0, [q0, k0], [dq, dk], lambda a,b:a@b.transpose(-1, -2)*attn.scaling,
                         dl, checks, cache) if double else l0+dl)
        if not double:
            logits = logits.to("mps", torch.float32)
    else:
        logits = q@k.transpose(-1, -2)*attn.scaling
    mask = promote(context["mask"]) if double else context["mask"]
    if bits[2] == "0":
        dl = promote(logits)-l0
        dp = p0*(dl-(p0*dl).sum(-1, keepdim=True))
        p = (affine("softmax", p0, [l0], [dl], lambda a:torch.softmax(a+promote(context["mask"]), -1),
                    dp, checks, cache) if double else p0+dp)
        if not double:
            p = p.to("mps", torch.float32)
    else:
        p = torch.softmax(logits+mask, dim=-1)
    if bits[3] == "0":
        dp, dv = promote(p)-p0, promote(v)-v0
        da = dp@v0+p0@dv
        a0 = promote(anchor["weighted"])
        a = (affine("av", a0, [p0, v0], [dp, dv], lambda a,b:a@b,
                    da, checks, cache) if double else a0+da)
        if not double:
            a = a.to("mps", torch.float32)
    else:
        a = p@v
    flat = a.transpose(1, 2).contiguous().reshape_as(operand)
    write = linear(flat, parameters(attn.o_proj)) if double else attn.o_proj(flat)
    return operand+write, z
