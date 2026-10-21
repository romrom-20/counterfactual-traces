"""Public-model operations used by the baseline/interaction experiment.

No whole-block Jacobians or saved native states are inserted downstream.
Only rows strictly after the changed Boolean token are intervened on.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / "settings.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def array(value):
    return value.detach().cpu().numpy().copy()


def check(records, name, error, bound=1e-4):
    item = dict(name=name, error=float(error), bound=bound)
    records.append(item)
    assert np.isfinite(item["error"]) and item["error"] <= bound, item


def load_model(offline):
    specification = SETTINGS["model"]
    tokenizer = AutoTokenizer.from_pretrained(**dict(
        pretrained_model_name_or_path=specification["model"],
        revision=specification["revision"], local_files_only=offline))
    model = AutoModelForCausalLM.from_pretrained(
        specification["model"], revision=specification["revision"],
        local_files_only=offline, dtype=torch.float32,
        attn_implementation="eager").to("mps").eval()
    model.requires_grad_(False)
    torch.set_num_threads(2)
    assert model.config._commit_hash == specification["revision"]
    assert len(model.model.layers) == 28 and model.config.hidden_size == 1536
    assert model.config._attn_implementation == "eager"
    assert tokenizer.encode(" false", add_special_tokens=False) == [895]
    assert tokenizer.encode(" true", add_special_tokens=False) == [830]
    return model


@contextmanager
def capture_mlp(layer):
    values, handles = {}, []

    def save(name, before=False):
        def hook(module, args, output=None):
            values[name] = (args[0] if before else output).detach().clone()
        return hook

    for module, name, before in (
        (layer.post_attention_layernorm, "mid", True),
        (layer.mlp.gate_proj, "gate", False),
        (layer.mlp.up_proj, "up", False),
        (layer.mlp.act_fn, "phi", False),
        (layer.mlp.down_proj, "product", True),
        (layer.mlp.down_proj, "write", False),
    ):
        registration = module.register_forward_pre_hook if before else module.register_forward_hook
        handles.append(registration(save(name, before)))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def observe(model, ids):
    states, context, handles = [], {}, []

    def first(module, args, kwargs):
        context.update(mask=kwargs["attention_mask"].detach().clone(),
                       rope=tuple(v.detach().clone() for v in kwargs["position_embeddings"]))

    def block_output(module, args, output):
        states.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    handles.append(model.model.layers[0].register_forward_pre_hook(first, with_kwargs=True))
    handles.extend(layer.register_forward_hook(block_output) for layer in model.model.layers)
    try:
        with torch.no_grad():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return states, context, result.logits[0, -1, [895, 830]].detach()


def false_anchor(capture, start):
    with torch.no_grad():
        sigma = torch.sigmoid(capture["gate"])
    return {name: array(value[0, start:]).astype(np.float64)
            for name, value in {**capture, "sigma": sigma}.items() if name != "mid"}


def corner(layer, mid, anchor, start, silu_finite, product_finite, checks):
    """Execute real MLP nodes, replacing SiLU/product by False-affine maps if requested."""
    live, saved = {}, {}

    def silu(module, args, output):
        gate = array(args[0][0, start:]).astype(np.float64)
        slope = anchor["sigma"] * (1 + anchor["gate"] * (1 - anchor["sigma"]))
        affine = anchor["phi"] + slope * (gate - anchor["gate"])
        value = output.clone()
        if not silu_finite:
            value[:, start:] = torch.tensor(affine[None], device="mps", dtype=torch.float32)
        live["phi"] = value
        saved.update(gate=gate, phi=array(value[0, start:]).astype(np.float64))
        check(checks, "silu_prefix", (value[:, :start] - output[:, :start]).abs().max(), 0.)
        return value

    def up(module, args, output):
        live["up"] = output
        saved["up"] = array(output[0, start:]).astype(np.float64)

    def product(module, args):
        affine = (anchor["product"] + anchor["up"] * (saved["phi"] - anchor["phi"])
                  + anchor["phi"] * (saved["up"] - anchor["up"]))
        value = args[0].clone()
        if not product_finite:
            value[:, start:] = torch.tensor(affine[None], device="mps", dtype=torch.float32)
        check(checks, "product_prefix", (value[:, :start] - args[0][:, :start]).abs().max(), 0.)
        live["product"] = value.detach()
        return (value,)

    handles = [layer.mlp.act_fn.register_forward_hook(silu),
               layer.mlp.up_proj.register_forward_hook(up),
               layer.mlp.down_proj.register_forward_pre_hook(product)]
    try:
        with torch.no_grad():
            original_write = layer.mlp(layer.post_attention_layernorm(mid))
    finally:
        for handle in handles:
            handle.remove()
    saved["original_write"] = array(original_write[0, start:])
    for name, parents, directions, function, smooth, expected in (
        ("silu", (anchor["gate"],), (saved["gate"] - anchor["gate"],),
         layer.mlp.act_fn, F.silu, anchor["phi"]),
        ("product", (anchor["phi"], anchor["up"]),
         (saved["phi"] - anchor["phi"], saved["up"] - anchor["up"]),
         lambda p, u: p * u, lambda p, u: p * u, anchor["product"]),
    ):
        tensor = lambda v: torch.tensor(v, device="mps", dtype=torch.float32)
        with torch.no_grad():
            primal, dual = torch.func.jvp(function, tuple(map(tensor, parents)),
                                         tuple(map(tensor, directions)))
        if name == "silu":
            analytic = anchor["sigma"] * (1 + anchor["gate"] * (1 - anchor["sigma"])) * directions[0]
        else:
            analytic = directions[0] * anchor["up"] + anchor["phi"] * directions[1]
        promote = lambda v: torch.tensor(v, dtype=torch.float64)
        fd = array((smooth(*(promote(p + 1e-5 * d) for p, d in zip(parents, directions)))
                    - smooth(*(promote(p - 1e-5 * d) for p, d in zip(parents, directions)))) / 2e-5)
        check(checks, name + "_primal", np.max(abs(array(primal).astype(np.float64) - expected)))
        check(checks, name + "_ad", np.max(abs(array(dual).astype(np.float64) - analytic)))
        check(checks, name + "_fd", np.max(abs(fd - analytic)))
        saved.update({name + "_analytic": analytic, name + "_dual": array(dual), name + "_fd": fd})
    return live["product"], saved


def continue_product(model, site, mid, full_product, reference, context, start, checks):
    """Install a product, use the real down projection and continue the entire finite suffix."""
    with torch.no_grad():
        write = model.model.layers[site - 1].mlp.down_proj(full_product)
        state = mid + write
        states = [state]
        check(checks, "prefix_state", (state[0, :start] - reference[site - 22, :start]).abs().max())
        for index in range(site, 28):
            state = model.model.layers[index](state, attention_mask=context["mask"],
                                              position_embeddings=context["rope"])
            state = state[0] if isinstance(state, tuple) else state
            states.append(state)
            check(checks, "prefix_state", (state[0, :start] - reference[index - 21, :start]).abs().max())
        normalized = model.model.norm(state)[0, -1]
        scores = F.linear(normalized, model.lm_head.weight[[895, 830]])
    return dict(product=array(full_product[0, start:]), write=array(write[0, start:]),
                trajectory=array(torch.stack(states)[:, 0, start:]),
                final=array(state[0, start:]), scores=array(scores))
