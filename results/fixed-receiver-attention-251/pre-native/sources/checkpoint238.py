"""Read pinned cached tensors for separate NumPy arithmetic, without a model forward."""

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .audit229 import sha


def locate(root):
    spec = json.loads((root/"research/boolean_coupling/settings.json").read_text())["model"]
    folder = Path(snapshot_download(repo_id=spec["model"], revision=spec["revision"], local_files_only=True))
    return folder, dict(revision=spec["revision"], config_sha256=sha(folder/"config.json"),
        checkpoint_sha256=sha(folder/"model.safetensors"))


def weights(handle, config, i, kind):
    def get(name):
        return handle.get_tensor(f"model.layers.{i}."+name).to(torch.float32).numpy().astype(float)
    if kind == "attention":
        result = dict(attn_norm=get("input_layernorm.weight"), attn_eps=np.array(config["rms_norm_eps"]),
            width=np.array(config["hidden_size"]//config["num_attention_heads"]),
            groups=np.array(config["num_attention_heads"]//config["num_key_value_heads"]),
            scaling=np.array((config["hidden_size"]//config["num_attention_heads"])**-.5))
        for name in ("q", "k", "v", "o"):
            for suffix in ("weight", "bias"):
                key = f"model.layers.{i}.self_attn.{name}_proj.{suffix}"
                if key in handle.keys():
                    result[name+"_"+suffix] = get(f"self_attn.{name}_proj.{suffix}")
        return result
    result = dict(norm=get("post_attention_layernorm.weight"), eps=np.array(config["rms_norm_eps"]))
    for name in ("gate", "up", "down"):
        for suffix in ("weight", "bias"):
            key = f"model.layers.{i}.mlp.{name}_proj.{suffix}"
            if key in handle.keys():
                result[name+"_"+suffix] = get(f"mlp.{name}_proj.{suffix}")
    return result


def mlp_derivative(w, x, d):
    rms = np.sqrt(np.mean(x*x, -1, keepdims=True)+w["eps"])
    z = x/rms*w["norm"]
    dz = w["norm"]*(d/rms-x*np.mean(x*d, -1, keepdims=True)/rms**3)
    g, u = (z@w[n+"_weight"].T+w.get(n+"_bias", 0) for n in ("gate", "up"))
    dg, du = (dz@w[n+"_weight"].T for n in ("gate", "up"))
    sigma = 1/(1+np.exp(-g))
    return d+((sigma*(1+g*(1-sigma))*dg)*u+g*sigma*du)@w["down_weight"].T


def read_head(folder):
    config = json.loads((folder/"config.json").read_text())
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        name = "lm_head.weight" if "lm_head.weight" in handle.keys() else "model.embed_tokens.weight"
        head = handle.get_slice(name)
        rows = torch.cat([head[i:i+1] for i in (895, 830)]).to(torch.float32).numpy().astype(float)
        return dict(final_norm=handle.get_tensor("model.norm.weight").to(torch.float32).numpy().astype(float),
            final_head=rows, final_eps=np.array(config["rms_norm_eps"]))
