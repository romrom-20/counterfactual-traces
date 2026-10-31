"""Finite terminal message interventions at fixed, saved native receiving inputs."""

import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from .core225b import array
from .publish285 import read, sha
from .routes297 import SOURCES as PREVIOUS
from .routes297 import record

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/messages298.py",
    "research/boolean_coupling/run298.py",
    "research/boolean_coupling/audit298.py",
    "research/boolean_coupling/fixtures298.py",
    "research/boolean_coupling/preflight298.py",
    "research/boolean_coupling/parents298.json",
    "docs/experiments/298-context-messages.md",
)
GROUPS = (
    "chosen",
    "swapped",
    "union",
    "complement",
    "neighbor",
    "random1",
    "random2",
    "all",
    "zero",
)
PRIMARY = dict(
    norm=0.001,
    relative=0.25,
    cosine=0.95,
    material=0.02,
    margin_absolute=0.1,
    margin_relative=0.25,
    population=6,
    per_schema=3,
    per_layout=1,
    control_max=2,
)
BOUNDS = dict(
    group_map=1e-8,
    message=1e-4,
    final_max=1e-3,
    final_relative=1e-5,
    head=0.005,
    raw_diagnostic=1e-4,
    random_norm_relative=1e-6,
)


def read_parent(folder, selection):
    paths = dict(
        native=folder / "observations" / (selection["key"] + ".npz"),
        witness=folder / "witnesses-native" / (selection["key"] + ".npz"),
    )
    for name, path in paths.items():
        assert sha(path) == selection["parent_hashes"][name], path.name
    with np.load(paths["native"]) as data:
        parent = dict(
            input=data["T_trace"][-2],
            expected_final=data["T_final"][1:],
            expected_scores=data["T_scores"],
        )
    with np.load(paths["witness"]) as data:
        for name, key in dict(
            head_native="N_output",
            head_false_query="B_output",
            p_native="N_prob",
            p_false_query="B_prob",
            values="v",
            groups="groups",
        ).items():
            parent[name] = data["T_27_" + key]
    assert len(parent["input"]) == selection["length"] - selection["position"] - 1
    return parent


def roles(ids, source, schema, eom_id, boolean_ids):
    assert schema in {"generic", "descriptive"}
    ends = [j for j, value in enumerate(ids[:source]) if value == eom_id]
    assert len(ends) == 2
    instruction = [j for j in range(ends[0]) if ids[j] in boolean_ids]
    assert instruction
    endset, boolset = set(ends), set(instruction)
    assert not endset & boolset
    chosen, swapped = (ends, instruction) if schema == "generic" else (instruction, ends)
    neighbors = []
    forbidden = endset | boolset
    for j in chosen:
        k = j - 1
        while k in forbidden or k in neighbors:
            k -= 1
        assert 0 <= k < source
        neighbors.append(k)
    return dict(
        chosen=chosen,
        swapped=swapped,
        union=sorted(forbidden),
        complement=[j for j in range(len(ids)) if j not in set(chosen)],
        neighbor=sorted(neighbors),
        all=list(range(len(ids))),
        zero=[],
    )


def load_terminal():
    spec = read(ROOT / "research/boolean_coupling/settings.json")["model"]
    folder = Path(
        snapshot_download(repo_id=spec["model"], revision=spec["revision"], local_files_only=True)
    )
    cp = dict(
        revision=spec["revision"],
        config_sha256=sha(folder / "config.json"),
        checkpoint_sha256=sha(folder / "model.safetensors"),
    )
    config = Qwen2Config(**read(folder / "config.json"))
    assert config.num_hidden_layers == 28 and config.hidden_size == 1536
    device = torch.device("mps")
    mlp = Qwen2MLP(config).float().eval().requires_grad_(False)
    middle_norm = (
        Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps).float().eval().requires_grad_(False)
    )
    final_norm = (
        Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps).float().eval().requires_grad_(False)
    )
    prefix = "model.layers.27."
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        mlp.load_state_dict(
            {name: handle.get_tensor(prefix + "mlp." + name).float() for name in mlp.state_dict()}
        )
        middle_norm.load_state_dict(
            dict(weight=handle.get_tensor(prefix + "post_attention_layernorm.weight").float())
        )
        final_norm.load_state_dict(dict(weight=handle.get_tensor("model.norm.weight").float()))
        wo = handle.get_tensor(prefix + "self_attn.o_proj.weight").float()
        head_name = (
            "lm_head.weight" if "lm_head.weight" in handle.keys() else "model.embed_tokens.weight"
        )
        head = handle.get_tensor(head_name).to(device=device, dtype=torch.float32)
    torch.set_num_threads(2)
    return SimpleNamespace(
        mlp=mlp.to(device),
        middle_norm=middle_norm.to(device),
        final_norm=final_norm.to(device),
        wo=wo.to(device),
        head=head,
        device=device,
        checkpoint=cp,
        layers=28,
    )


def cpu_weights(terminal):
    return dict(
        norm=array(terminal.middle_norm.weight).astype(float),
        eps=np.array(terminal.middle_norm.variance_epsilon),
        gate=array(terminal.mlp.gate_proj.weight).astype(float),
        up=array(terminal.mlp.up_proj.weight).astype(float),
        down=array(terminal.mlp.down_proj.weight).astype(float),
        final_norm=array(terminal.final_norm.weight).astype(float),
        final_eps=np.array(terminal.final_norm.variance_epsilon),
        head=array(terminal.head[[895, 830]]).astype(float),
        wo=array(terminal.wo).astype(float),
    )


def reference_weights(expected_checkpoint=None):
    """CPU-only auditor reads only required checkpoint slices, not a model."""
    spec = read(ROOT / "research/boolean_coupling/settings.json")["model"]
    folder = Path(
        snapshot_download(repo_id=spec["model"], revision=spec["revision"], local_files_only=True)
    )
    if expected_checkpoint is not None:
        actual = dict(
            revision=spec["revision"],
            config_sha256=sha(folder / "config.json"),
            checkpoint_sha256=sha(folder / "model.safetensors"),
        )
        assert actual == expected_checkpoint
    config = read(folder / "config.json")
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:

        def get(name):
            return handle.get_tensor(name).float().numpy().astype(float)

        prefix = "model.layers.27."
        head_name = (
            "lm_head.weight" if "lm_head.weight" in handle.keys() else "model.embed_tokens.weight"
        )
        sl = handle.get_slice(head_name)
        head = torch.cat([sl[i : i + 1] for i in [895, 830]]).float().numpy().astype(float)
        return dict(
            norm=get(prefix + "post_attention_layernorm.weight"),
            eps=np.array(config["rms_norm_eps"]),
            gate=get(prefix + "mlp.gate_proj.weight"),
            up=get(prefix + "mlp.up_proj.weight"),
            down=get(prefix + "mlp.down_proj.weight"),
            final_norm=get("model.norm.weight"),
            final_eps=np.array(config["rms_norm_eps"]),
            head=head,
            wo=get(prefix + "self_attn.o_proj.weight"),
        )


def generate(parent, selection, wo):
    dp = parent["p_native"].astype(float) - parent["p_false_query"].astype(float)
    values = np.repeat(parent["values"].astype(float), int(parent["groups"]), axis=0)
    deltas = {}
    for label, indices in selection["positions"].items():
        if indices:
            y = np.einsum("hrk,hkd->hrd", dp[..., indices], values[:, indices])
        else:
            y = np.zeros((*dp.shape[:-1], values.shape[-1]))
        deltas[label] = y.transpose(1, 0, 2).reshape(y.shape[1], -1) @ wo.T
    for label in ["random1", "random2"]:
        seed = int.from_bytes(
            hashlib.sha256((selection["key"] + "/" + label + "/298").encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        x = rng.normal(size=deltas["chosen"].shape)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        deltas[label] = x * np.linalg.norm(deltas["chosen"], axis=-1, keepdims=True)
    assert set(deltas) == set(GROUPS)
    return deltas


def run_case(terminal, parent, selection):
    """Only terminal row-wise operators execute; earlier native inputs are supplied."""
    device = terminal.device
    length, start = selection["length"], selection["position"]
    receiving = start + 1
    width = terminal.wo.shape[0]
    data, checks, counts = {}, [], Counter()
    full = torch.zeros((1, length, width), device=device, dtype=torch.float32)
    full[0, receiving:] = torch.tensor(parent["input"], device=device, dtype=torch.float32)
    base_writes = {}
    with torch.no_grad():
        for name, field in [
            ("native_reference", "head_native"),
            ("frozen_query_reference", "head_false_query"),
        ]:
            h = parent[field].transpose(1, 0, 2).reshape(length - receiving, width)
            padded = torch.zeros_like(full)
            padded[0, receiving:] = torch.tensor(h, device=device, dtype=torch.float32)
            base_writes[name] = F.linear(padded, terminal.wo)
            counts["full_shape_output_projections"] += 1
            data[name + "_write"] = array(base_writes[name][0, receiving:])
    deltas = generate(parent, selection, array(terminal.wo).astype(float))
    data.update({"delta_" + name: value for name, value in deltas.items()})

    def finish(name, write):
        with torch.no_grad():
            middle = full + write
            final = middle + terminal.mlp(terminal.middle_norm(middle))
            logits = F.linear(terminal.final_norm(final)[:, -1:], terminal.head)
            counts["terminal_MLPs"] += 1
            counts["full_vocabulary_heads"] += 1
            data[name + "_middle"] = array(middle[0, receiving:])
            data[name + "_write"] = array(write[0, receiving:])
            data[name + "_final"] = array(final[0, receiving:])
            data[name + "_scores"] = array(logits[0, -1, [895, 830]])

    finish("native_reference", base_writes["native_reference"])
    record(checks, "native_final_exact", data["native_reference_final"], parent["expected_final"])
    record(
        checks, "native_scores_exact", data["native_reference_scores"], parent["expected_scores"]
    )
    if not all(r["passes"] for r in checks):
        return data, checks, dict(counts), False
    finish("frozen_query_reference", base_writes["frozen_query_reference"])
    record(
        checks,
        "all_message_bridge",
        deltas["all"],
        data["native_reference_write"].astype(float)
        - data["frozen_query_reference_write"].astype(float),
        BOUNDS["message"],
    )
    for label in GROUPS:
        correction = torch.tensor(deltas[label], device=device, dtype=torch.float32)
        for operation, baseline, sign in [
            ("add", "frozen_query_reference", 1),
            ("remove", "native_reference", -1),
        ]:
            name = label + "_" + operation
            write = base_writes[baseline].clone()
            write[0, receiving:] += sign * correction
            record(
                checks,
                name + "/protected_padding",
                array(write[0, :receiving]),
                array(base_writes[baseline][0, :receiving]),
            )
            expected_write = np.add(
                data[baseline + "_write"],
                np.float32(sign) * deltas[label].astype(np.float32),
                dtype=np.float32,
            )
            record(checks, name + "/installation", array(write[0, receiving:]), expected_write)
            finish(name, write)
            record(
                checks,
                name + "/middle_installation",
                data[name + "_middle"],
                np.add(parent["input"], expected_write, dtype=np.float32),
            )
            if label == "zero":
                for field in ["_middle", "_final", "_scores"]:
                    record(
                        checks,
                        name + "/identity" + field,
                        data[name + field],
                        data[baseline + field],
                    )
            if label == "all":
                target = "native_reference" if operation == "add" else "frozen_query_reference"
                record(
                    checks,
                    name + "/final_calibration",
                    data[name + "_final"],
                    data[target + "_final"],
                    BOUNDS["final_max"],
                )
                record(
                    checks,
                    name + "/score_calibration",
                    data[name + "_scores"],
                    data[target + "_scores"],
                    BOUNDS["head"],
                )
    for label in ["random1", "random2"]:
        target = np.linalg.norm(deltas["chosen"], axis=-1)
        actual = np.linalg.norm(deltas[label].astype(np.float32).astype(float), axis=-1)
        record(
            checks,
            label + "/matched_norm",
            (actual - target) / np.maximum(target, 1e-30),
            np.zeros_like(target),
            BOUNDS["random_norm_relative"],
        )
    assert counts == dict(
        full_shape_output_projections=2, terminal_MLPs=20, full_vocabulary_heads=20
    )
    return data, checks, dict(counts), True
