"""Posthoc numerical repetition; accepts original or reproduced297 parents."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit298 import qualify, scientific
from .cache272 import save_json
from .messages298 import ROOT, SOURCES, cpu_weights, load_terminal, read_parent, roles, run_case
from .publish285 import read, sha
from .routes297 import execute


def repetition_parent(folder, selection):
    design, summary = read(folder / "design.json"), read(folder / "summary.json")
    assert (
        design["experiment"] == 297
        and design["numerical_repetition"]
        and not design["tiny_fixture"]
        and summary["passed"]
    )
    case = next(c for c in design["cases"] if c["key"] == selection["key"])
    assert case["false_ids"] == selection["ids"] and case["position"] == selection["position"]
    meta = read(folder / (selection["key"] + ".json"))
    expected = next(e for e in summary["records"] if e["key"] == selection["key"])
    assert sha(folder / (selection["key"] + ".json")) == expected["metadata_sha256"]
    bindings = {e["path"]: e["sha256"] for e in meta["files"]}
    native = folder / (selection["key"] + "-native.npz")
    witness = folder / (selection["key"] + "-native-witness.npz")
    assert sha(native) == bindings[native.name] and sha(witness) == bindings[witness.name]
    with np.load(native) as data:
        result = dict(
            input=data["T_trace"][-2],
            expected_final=data["T_final"][1:],
            expected_scores=data["T_scores"],
        )
    with np.load(witness) as data:
        for name, field in dict(
            head_native="N_output",
            head_false_query="B_output",
            p_native="N_prob",
            p_false_query="B_prob",
            values="v",
            groups="groups",
        ).items():
            result[name] = data["T_27_" + field]
    return result


def fixtures():
    torch.manual_seed(1298)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=1024,
                hidden_size=16,
                intermediate_size=23,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                tie_word_embeddings=False,
                attention_dropout=0.0,
                _attn_implementation="eager",
            )
        )
        .float()
        .eval()
        .requires_grad_(False)
    )
    layer = model.model.layers[-1]
    terminal = SimpleNamespace(
        mlp=layer.mlp,
        middle_norm=layer.post_attention_layernorm,
        final_norm=model.model.norm,
        wo=layer.self_attn.o_proj.weight,
        head=model.lm_head.weight,
        device=model.device,
    )
    jobs = []
    rng = np.random.default_rng(1298)
    for i, length in enumerate([14, 17]):
        f = rng.integers(3, 400, length).tolist()
        s = length - 5
        f[1], f[3], f[4], f[6], f[s] = 997, 998, 999, 999, 895
        t = f.copy()
        t[s] = 830
        schema = "generic" if i == 0 else "descriptive"
        c = dict(
            key="io298-" + str(i),
            group=dict(block="tiny", schema=schema, reverse=0, target=0),
            position=s,
            false_ids=f,
            true_ids=t,
        )
        fd, _, fc, _, fm = execute(model, c, "forecast")
        fm["checks"] = fc
        nd, nw, _, _, _ = execute(model, c, "native", fd, fm)
        parent = dict(
            input=nd["T_trace"][-2],
            expected_final=nd["T_final"][1:],
            expected_scores=nd["T_scores"],
        )
        for name, field in dict(
            head_native="N_output",
            head_false_query="B_output",
            p_native="N_prob",
            p_false_query="B_prob",
            values="v",
            groups="groups",
        ).items():
            parent[name] = nw["T_2_" + field]
        selection = dict(
            key=c["key"],
            group=c["group"],
            position=s,
            length=length,
            positions=roles(f, s, schema, 999, {997, 998}),
        )
        jobs.append((selection, parent))
    return terminal, jobs


def validate_sources():
    original = read(ROOT / "results/context-messages-298/design.json")
    actual = {n: sha(ROOT / n) for n in SOURCES}
    assert original["source_hashes"] == actual, "Original frozen scientific sources changed"
    return actual


def main(out, parent, tiny=False):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 8 * 1024**3
    hashes = validate_sources()
    torch.set_num_threads(2)
    if tiny:
        terminal, jobs = fixtures()
        mode = "tiny_cpu"
        parent_hash = None
    else:
        assert parent is not None
        terminal = load_terminal()
        mode = "original297" if (parent / "native-status.json").exists() else "reproduced297"
        selections = read(ROOT / "research/boolean_coupling/parents298.json")["cases"]
        reader = read_parent if mode == "original297" else repetition_parent
        jobs = [(c, reader(parent, c)) for c in selections]
        parent_hash = sha(
            parent / ("native-status.json" if mode == "original297" else "summary.json")
        )
    weights = cpu_weights(terminal)
    out.mkdir()
    counts = Counter()
    records = []
    total = 0
    required = raw_failed = 0
    save_json(
        out / "design.json",
        dict(
            experiment=298,
            numerical_repetition=True,
            retrospective=True,
            tiny_fixture=tiny,
            parent_mode=mode,
            parent_status_sha256=parent_hash,
            scientific_source_hashes=hashes,
            boundary="Numerical repetition with supplied terminal inputs; no fresh prospective or learned-circuit evidence.",
        ),
    )
    for selection, inputs in jobs:
        assert shutil.disk_usage(out).free >= 8 * 1024**3
        data, producer, used, complete = run_case(terminal, inputs, selection)
        p = out / (selection["key"] + ".npz")
        np.savez_compressed(p, **data)
        counts.update(used)
        total += p.stat().st_size
        assert total <= 64 * 1024**2
        with np.load(p) as loaded:
            assert set(loaded.files) == set(data) and all(
                np.array_equal(loaded[n], data[n]) for n in data
            )
        assert complete, "Native calibration failed; saved partial numerical repetition"
        checks = []
        raw = []
        qualify(selection, inputs, data, producer, weights, checks, raw)
        effects, vectors, joint = scientific(selection, data)
        save_json(
            p.with_suffix(".json"),
            dict(
                key=selection["key"],
                arrays_sha256=sha(p),
                checks=checks,
                raw=raw,
                effects=effects,
                vectors=vectors,
                joint=joint,
                counts=used,
            ),
        )
        required += len(checks)
        raw_failed += sum(not r["passes"] for r in raw)
        records.append(
            dict(
                key=selection["key"],
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
            )
        )
        assert all(r["passes"] for r in checks), (
            "Qualification failed; preserve outputs and original failures"
        )
        print("298 numerical repetition", len(records), "/", len(jobs), flush=True)
    summary = dict(
        passed=True,
        experiment=298,
        numerical_repetition=True,
        retrospective=True,
        tiny_fixture=tiny,
        cases=len(jobs),
        counts=dict(counts),
        required_checks=required,
        original_raw_failures=raw_failed,
        raw_bytes=total,
        records=records,
        tiny_full_forwards=18 if tiny else 0,
        checkpoint_full_forwards=0,
        general_question_B_solved=False,
    )
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--tiny-fixture", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Loading is always offline.")
    args = parser.parse_args()
    main(
        args.out.resolve(),
        None if args.parent is None else args.parent.resolve(),
        args.tiny_fixture,
    )
