"""Posthoc numerical repetition of299, without old raw arrays or GitHub writes.

This executes each case's native reference after its forecasts, rather than
recreating the original global public pre-native seal. It is not fresh evidence.
"""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit229 import sha
from .audit272 import read
from .audit299 import audit_bundle, local, native_sources, scientific
from .cache272 import save_json
from .checkpoint238 import locate, read_head
from .core225b import array, load_model
from .generator299 import ROOT, SOURCES, execute, sites


def tiny():
    torch.manual_seed(1299)
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
    )
    model.requires_grad_(False)
    cases = []
    rng = np.random.default_rng(1299)
    for i, length in enumerate([12, 17]):
        f = rng.integers(3, 400, length).tolist()
        s = length - 5
        f[s] = 895
        t = f.copy()
        t[s] = 830
        cases.append(
            dict(
                key="io299-" + str(i),
                group=dict(block="tiny", schema="tiny", reverse=0, target=0),
                position=s,
                false_ids=f,
                true_ids=t,
            )
        )
    head = dict(
        norm=array(model.model.norm.weight).astype(float),
        eps=np.array(model.config.rms_norm_eps),
        head=array(model.lm_head.weight)[[895, 830]].astype(float),
    )
    return model, cases, head


def main(out, fixture=False):
    assert not out.exists(), "New output directory required; no overwrite or automatic pruning"
    assert shutil.disk_usage(out.parent).free >= 8 * 1024**3
    start = time.perf_counter()
    torch.set_num_threads(2)
    source = read(ROOT / "results/query-generators-299/pre-native/design.json")
    assert source["source_hashes"] == {n: sha(ROOT / n) for n in SOURCES}
    if fixture:
        model, cases, head = tiny()
    else:
        cases = read(ROOT / "research/boolean_coupling/cases299.json")
        folder, checkpoint = locate(ROOT)
        w = read_head(folder)
        head = dict(norm=w["final_norm"], eps=w["final_eps"], head=w["final_head"])
        model = load_model(True)
        assert checkpoint == source["checkpoint"]
    maps = {"eps": np.array(model.config.rms_norm_eps)}
    for j in sites(model):
        layer = model.model.layers[j]
        for n, t in [
            ("query", layer.self_attn.q_proj.weight),
            ("bias", layer.self_attn.q_proj.bias),
            ("norm", layer.input_layernorm.weight),
            ("output", layer.self_attn.o_proj.weight),
        ]:
            maps[f"{j}_{n}"] = array(t)
    out.mkdir()
    save_json(
        out / "design.json",
        dict(
            experiment=299,
            numerical_repetition=True,
            globally_pre_native=False,
            tiny_fixture=fixture,
            cases=cases,
            frozen_source_hashes=source["source_hashes"],
            checkpoint_loaded=not fixture,
            checkpoint=None if fixture else checkpoint,
            raw_cap_bytes=768 * 1024**2,
        ),
    )
    counts, records, total, required, raw_failed = Counter(), [], 0, 0, 0
    for case in cases:
        assert shutil.disk_usage(out).free >= 8 * 1024**3
        f, fw, fc, n, fm = execute(model, case, "forecast")
        fm["checks"] = fc
        t, tw, tc, nn, tm = execute(model, case, "native", f, fm)
        tm["checks"] = tc
        checks, raw = [], []
        audit_bundle(case, f, fw, fm, head, "forecast", checks, raw, f, maps)
        audit_bundle(case, t, tw, tm, head, "native", checks, raw, f, maps)
        native_sources(case, fm, tm, checks)
        local(case, tw, maps, checks)
        counts.update(n)
        counts.update(nn)
        required += len(checks)
        raw_failed += sum(not r["passes"] for r in raw)
        files = []
        for label, payload in [
            ("forecast", f),
            ("forecast-witness", fw),
            ("native", t),
            ("native-witness", tw),
        ]:
            p = out / (case["key"] + "-" + label + ".npz")
            np.savez_compressed(p, **payload)
            total += p.stat().st_size
            with np.load(p) as loaded:
                assert set(loaded.files) == set(payload) and all(
                    np.array_equal(loaded[k], payload[k]) for k in payload
                )
            files.append(dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size))
        assert total <= 768 * 1024**2
        rows, _, _ = scientific(case, f, t)
        record = dict(
            key=case["key"], files=files, checks=checks, raw_diagnostics=raw, decisions=rows
        )
        save_json(out / (case["key"] + ".json"), record)
        records.append(dict(key=case["key"], metadata_sha256=sha(out / (case["key"] + ".json"))))
        if not all(r["passes"] for r in checks):
            save_json(
                out / "numerical-failure.json",
                dict(key=case["key"], failed=[r for r in checks if not r["passes"]]),
            )
            raise AssertionError("Numerical qualification failed; preserve all outputs")
        print("299 numerical repetition", len(records), "/", len(cases), flush=True)
    summary = dict(
        passed=True,
        experiment=299,
        numerical_repetition=True,
        globally_pre_native=False,
        tiny_fixture=fixture,
        checkpoint_loaded=not fixture,
        cases=len(cases),
        counts=dict(counts),
        required_checks=required,
        failed_checks=0,
        raw_diagnostic_failures=raw_failed,
        raw_bytes=total,
        records=records,
        seconds=time.perf_counter() - start,
        general_question_B_solved=False,
    )
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="All loading is always offline; accepted explicitly for clarity.",
    )
    parser.add_argument(
        "--tiny-fixture",
        action="store_true",
        help="Exercise save/reload/qualification on a small random CPU model.",
    )
    args = parser.parse_args()
    main(args.out.resolve(), args.tiny_fixture)
