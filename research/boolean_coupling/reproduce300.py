"""Numerical repetition of300 with original parents or deliberate parent regeneration."""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .audit229 import sha
from .audit272 import read
from .audit300 import bundle, scientific
from .cache272 import save_json
from .core225b import array, load_model
from .generator299 import execute as generate
from .interaction300 import ROOT, SOURCES, execute


def tiny():
    torch.manual_seed(1300)
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
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.mul_(10)
            layer.self_attn.k_proj.weight.mul_(10)
            layer.mlp.down_proj.weight.mul_(3)
    cases = []
    rng = np.random.default_rng(1300)
    for i, length in enumerate((12, 17)):
        f = rng.integers(3, 400, length).tolist()
        pos = length - 5
        f[pos] = 1
        t = f.copy()
        t[pos] = 2
        cases.append(
            dict(
                key="io300-" + str(i),
                position=pos,
                false_ids=f,
                true_ids=t,
                group=dict(block="tiny", schema="tiny", reverse=0, target=0),
            )
        )
    return model, cases


def main(out, parent, bootstrap, fixture):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 8 * 1024**3
    source = read(ROOT / "results/query-interaction-300/design.json")
    assert source["source_hashes"] == {n: sha(ROOT / n) for n in SOURCES}
    if fixture:
        model, cases = tiny()
    else:
        model = load_model(True)
        cases = [c["case"] for c in source["parents"]["cases"]]
        assert parent is not None or bootstrap, (
            "Supply original299 parent directory or --bootstrap-parents"
        )
    j = len(model.model.layers) // 2
    weights = dict(
        output=array(model.model.layers[j].self_attn.o_proj.weight),
        norm=array(model.model.norm.weight),
        eps=np.array(model.config.rms_norm_eps),
        head=array(model.lm_head.weight)[[895, 830]],
    )
    out.mkdir()
    save_json(
        out / "design.json",
        dict(
            experiment=300,
            numerical_repetition=True,
            retrospective=True,
            native_parents_supplied=True,
            tiny_fixture=fixture,
            bootstrap_parents=bootstrap,
            checkpoint_loaded=not fixture,
            cases=cases,
            source_hashes=source["source_hashes"],
            raw_cap_bytes=256 * 1024**2,
        ),
    )
    counts, records, total, checks_total = Counter(), [], 0, 0
    for index, case in enumerate(cases):
        assert shutil.disk_usage(out).free >= 8 * 1024**3
        key = case["key"]
        if fixture or bootstrap:
            f, _, fc, used, fm = generate(model, case, "forecast")
            fm["checks"] = fc
            reference, p, _, nused, pm = generate(model, case, "native", f, fm)
            counts["parent_regeneration_forwards"] += used["full_forwards"] + nused["full_forwards"]
            del f
            # Unchanged299 native/witness serialization must match the original
            # parent hashes in checkpoint mode. No inferred new qualification.
            for name, values in (("native", reference), ("witness", p)):
                path = out / (key + "-parent-" + name + ".npz")
                np.savez_compressed(path, **values)
                total += path.stat().st_size
                if not fixture:
                    item = source["parents"]["cases"][index]
                    assert (
                        sha(path)
                        == item["native_arrays_sha256" if name == "native" else "witness_sha256"]
                    ), (
                        "Parent regeneration differs from original sealed bytes; preserve this attempt"
                    )
        else:
            item = source["parents"]["cases"][index]
            pp = parent / "witnesses-native" / (key + ".npz")
            rp = parent / "observations" / (key + ".npz")
            assert (
                sha(pp) == item["witness_sha256"]
                and sha(rp) == item["native_arrays_sha256"]
                and sha(rp.with_suffix(".json")) == item["native_metadata_sha256"]
            )
            with np.load(pp) as z:
                p = dict(z)
            with np.load(rp) as z:
                reference = dict(z)
            pm = read(rp.with_suffix(".json"))
        data, ck, used, meta = execute(model, case, p, reference, pm)
        meta["checks"] = ck
        counts.update(used)
        checks, raw = [], []
        bundle(case, data, p, reference, meta, weights, checks, raw)
        checks_total += len(checks)
        path = out / (key + ".npz")
        np.savez_compressed(path, **data)
        total += path.stat().st_size
        with np.load(path) as z:
            assert all(np.array_equal(z[n], v) for n, v in data.items())
        assert total <= 256 * 1024**2
        rows, _, _ = scientific(case, data)
        record = {
            **meta,
            "key": key,
            "arrays_sha256": sha(path),
            "checks": checks,
            "raw": raw,
            "decisions": rows,
            "counts": used,
        }
        # meta contains the original installation checks; retain those separately.
        record["checks"] = checks
        save_json(out / (key + ".json"), record)
        assert all(r["passes"] for r in checks), "Required check failed; retain outputs"
        records.append(dict(key=key, metadata_sha256=sha(out / (key + ".json"))))
    summary = dict(
        passed=True,
        experiment=300,
        numerical_repetition=True,
        retrospective=True,
        native_parents_supplied=True,
        tiny_fixture=fixture,
        checkpoint_loaded=not fixture,
        counts=dict(counts),
        checks=checks_total,
        raw_bytes=total,
        total_full_forwards=counts["full_forwards"] + counts["parent_regeneration_forwards"],
        cases=len(cases),
        records=records,
        general_question_B_solved=False,
    )
    save_json(out / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--bootstrap-parents", action="store_true")
    parser.add_argument("--tiny-fixture", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Loading is always offline.")
    args = parser.parse_args()
    main(
        args.out.resolve(),
        args.parent.resolve() if args.parent else None,
        args.bootstrap_parents,
        args.tiny_fixture,
    )
