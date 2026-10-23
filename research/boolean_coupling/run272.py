"""Frozen retrospective cache/budget baseline on all32 old269 contexts."""

import argparse
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from .audit229 import sha
from .cache272 import (
    ROOT,
    SOURCES,
    array,
    budget,
    cache_seals,
    forward,
    installed_sources,
    prefix,
    save_json,
    timed,
)
from .checkpoint238 import locate
from .core225b import load_model


def read(path):
    import json
    return json.loads(path.read_text())


def runtime(model):
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        transformers=transformers.__version__, device="mps", dtype="float32",
        metric_dtype="cpu_float64", revision=model.config._commit_hash,
        attention=model.config._attn_implementation, training=model.training)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main(reference, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 800 * 1024**2
    cases = read(ROOT / "research/boolean_coupling/cases269.json")
    assert len(cases) == 32 and cases == read(reference / "design.json")["cases"]
    assert read(reference / "audit.json")["status"] == "passed"
    out.mkdir(parents=True)
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name in sources:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    _, checkpoint = locate(ROOT)
    save_json(out / "design.json", dict(experiment=272, retrospective=True, cases=cases,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=sources, installed_sources=installed_sources(), checkpoint=checkpoint,
        reference_hashes={n: sha(reference / n) for n in ("design.json", "forecast-status.json",
            "native-status.json", "audit.json", "runtime.json")}, began=stamp(), benchmark_repetitions=3,
        primary="P32/32 original native joint, and measured dense-MAC ratio<=.05 in32/32; separate strict raw numerical checks retained.",
        oracle_native_receiving_inputs=False, forecast_inputs="One full observed False KV cache, known equal-length edit, weights; exact finite affected-suffix recomputation."))
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    assert all(v == read(reference / "runtime.json")[k] for k, v in runtime(model).items()
               if k != "metric_dtype")
    forecasts = {r["key"]: r for r in read(reference / "forecast-status.json")["cases"]}
    natives = {r["key"]: r for r in read(reference / "native-status.json")["cases"]}
    entries, total_bytes = [], 0
    for index, case in enumerate(cases):
        assert shutil.disk_usage(out).free >= 768 * 1024**2
        key, start = case["key"], case["position"]
        length = len(case["false_ids"])
        assert length == len(case["true_ids"]) and case["false_ids"][:start] == case["true_ids"][:start]
        assert [(i, a, b) for i, (a, b) in enumerate(zip(case["false_ids"], case["true_ids"], strict=True)) if a != b] == [(start, 895, 830)]
        fids, tids = (torch.tensor([case[n + "_ids"]], device="mps") for n in ("false", "true"))
        fs, fscore, observed, fm = forward(model, fids, length, cache=True, capture_start=start)
        original_cache = cache_seals(observed, length)
        shared = cache_seals(observed, start)
        assert len(shared) == 28 and all(r["shape"][-2] == start for r in shared)
        values = dict(F_states=fs, F_scores=array(fscore))
        records = {"F": fm}
        for arm, stop, ids in (("Z", start, fids[:, start:]),
                               ("S", start + 1, tids[:, start + 1:]),
                               ("P", start, tids[:, start:])):
            past = prefix(observed, stop, model.config)
            before = cache_seals(past, stop)
            assert before == cache_seals(observed, stop)
            states, scores, _, meta = forward(model, ids, length, past=past, cache=True, capture_start=0)
            if arm == "S":
                states = np.concatenate((fs[:, :1], states), axis=1)
            values[arm + "_states"], values[arm + "_scores"] = states, array(scores)
            meta["copied_prefix"] = before
            records[arm] = meta
        assert cache_seals(observed, length) == original_cache
        ts, tscore, _, tm = forward(model, tids, length, capture_start=start)
        values.update(T_states=ts, T_scores=array(tscore))
        records["T"] = tm
        assert fm["prefix_states"] == tm["prefix_states"]
        for left, right in zip(fm["attention"], tm["attention"], strict=True):
            assert left["prefix_key"] == right["prefix_key"]
            assert left["prefix_value"] == right["prefix_value"]
        for arm in ("P", "Z"):
            for row, seal in zip(records[arm]["attention"], shared, strict=True):
                assert row["prefix_key"] == seal["key"] and row["prefix_value"] == seal["value"]
        reference_bindings = {}
        for folder, entry, arm in (("forecasts", forecasts[key], "F"), ("observations", natives[key], "T")):
            path = reference / folder / (key + ".npz")
            assert sha(path) == entry["arrays_sha256"]
            meta_hash = entry.get("metadata_sha256", entry.get("record_sha256"))
            assert sha(path.with_suffix(".json")) == meta_hash
            with np.load(path) as old:
                assert np.array_equal(values[arm + "_states"][-1], old[arm + "_final"])
                assert np.array_equal(values[arm + "_scores"], old[arm + "_scores"])
            reference_bindings[arm] = dict(arrays_sha256=sha(path), metadata_sha256=meta_hash)
        timings = []
        for repeat in range(3):
            order = ("P", "T") if (index + repeat) % 2 == 0 else ("T", "P")
            for arm in order:
                seconds, scores = (timed(model, tids[:, start:], length, observed, start) if arm == "P"
                                   else timed(model, tids, length))
                timings.append(dict(repetition=repeat, arm=arm, seconds=seconds, scores=scores.tolist(),
                                    order=list(order), clone_prefix_included=arm == "P"))
        assert cache_seals(observed, length) == original_cache
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **values)
        total_bytes += path.stat().st_size
        cache_bytes = sum(2 * int(np.prod(r["shape"])) * 4 for r in shared)
        save_json(path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), records=records,
            source_false_cache_sha256=original_cache, shared_prefix=shared, prefix_bytes=cache_bytes,
            reference=reference_bindings, budgets={a: budget(m) for a, m in records.items()},
            benchmark=timings, statuses={a: "complete" for a in ("F", "Z", "S", "P", "T")}))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        assert total_bytes <= 220 * 1024**2 and shutil.disk_usage(out).free >= 512 * 1024**2
        print("272", index + 1, "/32", case["group"]["schema"], case["group"]["reverse"], "complete", flush=True)
    save_json(out / "status.json", dict(status="complete", cases=entries, raw_bytes=total_bytes,
        profiled_full_forwards=64, profiled_suffix_forwards=96,
        benchmark_full_forwards=96, benchmark_suffix_forwards=96, ended=stamp()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference.resolve(), args.out.resolve())
    except Exception as exc:
        if args.out.exists() and not (args.out / "failure.json").exists():
            save_json(args.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
