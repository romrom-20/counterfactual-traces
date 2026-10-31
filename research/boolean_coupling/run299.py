"""Immutable forecast, global public seal, then native query-generator measurement."""

import argparse
import hashlib
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .core225b import load_model
from .fixtures296 import digest
from .generator299 import BOUNDS, EFFECT, PRIMARY, ROOT, SOURCES, execute
from .run296 import runtime, save_case, source_check, stamp

PUBLIC_SEAL = "results/query-generators-299/pre-native/forecast-seal.json"


def free(run):
    assert shutil.disk_usage(run).free >= 8 * 1024**3, "8GiB reserve; preserve stop"


def forecast(run):
    assert not run.exists()
    free(run.parent)
    cases = read(ROOT / "research/boolean_coupling/cases299.json")
    catalog = set(read(ROOT / "research/boolean_coupling/history299.json")["hashes"])
    seen = set()
    for c in cases:
        a, b, j = c["false_ids"], c["true_ids"], c["position"]
        hs = [digest(a), digest(b)]
        assert hs == c["input_hashes"] and not set(hs) & (catalog | seen)
        seen.update(hs)
        assert len(a) == len(b) and [
            i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y
        ] == [j]
        assert a[j] == 895 and b[j] == 830
    assert len(cases) == 32 and len(seen) == 64
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    assert len(hashes) == len(SOURCES) and len({Path(n).name for n in SOURCES}) == len(SOURCES)
    for n, h in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        ), n
    cpu = read(ROOT.parent / "299-cpu/qualification.json")
    assert cpu["passed"] and cpu["source_hashes"] == hashes
    folder, checkpoint = locate(ROOT)
    run.mkdir()
    for name in ["forecasts", "observations", "sources", "witnesses-forecast", "witnesses-native"]:
        (run / name).mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, run / "sources" / Path(n).name)
    old = read_head(folder)
    np.savez_compressed(
        run / "head.npz", norm=old["final_norm"], eps=old["final_eps"], head=old["final_head"]
    )
    config = read(folder / "config.json")
    indices = sorted({1, config["num_hidden_layers"] // 2, config["num_hidden_layers"] - 1})
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        maps = {}
        for j in indices:
            for n, weight in [
                ("output", "self_attn.o_proj.weight"),
                ("query", "self_attn.q_proj.weight"),
                ("bias", "self_attn.q_proj.bias"),
                ("norm", "input_layernorm.weight"),
            ]:
                maps[f"{j}_{n}"] = (
                    handle.get_tensor(f"model.layers.{j}.{weight}").to(torch.float32).numpy()
                )
        maps["eps"] = np.array(config["rms_norm_eps"])
    np.savez_compressed(run / "query-weights.npz", **maps)
    assert (run / "query-weights.npz").stat().st_size <= 64 * 1024**2
    save_json(
        run / "design.json",
        dict(
            experiment=299,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=checkpoint,
            cases=cases,
            primary=PRIMARY,
            effect=EFFECT,
            bounds=BOUNDS,
            guards_MiB=dict(free=8192, forecast=512, native=256, maps=64),
            head_sha256=sha(run / "head.npz"),
            query_weights_sha256=sha(run / "query-weights.npz"),
            created_utc=stamp(),
            ordinary_true=0,
            fresh_input_hashes=64,
            historical_input_hashes=len(catalog),
            cpu_qualification_sha256=sha(ROOT.parent / "299-cpu/qualification.json"),
            boundary="One False pre-MLP and MLP-write trace; known edit; query-only operand interventions with other own-state operations live. Expensive, not a compact predictor or learned circuit.",
        ),
    )
    model = load_model(True)
    save_json(run / "runtime.json", runtime(model))
    counts, entries, total = Counter(), [], 0
    for c in cases:
        free(run)
        payload, wit, checks, used, meta = execute(model, c, "forecast")
        counts.update(used)
        entry = save_case(run, "forecast", c, payload, wit, checks, used, meta)
        entries.append(entry)
        total += entry["bytes"] + entry["witness_bytes"]
        assert total <= 512 * 1024**2
        save_json(
            run / f"forecast-progress-{len(entries):04}.json",
            dict(cases=len(entries), counts=dict(counts), raw_bytes=total),
        )
        print("299 forecast", len(entries), "/32", flush=True)
    assert counts == dict(
        full_forwards=224,
        extra_full_RMSNorm=5400,
        witness_extra_full_query_projection=288,
        witness_extra_full_QK=288,
        witness_extra_full_softmax=288,
        witness_extra_full_PV=288,
    ), dict(counts)
    save_json(
        run / "forecast-status.json",
        dict(
            status="complete", cases=entries, counts=dict(counts), raw_bytes=total, ordinary_true=0
        ),
    )


def seal(run):
    source_check(run)
    assert not list((run / "observations").glob("*"))
    audit = read(run / "pre-audit.json")
    assert audit["status"] == "passed"
    save_json(
        run / "forecast-seal.json",
        dict(
            experiment=299,
            status="sealed",
            created_utc=stamp(),
            ordinary_true=0,
            design_sha256=sha(run / "design.json"),
            status_sha256=sha(run / "forecast-status.json"),
            pre_audit_sha256=sha(run / "pre-audit.json"),
            cases=read(run / "forecast-status.json")["cases"],
        ),
    )


def native(run, proof_path):
    d, proof, seal = source_check(run), read(proof_path), read(run / "forecast-seal.json")
    assert not (run / "native-status.json").exists() and not list((run / "observations").glob("*"))
    assert (
        proof["seal_sha256"] == sha(run / "forecast-seal.json")
        and proof["accepted_utc"] > seal["created_utc"]
    )
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", proof["ref"]], cwd=ROOT, text=True
    ).split()[0]
    assert remote == proof["remote_head"] == proof["commit"]
    assert (
        subprocess.check_output(["git", "show", proof["commit"] + ":" + PUBLIC_SEAL], cwd=ROOT)
        == (run / "forecast-seal.json").read_bytes()
    )
    for n, h in [
        ("design.json", seal["design_sha256"]),
        ("forecast-status.json", seal["status_sha256"]),
        ("pre-audit.json", seal["pre_audit_sha256"]),
    ]:
        assert sha(run / n) == h
    status = read(run / "forecast-status.json")
    for e in status["cases"]:
        p = run / "forecasts" / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        if e["witness_sha256"]:
            assert sha(run / "witnesses-forecast" / p.name) == e["witness_sha256"]
    _, checkpoint = locate(ROOT)
    assert checkpoint == d["checkpoint"]
    assert (
        sha(run / "head.npz") == d["head_sha256"]
        and sha(run / "query-weights.npz") == d["query_weights_sha256"]
    )
    save_json(run / "measurement-proof.json", proof)
    model = load_model(True)
    assert runtime(model) == read(run / "runtime.json")
    counts, entries, total = Counter(), [], 0
    for c, e in zip(d["cases"], status["cases"], strict=True):
        free(run)
        assert c["key"] == e["key"]
        p = run / "forecasts" / (c["key"] + ".npz")
        with np.load(p) as z:
            forecast_data = dict(z)
        started = stamp()
        payload, wit, checks, used, meta = execute(
            model, c, "native", forecast_data, read(p.with_suffix(".json"))
        )
        counts.update(used)
        entry = save_case(run, "native", c, payload, wit, checks, used, meta, started, e)
        entries.append(entry)
        total += entry["bytes"] + entry["witness_bytes"]
        assert total <= 256 * 1024**2
        save_json(
            run / f"native-progress-{len(entries):04}.json",
            dict(cases=len(entries), counts=dict(counts), raw_bytes=total),
        )
        print("299 native", len(entries), "/32", flush=True)
    assert counts == dict(
        full_forwards=64,
        extra_full_RMSNorm=1008,
        witness_extra_full_query_projection=144,
        witness_extra_full_QK=144,
        witness_extra_full_softmax=144,
        witness_extra_full_PV=144,
    ), dict(counts)
    save_json(
        run / "native-status.json",
        dict(
            status="complete",
            cases=entries,
            counts=dict(counts),
            raw_bytes=total,
            ordinary_true=32,
            forecast_seal_sha256=sha(run / "forecast-seal.json"),
            proof_sha256=sha(run / "measurement-proof.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["forecast", "seal", "native"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    run = args.out.resolve()
    try:
        if args.phase == "forecast":
            forecast(run)
        elif args.phase == "seal":
            seal(run)
        else:
            native(run, args.proof.resolve())
    except Exception as error:
        if run.exists():
            save_json(
                run / ("failure-" + args.phase + ".json"), dict(error=repr(error), stamp=stamp())
            )
        raise
