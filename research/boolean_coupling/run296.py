"""Forecast/seal/native phases; verify public global seal before ordinary True."""

import argparse
import hashlib
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate, read_head
from .core225b import load_model
from .fixtures296 import digest
from .query296 import BOUNDS, PRIMARY, ROOT, SOURCES, execute

PUBLIC_SEAL = "results/query-geometry-296/pre-native/forecast-seal.json"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_check(run):
    d = read(run / "design.json")
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h, n
    assert installed_sources() == d["installed_sources"]
    return d


def runtime(model):
    return dict(
        device=str(model.device),
        dtype=str(next(model.parameters()).dtype),
        torch=str(torch.__version__),
        training=model.training,
        attention=model.config._attn_implementation,
        revision=model.config._commit_hash,
        layers=len(model.model.layers),
    )


def save_case(
    run, phase, case, payload, witness, checks, counts, meta, started=None, parent_entry=None
):
    folder = run / ("forecasts" if phase == "forecast" else "observations")
    p = folder / (case["key"] + ".npz")
    assert not p.exists()
    np.savez_compressed(p, **payload)
    wh = None
    if witness:
        wp = run / ("witnesses-" + phase) / p.name
        np.savez_compressed(wp, **witness)
        wh = sha(wp)
    m = dict(
        key=case["key"],
        group=case["group"],
        arrays_sha256=sha(p),
        witness_sha256=wh,
        checks=checks,
        counts=counts,
        ordinary_true=0 if phase == "forecast" else 1,
        **meta,
    )
    if phase == "native":
        m.update(native_started_utc=started, forecast_arrays_sha256=parent_entry["arrays_sha256"])
    save_json(p.with_suffix(".json"), m)
    return dict(
        key=case["key"],
        arrays_sha256=sha(p),
        metadata_sha256=sha(p.with_suffix(".json")),
        bytes=p.stat().st_size,
        witness_sha256=wh,
        witness_bytes=0 if not witness else wp.stat().st_size,
    )


def forecast(run):
    assert not run.exists() and shutil.disk_usage(run.parent).free >= 1024 * 1024**2
    cases = read(ROOT / "research/boolean_coupling/cases296.json")
    catalog = set(read(ROOT / "research/boolean_coupling/history296.json")["hashes"])
    seen = set()
    for c in cases:
        a, b = c["false_ids"], c["true_ids"]
        j = c["position"]
        hs = [digest(a), digest(b)]
        assert hs == c["input_hashes"] and not set(hs) & (catalog | seen)
        seen.update(hs)
        assert (
            len(a) == len(b)
            and [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y] == [j]
            and a[j] == 895
            and b[j] == 830
        )
    assert len(cases) == 32 and len(seen) == 64
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n, h in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        ), n
    folder, checkpoint = locate(ROOT)
    run.mkdir()
    for n in ["forecasts", "observations", "sources", "witnesses-forecast", "witnesses-native"]:
        (run / n).mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, run / "sources" / Path(n).name)
    old = read_head(folder)
    np.savez_compressed(
        run / "head.npz", norm=old["final_norm"], eps=old["final_eps"], head=old["final_head"]
    )
    save_json(
        run / "design.json",
        dict(
            experiment=296,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=checkpoint,
            cases=cases,
            primary=PRIMARY,
            bounds=BOUNDS,
            guards_MiB=[1024, 512, 256, 128],
            head_sha256=sha(run / "head.npz"),
            created_utc=stamp(),
            ordinary_true=0,
            fresh_input_hashes=64,
            historical_input_hashes=len(catalog),
            boundary="One ordinary False query trace, known edit and own evolving finite states. Not compact, learned circuit or general QuestionB solution.",
        ),
    )
    model = load_model(True)
    save_json(run / "runtime.json", runtime(model))
    counts = Counter()
    entries = []
    total = 0
    for c in cases:
        assert shutil.disk_usage(run).free >= 512 * 1024**2
        payload, wit, ch, used, meta = execute(model, c, "forecast")
        counts.update(used)
        e = save_case(run, "forecast", c, payload, wit, ch, used, meta)
        entries.append(e)
        total += e["bytes"] + e["witness_bytes"]
        assert total <= 256 * 1024**2
        save_json(
            run / "forecast-progress.json",
            dict(cases=len(entries), counts=dict(counts), raw_bytes=total),
        )
        print("296 forecast", len(entries), "/32", flush=True)
    assert counts == dict(
        full_forwards=256, extra_full_QK_softmax_PV_each=6272, witness_extra_QK=120
    ), dict(counts)
    save_json(
        run / "forecast-status.json",
        dict(
            status="complete", cases=entries, counts=dict(counts), raw_bytes=total, ordinary_true=0
        ),
    )


def seal(run):
    source_check(run)
    assert not (run / "forecast-seal.json").exists() and not list((run / "observations").glob("*"))
    a = read(run / "pre-audit.json")
    assert a["status"] == "passed"
    save_json(
        run / "forecast-seal.json",
        dict(
            experiment=296,
            status="sealed",
            created_utc=stamp(),
            design_sha256=sha(run / "design.json"),
            status_sha256=sha(run / "forecast-status.json"),
            pre_audit_sha256=sha(run / "pre-audit.json"),
            ordinary_true=0,
            cases=read(run / "forecast-status.json")["cases"],
        ),
    )


def native(run, proof_path):
    d = source_check(run)
    proof = read(proof_path)
    seal = read(run / "forecast-seal.json")
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
    st = read(run / "forecast-status.json")
    for e in st["cases"]:
        p = run / "forecasts" / (e["key"] + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        if e["witness_sha256"]:
            assert sha(run / "witnesses-forecast" / p.name) == e["witness_sha256"]
    _, cp = locate(ROOT)
    assert cp == d["checkpoint"]
    save_json(run / "measurement-proof.json", proof)
    model = load_model(True)
    assert runtime(model) == read(run / "runtime.json")
    counts = Counter()
    entries = []
    total = 0
    for c, e in zip(d["cases"], st["cases"], strict=True):
        assert c["key"] == e["key"] and shutil.disk_usage(run).free >= 512 * 1024**2
        p = run / "forecasts" / (c["key"] + ".npz")
        m = read(p.with_suffix(".json"))
        with np.load(p) as z:
            f = dict(z)
        started = stamp()
        payload, wit, ch, used, meta = execute(model, c, "native", f, m)
        counts.update(used)
        ne = save_case(run, "native", c, payload, wit, ch, used, meta, started, e)
        entries.append(ne)
        total += ne["bytes"] + ne["witness_bytes"]
        assert total <= 128 * 1024**2
        save_json(
            run / "native-progress.json",
            dict(cases=len(entries), counts=dict(counts), raw_bytes=total),
        )
        print("296 native", len(entries), "/32", flush=True)
    assert counts == dict(full_forwards=64, extra_full_QK_softmax_PV_each=896, witness_extra_QK=48)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["forecast", "seal", "native"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--proof", type=Path)
    a = p.parse_args()
    run = a.out.resolve()
    try:
        if a.phase == "forecast":
            forecast(run)
        elif a.phase == "seal":
            seal(run)
        else:
            native(run, a.proof.resolve())
    except Exception as ex:
        if run.exists():
            save_json(run / ("failure-" + a.phase + ".json"), dict(error=repr(ex), stamp=stamp()))
        raise
