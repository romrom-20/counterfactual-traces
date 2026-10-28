"""Frozen retrospective response-subspace producer; no new checkpoint True calls."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import load_model
from .joint292 import ROOT
from .run285 import model_signature, release, runtime
from .run286 import head_data
from .subspace293 import ARMS, BOUNDS, PRIMARY, SOURCES, Restricted, check


def selected(design):
    return [c for c in design["cases"] if c["group"]["reverse"] == 0 and c["group"]["target"] == 0]


def binding(run, model=None):
    d = read(run / "design.json")
    assert d["experiment"] == 293 and d["retrospective"] and d["ordinary_checkpoint_true"] == 0
    assert d["arms"] == list(ARMS) and d["bounds"] == BOUNDS and d["primary"] == PRIMARY
    assert d["guards_MiB"] == [512, 384, 256] and d["installed_sources"] == installed_sources()
    assert set(d["source_hashes"]) == set(SOURCES)
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    assert d["checkpoint"] == (model_signature(model) if d["synthetic"] else locate(ROOT)[1])
    return d


def archive(out, key, data, meta):
    p = out / "cases" / (key + ".npz")
    assert not p.exists()
    np.savez_compressed(p, **data)
    save_json(p.with_suffix(".json"), dict(key=key, arrays_sha256=sha(p), **meta))
    return dict(
        key=key,
        arrays_sha256=sha(p),
        metadata_sha256=sha(p.with_suffix(".json")),
        bytes=p.stat().st_size,
    )


def main(out, reference=None, model=None, cases=None, refs=None):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 512 * 1024**2
    synthetic = model is not None
    if not synthetic:
        parent = read(reference / "design.json")
        cases = selected(parent)
        assert len(cases) == 8
        fs, ns = read(reference / "forecast-status.json"), read(reference / "native-status.json")
        assert read(reference / "native-audit.json")["status"] == "passed"
        refs = {}
        for c in cases:
            k = c["key"]
            fe = next(e for e in fs["cases"] if e["key"] == k)
            te = next(e for e in ns["cases"] if e["key"] == k)
            paths = [reference / phase / (k + ".npz") for phase in ("forecasts", "native")]
            assert all(sha(p) == e["arrays_sha256"] for p, e in zip(paths, (fe, te), strict=True))
            with np.load(paths[0]) as f, np.load(paths[1]) as t:
                refs[k] = (
                    {n: f[n] for n in f.files},
                    {n: t[n] for n in t.files},
                    read(paths[0].with_suffix(".json")),
                )
    assert cases and refs
    out.mkdir()
    (out / "sources").mkdir()
    (out / "cases").mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    if not synthetic:
        for n, h in hashes.items():
            assert (
                hashlib.sha256(
                    subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
                ).hexdigest()
                == h
            )
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=293,
            retrospective=True,
            synthetic=synthetic,
            ordinary_checkpoint_true=0,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=model_signature(model) if synthetic else locate(ROOT)[1],
            cases=cases,
            arms=list(ARMS),
            primary=PRIMARY,
            bounds=BOUNDS,
            guards_MiB=[512, 384, 256],
            reference_hashes={}
            if synthetic
            else {
                n: sha(reference / n)
                for n in (
                    "design.json",
                    "forecast-status.json",
                    "native-status.json",
                    "native-audit.json",
                )
            },
            boundary="Eight old292 inputs; False plus known edit and own states.32 residual directions; dense computation remains. No semantic circuit or efficiency claim.",
        ),
    )
    model = load_model(True) if model is None else model
    save_json(out / "runtime.json", runtime(model, synthetic))
    np.savez_compressed(out / "head.npz", **head_data(model))
    save_json(out / "head.json", dict(arrays_sha256=sha(out / "head.npz")))
    entries = []
    total = 0
    for c in cases:
        release(model)
        assert shutil.disk_usage(out).free >= 384 * 1024**2
        f, t, oldmeta = refs[c["key"]]
        engine = Restricted(model, c)
        checks = []
        for n, v in engine.false.items():
            check(checks, "reference_F_" + n, v, f["F_" + n])
        assert engine.false_meta["parent_seals"] == oldmeta["arms"]["F"]["parent_seals"]
        data = {}
        metas = {}
        for n in ("final", "scores"):
            data["F_" + n] = f["F_" + n]
            data["T_" + n] = t[n]
        data["F_first_source"] = f["F_states"][0, 0]
        data["T_first_source"] = t["states"][0, 0]
        for j, site in enumerate(engine.sites):
            for n in ("g", "u", "phi", "slope"):
                data[f"False{site}_" + n] = f["F_witness_" + n][j]
        for arm in ARMS:
            pack, meta = engine.run(arm)
            if arm == "P":
                for n in ("final", "scores"):
                    check(checks, "reference_P_" + n, pack[n], f["P_" + n])
                check(checks, "reference_P_first_source", pack["first_source"], f["P_states"][0, 0])
                assert meta["engine"]["parent_seals"] == oldmeta["arms"]["P"]["parent_seals"]
            data.update({arm + "_" + n: v for n, v in pack.items()})
            metas[arm] = meta
            zero, zm = engine.run(arm, zero=True)
            metas[arm + "0"] = zm
            check(checks, arm + "_zero_final", zero["final"], f["F_final"])
            check(checks, arm + "_zero_scores", zero["scores"], f["F_scores"])
        meta = dict(
            checks=checks,
            arms=metas,
            ordinary_checkpoint_true=0,
            reference_forecast_hash=hashlib.sha256(
                json.dumps(oldmeta, sort_keys=True).encode()
            ).hexdigest(),
        )
        e = archive(out, c["key"], data, meta)
        entries.append(e)
        total += e["bytes"]
        assert total <= 256 * 1024**2
        print("293 completed", len(entries), "/", len(cases), "bytes", total, flush=True)
        del engine, data, metas
        release(model)
    save_json(
        out / "status.json",
        dict(status="complete", cases=entries, raw_bytes=total, ordinary_checkpoint_true=0),
    )
    from .audit293 import main as audit

    audit(out, model=model)
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.out.resolve(), a.reference.resolve())
    except Exception as e:
        if a.out.exists() and not (a.out / "failure.json").exists():
            save_json(a.out / "failure.json", dict(error=repr(e)))
        raise
