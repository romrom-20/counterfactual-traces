"""Globally seal fresh all-gate forecasts before allowing native measurement."""

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
from .joint292 import ARMS, BOUNDS, PRIMARY, PUBLIC_SEAL, ROOT, SOURCES, Engine
from .run272 import stamp
from .run285 import model_signature, release, runtime
from .run286 import head_data


def validate_cases(cases, history, synthetic):
    assert len(cases) == (4 if synthetic else 32)
    assert len({c["key"] for c in cases}) == len(cases)
    seen = set()
    groups = {}
    for c in cases:
        f, t = c["false_ids"], c["true_ids"]
        assert len(f) == len(t) and 0 < c["position"] < len(f) - 1
        assert [j for j, (a, b) in enumerate(zip(f, t, strict=True)) if a != b] == [c["position"]]
        assert f[c["position"]] == 895 and t[c["position"]] == 830
        hs = [
            hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()
            for x in (f, t)
        ]
        assert hs == c["input_hashes"] and not set(hs) & (set(history) | seen)
        seen.update(hs)
        g = c["group"]
        groups.setdefault(g["block"], set()).add((g["schema"], g["reverse"], g["target"]))
    assert len(groups) == 4
    if not synthetic:
        expected = {(s, r, t) for s in ("generic", "descriptive") for r in (0, 1) for t in (0, 1)}
        assert all(x == expected for x in groups.values())


def binding(run, model=None):
    d = read(run / "design.json")
    assert d["experiment"] == 292 and d["arms"] == list(ARMS) and d["globally_pre_native"]
    assert d["primary"] == PRIMARY and d["bounds"] == BOUNDS and d["guards_MiB"] == [512, 384, 256]
    assert d["installed_sources"] == installed_sources() and set(d["source_hashes"]) == set(SOURCES)
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    assert d["checkpoint"] == (model_signature(model) if d["synthetic"] else locate(ROOT)[1])
    validate_cases(d["cases"], d["historical_hashes"], d["synthetic"])
    return d


def archive(folder, key, data, metadata):
    path = folder / (key + ".npz")
    assert not path.exists()
    np.savez_compressed(path, **data)
    save_json(path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), **metadata))
    return dict(
        key=key,
        arrays_sha256=sha(path),
        metadata_sha256=sha(path.with_suffix(".json")),
        bytes=path.stat().st_size,
    )


def entry(run, phase, item):
    p = run / phase / (item["key"] + ".npz")
    assert (
        sha(p) == item["arrays_sha256"] and sha(p.with_suffix(".json")) == item["metadata_sha256"]
    )
    assert p.stat().st_size == item["bytes"]
    with np.load(p) as z:
        return {n: z[n] for n in z.files}


def forecast(out, model=None, cases=None):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 512 * 1024**2
    synthetic = model is not None
    cases = read(ROOT / "research/boolean_coupling/cases292.json") if cases is None else cases
    history = (
        [] if synthetic else read(ROOT / "research/boolean_coupling/history292.json")["hashes"]
    )
    validate_cases(cases, history, synthetic)
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
    out.mkdir()
    for n in ("sources", "forecasts"):
        (out / n).mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=292,
            synthetic=synthetic,
            globally_pre_native=True,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=model_signature(model) if synthetic else locate(ROOT)[1],
            cases=cases,
            historical_hashes=history,
            arms=list(ARMS),
            primary=PRIMARY,
            bounds=BOUNDS,
            guards_MiB=[512, 384, 256],
            began=stamp(),
            ordinary_true=0,
            boundary="One full False observation, known edit, weights and own evolving states. All affected source/receiver SiLU predicted, no edited finite SiLU. Finite attention/norm/projections/products/head; no compact-cost or circuit claim.",
        ),
    )
    model = load_model(True) if model is None else model
    save_json(out / "runtime.json", runtime(model, synthetic))
    np.savez_compressed(out / "head.npz", **head_data(model))
    save_json(
        out / "head.json", dict(arrays_sha256=sha(out / "head.npz"), fixed_weight_rows=[895, 830])
    )
    entries, total = [], 0
    for c in cases:
        release(model)
        assert shutil.disk_usage(out).free >= 384 * 1024**2
        engine = Engine(model, c)
        f, fmeta = engine.run("F")
        data = {"F_" + n: v for n, v in f.items()}
        metadata = {"F": fmeta}
        for arm in ARMS:
            pack, meta = engine.run(arm)
            data.update({arm + "_" + n: v for n, v in pack.items()})
            zero, zmeta = engine.run(arm, zero=True)
            metadata[arm], metadata[arm + "0"] = meta, zmeta
            assert all(x["ordinary_true"] == 0 for x in (meta, zmeta))
            assert (
                meta["current_affected_finite_SiLU_coordinates"]
                == zmeta["current_affected_finite_SiLU_coordinates"]
                == 0
            )
        e = archive(
            out / "forecasts",
            c["key"],
            data,
            dict(measured=stamp(), ordinary_true=0, arms=metadata),
        )
        entries.append(e)
        total += e["bytes"]
        assert total <= 256 * 1024**2
        print(
            "292 forecast",
            len(entries),
            "/",
            len(cases),
            "numeric failures",
            sum(not x["passes"] for m in metadata.values() for x in m["checks"]),
            flush=True,
        )
        del engine, data, metadata, f, pack, zero
        release(model)
    save_json(
        out / "forecast-status.json",
        dict(
            status="complete",
            ordinary_true=0,
            cases=entries,
            raw_bytes=total,
            ended=stamp(),
            counts=dict(
                full_F=len(cases),
                full_forecast=5 * len(cases),
                full_Fzero=5 * len(cases),
                affected_current_finite_SiLU=0,
            ),
        ),
    )


def seal(run, model=None):
    d = binding(run, model)
    assert not (run / "native-started.json").exists()
    a, st = (read(run / n) for n in ("forecast-audit.json", "forecast-status.json"))
    assert a["status"] == "passed" and st["status"] == "complete"
    assert st["ordinary_true"] == a["ordinary_true"] == 0
    assert a["design_sha256"] == sha(run / "design.json") and a["status_sha256"] == sha(
        run / "forecast-status.json"
    )
    for n, h in a["csv_sha256"].items():
        assert sha(run / n) == h
    for e in st["cases"]:
        entry(run, "forecasts", e)
    save_json(
        run / "forecast-seal.json",
        dict(
            experiment=292,
            synthetic=d["synthetic"],
            ordinary_true=0,
            source_hashes=d["source_hashes"],
            cases=st["cases"],
            design_sha256=sha(run / "design.json"),
            status_sha256=sha(run / "forecast-status.json"),
            audit_sha256=sha(run / "forecast-audit.json"),
            csv_sha256=a["csv_sha256"],
            source_commit=d["source_commit"],
            sealed=stamp(),
        ),
    )


def native(run, public_commit=None, public_branch=None, model=None):
    d = binding(run, model)
    assert not (run / "native-started.json").exists() and not (run / "native").exists()
    s, a, st = (
        read(run / n) for n in ("forecast-seal.json", "forecast-audit.json", "forecast-status.json")
    )
    assert s["cases"] == st["cases"] and s["source_hashes"] == d["source_hashes"]
    assert (
        s["ordinary_true"] == a["ordinary_true"] == st["ordinary_true"] == 0
        and a["status"] == "passed"
    )
    for field, name in [
        ("design_sha256", "design.json"),
        ("status_sha256", "forecast-status.json"),
        ("audit_sha256", "forecast-audit.json"),
    ]:
        assert s[field] == sha(run / name)
    assert s["csv_sha256"] == a["csv_sha256"]
    for n, h in s["csv_sha256"].items():
        assert sha(run / n) == h
    for e in s["cases"]:
        entry(run, "forecasts", e)
    expected = sha(run / "forecast-seal.json")
    if d["synthetic"]:
        assert public_commit == "synthetic292:" + expected and public_branch == "synthetic292"
        verified = False
    else:
        assert model is None and public_commit and public_branch
        assert (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            == public_commit
        )
        assert (
            subprocess.check_output(
                ["git", "ls-remote", "origin", "refs/heads/" + public_branch], cwd=ROOT, text=True
            ).split()[0]
            == public_commit
        )
        assert (
            hashlib.sha256(
                subprocess.check_output(
                    ["git", "show", public_commit + ":" + PUBLIC_SEAL], cwd=ROOT
                )
            ).hexdigest()
            == expected
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", d["source_commit"], public_commit],
            cwd=ROOT,
            check=True,
        )
        verified = True
    save_json(
        run / "native-started.json",
        dict(
            began=stamp(),
            forecast_seal_sha256=expected,
            public_commit=public_commit,
            public_branch=public_branch,
            remote_head_verified=verified,
            ordinary_true_before=0,
            synthetic=d["synthetic"],
        ),
    )
    (run / "native").mkdir()
    model = load_model(True) if model is None else model
    entries, total = [], 0
    for c, e in zip(d["cases"], st["cases"], strict=True):
        release(model)
        assert shutil.disk_usage(run).free >= 384 * 1024**2
        fmeta = read(run / "forecasts" / (c["key"] + ".json"))["arms"]["F"]
        engine = Engine(model, c, allow_native=True)
        pack, meta = engine.run("T")
        assert meta["prefixes"] == fmeta["prefixes"] and not meta["editing_hooks_applied"]
        item = archive(
            run / "native", c["key"], pack, dict(measured=stamp(), ordinary_true=1, arm=meta)
        )
        entries.append(item)
        total += item["bytes"]
        assert total + st["raw_bytes"] <= 256 * 1024**2
        print("292 native", len(entries), "/", len(d["cases"]), flush=True)
        del engine, pack
    save_json(
        run / "native-status.json",
        dict(
            status="complete",
            ordinary_true=len(entries),
            cases=entries,
            raw_bytes=total,
            ended=stamp(),
            forecast_seal_sha256=expected,
            native_started_sha256=sha(run / "native-started.json"),
            counts=dict(full_true=len(entries)),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("forecast", "seal", "native"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--public-commit")
    p.add_argument("--public-branch")
    args = p.parse_args()
    try:
        if args.phase == "forecast":
            forecast(args.run.resolve())
        elif args.phase == "seal":
            seal(args.run.resolve())
        else:
            native(args.run.resolve(), args.public_commit, args.public_branch)
    except Exception as exc:
        if args.run.exists():
            name = args.run / ("failure-" + args.phase + ".json")
            if not name.exists():
                save_json(name, dict(time=stamp(), error=repr(exc)))
        raise
