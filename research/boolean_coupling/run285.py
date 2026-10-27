"""Globally sealed structural forecasts followed by gated native measurement."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import transformers

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .composition280 import digest
from .core225b import array, load_model
from .run272 import stamp
from .source279 import derivatives
from .structure285 import ARMS, ROOT, SOURCES, Engine

PUBLIC_SEAL = "results/native-structure-285/pre-native/forecast-seal.json"
BASE_FIELDS = (
    "before",
    "mid",
    "gate",
    "up",
    "phi",
    "raw_write",
    "write",
    "source",
    "final",
    "scores",
    "norm",
    "z",
    "attwrite",
)
ATTENTION_FIELDS = ("q", "K", "V", "full_k", "full_v", "p", "y", "mask", "scaling")


def group_key(group, fields=("schema", "reverse", "target", "assignment")):
    return json.dumps({name: group[name] for name in fields}, sort_keys=True)


def load_bank(reference, config):
    """Read only sealed historical FORECAST files, never native observations."""
    assert config["experiment"] == 285 and config["reference_experiment"] == 279
    assert config["match_fields"] == ["schema", "reverse", "target", "assignment"]
    assert set(config["reference_hashes"]) == {
        "design.json",
        "forecast-status.json",
        "forecast-audit.json",
        "forecast-seal.json",
    }
    for name, expected in config["reference_hashes"].items():
        assert sha(reference / name) == expected, name
    design, status, audit, seal = (
        read(reference / name)
        for name in (
            "design.json",
            "forecast-status.json",
            "forecast-audit.json",
            "forecast-seal.json",
        )
    )
    assert design["experiment"] == 279
    assert status["status"] == "complete" and audit["status"] == "passed"
    assert status["ordinary_true"] == audit["ordinary_true"] == seal["ordinary_true"] == 0
    assert seal["cases"] == status["cases"]
    cases = {case["key"]: case for case in design["cases"]}
    entries = {entry["key"]: entry for entry in status["cases"]}
    bank, bindings = {}, []
    assert len(config["groups"]) == 8
    for item in config["groups"]:
        group = item["group"]
        assert group["assignment"] == 0
        key = group_key(group)
        assert key not in bank and len(item["donors"]) == 2
        blocks, deltas = set(), []
        for entry in item["donors"]:
            donor = entry["key"]
            assert entry == entries[donor]
            assert group_key(cases[donor]["group"]) == key
            blocks.add(cases[donor]["group"]["block"])
            path = reference / "forecasts" / (donor + ".npz")
            assert sha(path) == entry["arrays_sha256"]
            assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            meta = read(path.with_suffix(".json"))
            assert meta["arrays_sha256"] == entry["arrays_sha256"] and meta["ordinary_true"] == 0
            with np.load(path) as saved:
                assert not any(n.startswith(("T_", "native_")) for n in saved.files)
                delta = saved["writes_C"].astype(float) - saved["false_write"].astype(float)
            assert delta.ndim == 1 and np.isfinite(delta).all()
            deltas.append(delta)
        assert len(blocks) == 2
        mean = np.mean(np.stack(deltas), axis=0, dtype=np.float64)
        assert np.linalg.norm(mean) > 0
        bank[key] = mean
        bindings.append(
            dict(
                group=group,
                donor_keys=[r["key"] for r in item["donors"]],
                delta_sha256=digest(mean),
            )
        )
    return bank, bindings


def validate_cases(cases, history):
    assert len(cases) == 32 and len({c["key"] for c in cases}) == 32
    seen, layouts = set(), {}
    for case in cases:
        f, t, start = case["false_ids"], case["true_ids"], case["position"]
        assert len(f) == len(t) and 0 < start < len(f) - 1
        assert [i for i, (a, b) in enumerate(zip(f, t, strict=True)) if a != b] == [start]
        assert f[start] == 895 and t[start] == 830
        hashes = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in (f, t)
        ]
        assert hashes == case["input_hashes"] and not set(hashes) & (set(history) | seen)
        seen.update(hashes)
        g = case["group"]
        assert (
            set(g) == {"assignment", "block", "reverse", "schema", "target"}
            and g["assignment"] == 0
        )
        index = (g["schema"], g["reverse"], g["target"])
        assert index not in layouts.setdefault(g["block"], set())
        layouts[g["block"]].add(index)
    expected = {(s, r, t) for s in ("generic", "descriptive") for r in (0, 1) for t in (0, 1)}
    assert len(layouts) == 4 and all(rows == expected for rows in layouts.values())


def model_signature(model):
    return dict(
        synthetic=True,
        hidden=model.config.hidden_size,
        neurons=model.config.intermediate_size,
        layers=len(model.model.layers),
        weights={n: digest(p.detach().cpu().numpy()) for n, p in model.named_parameters()},
    )


def runtime(model, synthetic):
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        transformers=transformers.__version__,
        device=str(model.device),
        dtype=str(model.dtype),
        metric_dtype="cpu_float64",
        synthetic=synthetic,
        revision=getattr(model.config, "_commit_hash", None),
        attention=model.config._attn_implementation,
        training=model.training,
    )


def release(model):
    if model.device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
        torch.mps.synchronize()


def bindings(run, reference, model=None):
    design = read(run / "design.json")
    assert design["experiment"] == 285 and design["globally_pre_native"]
    assert tuple(design["arms"]) == ARMS and design["installed_sources"] == installed_sources()
    for name, expected in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == expected, name
    assert design["bank_config_sha256"] == sha(run / "bank-config.json")
    config = read(run / "bank-config.json")
    assert config["reference_hashes"] == design["reference_hashes"]
    bank, bank_bindings = load_bank(reference, config)
    assert bank_bindings == design["bank_bindings"]
    validate_cases(design["cases"], design["historical_input_hashes"])
    if design["synthetic"]:
        assert model is not None and model.device.type == "cpu"
        assert design["checkpoint"] == model_signature(model)
    else:
        assert model is None or model.device.type == "mps"
        assert design["checkpoint"] == locate(ROOT)[1]
        assert sha(ROOT / "research/boolean_coupling/bank285.json") == design["bank_config_sha256"]
    return design, bank


def forecast(reference, run, model=None, cases=None, bankconfig=None):
    assert not run.exists() and shutil.disk_usage(run.parent).free >= 1024 * 1024**2
    synthetic = model is not None
    if synthetic:
        assert model.device.type == "cpu" and cases is not None and bankconfig is not None
        history = {"hashes": []}
    else:
        assert cases is None and bankconfig is None
        cases = read(ROOT / "research/boolean_coupling/cases285.json")
        bankconfig = read(ROOT / "research/boolean_coupling/bank285.json")
        history = read(ROOT / "research/boolean_coupling/history285.json")
    validate_cases(cases, history["hashes"])
    bank, bank_bindings = load_bank(reference, bankconfig)
    run.mkdir()
    for folder in ("sources", "forecasts"):
        (run / folder).mkdir()
    save_json(run / "bank-config.json", bankconfig)
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        shutil.copy2(ROOT / name, run / "sources" / Path(name).name)
    save_json(
        run / "design.json",
        dict(
            experiment=285,
            globally_pre_native=True,
            synthetic=synthetic,
            cases=cases,
            historical_input_hashes=history["hashes"],
            arms=ARMS,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=hashes,
            installed_sources=installed_sources(),
            checkpoint=model_signature(model) if synthetic else locate(ROOT)[1],
            began=stamp(),
            bank_config_sha256=sha(run / "bank-config.json"),
            reference_hashes=bankconfig["reference_hashes"],
            bank_bindings=bank_bindings,
            storage=dict(initial_free_MiB=1024, per_case_free_MiB=512, combined_raw_cap_MiB=96),
            primary="H1 C native first-source write10%/.995 plus receiving/verdict5%/.99/.1nat; H2 C-M local and actual receiving/verdict repair25%/.95 with .02nat materiality; each >=24/32 and>=6/8 each layout.",
            boundary="Four fresh causal-prefix layouts in related synthetic task; first-source SiLU only approximated, all later operations finite. Frozen forecast-only donor bank, M uses own C forecast norm. No circuit, efficiency, novelty priority or complete Question B claim.",
        ),
    )
    model = load_model(True) if model is None else model
    release(model)
    save_json(run / "runtime.json", runtime(model, synthetic))
    entries, total = [], 0
    for case in cases:
        release(model)
        assert shutil.disk_usage(run).free >= 512 * 1024**2
        donor = bank[group_key(case["group"])]
        engine, values, metadata = Engine(model, case, donor), {}, {}
        contexts = None
        for arm in ARMS:
            with engine:
                packed, meta, context = engine.forward(arm)
            release(model)
            if arm != "F":
                for name in ("prefix_hashes", "K_prefix", "V_prefix"):
                    assert meta[name] == metadata["F"][name]
                assert meta["false_primal_checks"] == 1
                assert meta["current_source_finite_sites"] == len(model.model.layers) - 1
                assert meta["full_current_first_source_activation"] == 0
            for name in BASE_FIELDS:
                values[arm + "_" + name] = packed[name]
            if arm in ("F", "C"):
                for name in ATTENTION_FIELDS:
                    values[arm + "_" + name] = packed[name]
            if arm == "F":
                contexts = context
                values.update(
                    context_mask=array(context["mask"]),
                    rope_cos=array(context["rope"][0]),
                    rope_sin=array(context["rope"][1]),
                )
                values.update({"coeff_" + n: v for n, v in derivatives(packed["gate"]).items()})
            elif arm != "FZ":
                for name in ("before", "mid", "gate", "up", "norm", "z", "attwrite"):
                    assert np.array_equal(packed[name], values["C_" + name]), (arm, name)
            metadata[arm] = meta
        assert contexts is not None
        for name in BASE_FIELDS:
            assert np.array_equal(values["F_" + name], values["FZ_" + name]), name
        values["donor_delta"] = donor
        path = run / "forecasts" / (case["key"] + ".npz")
        np.savez_compressed(path, **values)
        total += path.stat().st_size
        assert total <= 96 * 1024**2
        match = next(r for r in bank_bindings if group_key(r["group"]) == group_key(case["group"]))
        save_json(
            path.with_suffix(".json"),
            dict(
                key=case["key"],
                arrays_sha256=sha(path),
                arms=metadata,
                ordinary_true=0,
                full_current_first_source_SiLU=0,
                donor_keys=match["donor_keys"],
                bank_delta_sha256=digest(donor),
                False_zero_exact=True,
                all_prefixes_exact=True,
                measured=stamp(),
            ),
        )
        entries.append(
            dict(
                key=case["key"],
                arrays_sha256=sha(path),
                metadata_sha256=sha(path.with_suffix(".json")),
                bytes=path.stat().st_size,
            )
        )
        print("285 forecast", len(entries), "/32", flush=True)
    save_json(
        run / "forecast-status.json",
        dict(
            status="complete",
            ordinary_true=0,
            cases=entries,
            raw_bytes=total,
            ended=stamp(),
            counts=dict(
                full_forwards=224,
                full_false=32,
                candidate_full_forwards=192,
                first_source_current_SiLU_coordinates=0,
                other_source_sites_finite=len(model.model.layers) - 1,
                all_receiving_operations_finite=True,
            ),
        ),
    )


def native(run, reference, public_commit=None, public_branch=None, model=None):
    design, bank = bindings(run, reference, model)
    assert not (run / "native-started.json").exists() and not (run / "native").exists()
    audit, seal, status = (
        read(run / n) for n in ("forecast-audit.json", "forecast-seal.json", "forecast-status.json")
    )
    assert audit["status"] == "passed" and status["status"] == "complete"
    assert audit["ordinary_true"] == seal["ordinary_true"] == status["ordinary_true"] == 0
    assert seal["cases"] == status["cases"] and len(status["cases"]) == 32
    assert seal["design_sha256"] == sha(run / "design.json")
    assert seal["forecast_audit_sha256"] == sha(run / "forecast-audit.json")
    assert seal["forecast_status_sha256"] == sha(run / "forecast-status.json")
    assert seal["bank_config_sha256"] == sha(run / "bank-config.json")
    assert seal["source_hashes"] == design["source_hashes"]
    assert audit["design_sha256"] == sha(run / "design.json")
    assert audit["forecast_status_sha256"] == sha(run / "forecast-status.json")
    assert audit["bank_config_sha256"] == sha(run / "bank-config.json")
    assert seal["csv_sha256"] == audit["csv_sha256"]
    for name, expected in audit["csv_sha256"].items():
        assert sha(run / name) == expected
    assert audit["source_sha256"] == sha(ROOT / "research/boolean_coupling/audit285.py")
    assert seal["full_current_first_source_SiLU"] == audit["full_current_first_source_SiLU"] == 0
    assert [e["key"] for e in status["cases"]] == [c["key"] for c in design["cases"]]
    assert len({e["key"] for e in status["cases"]}) == 32
    assert status["raw_bytes"] == sum(e["bytes"] for e in status["cases"])
    # Fail closed globally: all qualified arrays and metadata still have their
    # sealed bytes before even the first native forward is permitted.
    for entry in seal["cases"]:
        path = run / "forecasts" / (entry["key"] + ".npz")
        assert sha(path) == entry["arrays_sha256"] and path.stat().st_size == entry["bytes"]
        assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert meta["key"] == entry["key"] and meta["arrays_sha256"] == entry["arrays_sha256"]
        assert meta["ordinary_true"] == meta["full_current_first_source_SiLU"] == 0
        assert set(meta["arms"]) == set(ARMS)
    expected_seal = sha(run / "forecast-seal.json")
    if design["synthetic"]:
        assert (
            model is not None
            and public_commit == "synthetic285:" + expected_seal
            and public_branch == "synthetic285"
        )
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
        published = subprocess.check_output(
            ["git", "show", public_commit + ":" + PUBLIC_SEAL], cwd=ROOT
        )
        assert hashlib.sha256(published).hexdigest() == expected_seal
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", design["source_commit"], public_commit],
            cwd=ROOT,
            check=True,
        )
        verified = True
    save_json(
        run / "native-started.json",
        dict(
            began=stamp(),
            forecast_seal_sha256=expected_seal,
            public_commit=public_commit,
            public_branch=public_branch,
            remote_head_verified=verified,
            synthetic=design["synthetic"],
            ordinary_true_before=0,
        ),
    )
    (run / "native").mkdir()
    model = load_model(True) if model is None else model
    release(model)
    entries, total = [], 0
    for case in design["cases"]:
        release(model)
        assert shutil.disk_usage(run).free >= 512 * 1024**2
        engine = Engine(model, case, bank[group_key(case["group"])])
        with engine:
            packed, meta, _ = engine.forward("T")
        release(model)
        assert meta["false_primal_checks"] == 0 and meta["current_source_finite_sites"] == len(
            model.model.layers
        )
        path = run / "native" / (case["key"] + ".npz")
        np.savez_compressed(path, **{"T_" + n: packed[n] for n in BASE_FIELDS})
        total += path.stat().st_size
        assert status["raw_bytes"] + total <= 96 * 1024**2
        save_json(
            path.with_suffix(".json"),
            dict(
                key=case["key"],
                arrays_sha256=sha(path),
                arm=meta,
                measured=stamp(),
                ordinary_true=True,
                editing_hooks_applied=False,
            ),
        )
        entries.append(
            dict(
                key=case["key"],
                arrays_sha256=sha(path),
                metadata_sha256=sha(path.with_suffix(".json")),
                bytes=path.stat().st_size,
            )
        )
        print("285 native", len(entries), "/32", flush=True)
    save_json(
        run / "native-status.json",
        dict(
            status="complete",
            ordinary_true=32,
            cases=entries,
            raw_bytes=total,
            ended=stamp(),
            native_started_sha256=sha(run / "native-started.json"),
            forecast_seal_sha256=expected_seal,
            counts=dict(full_forwards=32),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "native"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-279", type=Path, required=True)
    parser.add_argument("--public-commit")
    parser.add_argument("--public-branch")
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            forecast(args.reference_279.resolve(), args.run.resolve())
        else:
            native(
                args.run.resolve(),
                args.reference_279.resolve(),
                args.public_commit,
                args.public_branch,
            )
    except Exception as exc:
        if args.run.exists():
            p = args.run / ("failure-" + args.phase + ".json")
            if not p.exists():
                save_json(p, dict(error=repr(exc), time=stamp()))
        raise
