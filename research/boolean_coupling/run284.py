"""284 globally sealed forecasts, followed by a separately gated native phase."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .carried283 import products, random_writes, response
from .carried284 import (
    FORECAST,
    NATIVE,
    ROOT,
    SOURCES,
    Memory,
    bindings,
    load_forecast,
    validate_inputs,
)
from .checkpoint238 import locate, weights
from .composition280 import digest
from .core225b import load_model
from .origin282 import Engine, local_pair, pack_observations, remainder, split_inputs
from .run272 import runtime, stamp


def forecast(out, cases=None, historical=None, synthetic=False, repetition=False):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 640 * 1024**2
    if not synthetic and not repetition:
        assert not (ROOT / "results/carried-state-forecast-284/native-audit.json").exists(), (
            "Published literals are now known; use explicit repetition mode."
        )
    cases = read(ROOT / "research/boolean_coupling/cases284.json") if cases is None else cases
    history = (
        read(ROOT / "research/boolean_coupling/historical-input-hashes-284.json")
        if historical is None
        else historical
    )
    validate_inputs(cases, history["hashes"])
    if not synthetic:
        for r in history["provenance"]:
            assert sha(ROOT / "research/boolean_coupling" / r["path"]) == r["sha256"]
    folder, checkpoint = locate(ROOT)
    out.mkdir()
    for name in ("sources", "forecasts"):
        (out / name).mkdir()
    source_hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=284,
            globally_pre_native=True,
            synthetic=synthetic,
            published_literal_repetition=repetition,
            cases=cases,
            historical_input_hashes=history["hashes"],
            forecast_arms=FORECAST,
            native_arms=NATIVE,
            checkpoint=checkpoint,
            installed_sources=installed_sources(),
            source_hashes=source_hashes,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            began=stamp(),
            unchanged_method_sha256=sha(ROOT / "research/boolean_coupling/carried283.py"),
            storage=dict(
                initial_free_MiB=640,
                per_case_free_MiB=512,
                new_array_cap_MiB=96,
                lossless_aliases=True,
            ),
            primary="UNCHANGED283P: original native + full C1-B local/effect + paired local + C1-K incremental local; >=24/32+>=6/8stratum.",
            numerical_scope="Same staged Float64 derivative/public Float32 scope as283, original CPU64up/raw failures retained.",
            boundary="Fresh literal instances of dependent templates; one observedFalse plus known edit, extra finiteCmap and substantial finite source/receiver work. No efficient/circuit/priority/generalQB claim.",
        ),
    )
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 1, "mlp")
    save_json(out / "weight-hashes.json", {n: digest(v) for n, v in w.items()})
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    entries, total = [], 0
    for case in cases:
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        key, values, metadata = case["key"], {}, {}
        engine = Engine(model, case)

        def execute(arm):
            with engine:
                packed, meta, nodes = engine.forward(arm)
            if arm != "F":
                assert meta["prefix_hashes"] == metadata["F"]["prefix_hashes"]
            if arm not in ("F", "B", "FZ"):
                assert np.array_equal(packed["source"][0], values["B_source"][0])
                for n in ("mid", "gate", "up"):
                    assert np.array_equal(nodes[1][n], values["B_node2_" + n])
            if arm in ("F", "B"):
                for site in (1, 2):
                    for n in ("mid", "gate", "up", "phi", "raw_write"):
                        values[f"{arm}_node{site}_" + n] = nodes[site - 1][n]
            values.update({arm + "_" + n: v for n, v in packed.items() if n != "phi"})
            metadata[arm] = meta
            return packed, nodes

        f, fn = execute("F")
        b, bn = execute("B")
        points, components = split_inputs(
            dict(mid=fn[1]["mid"], before=f["source"][0]),
            dict(mid=bn[1]["mid"], before=b["source"][0]),
        )
        values.update({"point_" + n: v for n, v in points.items()})
        values.update({"component_" + n: v for n, v in components.items()})
        corner = local_pair(model, case, engine.anchors[1], points["C"])
        calculated = response(engine.anchors[1], corner, points["U"], w)
        values.update({"corner_C_" + n: v for n, v in corner.items()})
        values.update({"response_" + n: v for n, v in calculated.items()})
        engine.base_product = (bn[1]["phi"] * bn[1]["up"]).astype(np.float32)
        assert digest(engine.base_product) == metadata["B"]["product_hashes"][1]
        engine.product_deltas = products(remainder(corner), calculated)
        for arm in FORECAST[2:]:
            packed, _ = execute(arm)
            if arm == "P":
                engine.draws = random_writes(
                    case, packed["write"].astype(float) - values["K_write"].astype(float)
                )
                values.update({"draw_" + n: v for n, v in engine.draws.items()})
        for n in ("source", "final", "scores", "write"):
            assert np.array_equal(values["FZ_" + n], values["F_" + n])
        p = out / "forecasts" / (key + ".npz")
        aliases, logical = pack_observations(p, values, Memory())
        total += p.stat().st_size
        assert total <= 96 * 1024**2
        save_json(
            p.with_suffix(".json"),
            dict(
                key=key,
                arrays_sha256=sha(p),
                arms=metadata,
                array_aliases=aliases,
                logical_arrays=logical,
                ordinary_true=0,
                full_current_source2_primal=0,
                own_B_inputs=True,
                zero_exact=True,
                all_prefixes_exact=True,
                staged_not_whole_model_derivative=True,
            ),
        )
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
                bytes=p.stat().st_size,
            )
        )
        print("284 forecasts", len(entries), "/32", flush=True)
    save_json(
        out / "forecast-status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=total,
            ordinary_true=0,
            full_current_source2_primal=0,
            ended=stamp(),
            counts=dict(
                full_forwards=320,
                finite_C_corner_maps=32,
                local_down_maps=64,
                staged_norm_jvps=32,
                staged_gate_up_matrix_products=64,
                other_source_sites_finite=len(model.model.layers) - 2,
                all_receiving_operations_finite=True,
            ),
        ),
    )


def native(run, receipt):
    design = bindings(run)
    status, audit, seal = (
        read(run / (n + ".json")) for n in ("forecast-status", "forecast-audit", "forecast-seal")
    )
    assert (
        status["status"] == "complete"
        and audit["status"] == "passed"
        and len(status["cases"]) == 32
    )
    assert seal["forecast_audit_sha256"] == sha(run / "forecast-audit.json") and seal[
        "forecast_status_sha256"
    ] == sha(run / "forecast-status.json")
    assert receipt["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    assert seal["design_sha256"] == audit["design_sha256"] == sha(run / "design.json")
    assert audit["source_sha256"] == sha(ROOT / "research/boolean_coupling/audit284.py")
    assert seal["cases"] == status["cases"]
    if not design["synthetic"] and not design["published_literal_repetition"]:
        assert receipt == receipt_for(run, receipt["public_forecast_commit"], receipt["branch"])
    assert not (run / "native-started.json").exists() and not (run / "native").exists()
    save_json(
        run / "native-started.json",
        dict(
            began=stamp(),
            receipt=receipt,
            forecast_seal_sha256=sha(run / "forecast-seal.json"),
            native_before_forecast_seal=0,
            full_current_source2_before_forecast_seal=0,
            published_literal_repetition=design["published_literal_repetition"],
        ),
    )
    (run / "native").mkdir()
    model = load_model(True)
    assert installed_sources() == design["installed_sources"]
    cases = {c["key"]: c for c in design["cases"]}
    entries, total = [], 0
    for entry in status["cases"]:
        assert shutil.disk_usage(run).free >= 512 * 1024**2
        key = entry["key"]
        case = cases[key]
        d, fmeta = load_forecast(run, entry)
        engine = Engine(model, case)
        engine.anchors = [{n: d[f"F_node{site}_" + n] for n in ("gate", "phi")} for site in (1, 2)]
        engine.base_product = (d["B_node2_phi"] * d["B_node2_up"]).astype(np.float32)
        values, metadata = {}, {}
        for arm in NATIVE:
            with engine:
                packed, meta, nodes = engine.forward(arm)
            assert meta["prefix_hashes"] == fmeta["arms"]["F"]["prefix_hashes"]
            if arm != "T":
                assert np.array_equal(packed["source"][0], d["B_source"][0])
                for n in ("mid", "gate", "up"):
                    assert np.array_equal(nodes[1][n], d["B_node2_" + n])
            values.update({arm + "_" + n: v for n, v in packed.items() if n != "phi"})
            metadata[arm] = meta
            if arm == "C1":
                for n in ("gate", "up", "phi"):
                    values["C1_node2_" + n] = nodes[1][n]
                values["full_product"] = (nodes[1]["phi"] * nodes[1]["up"]).astype(np.float32)
                full = values["full_product"].astype(float) - engine.base_product.astype(float)
                values["full_remainder"] = full
                engine.product_deltas.update(U=full, N=full - d["response_increment"])
        for n in ("source", "final", "scores", "write"):
            assert np.array_equal(values["U_" + n], values["C1_" + n])
        p = run / "native" / (key + ".npz")
        aliases, logical = pack_observations(p, values, Memory(d))
        total += p.stat().st_size
        assert total + status["raw_bytes"] <= 96 * 1024**2
        save_json(
            p.with_suffix(".json"),
            dict(
                key=key,
                arrays_sha256=sha(p),
                arms=metadata,
                array_aliases=aliases,
                logical_arrays=logical,
                forecast_arrays_sha256=entry["arrays_sha256"],
                forecast_metadata_sha256=entry["metadata_sha256"],
                full_control_exact=True,
            ),
        )
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
                bytes=p.stat().st_size,
            )
        )
        print("284 native", len(entries), "/32", flush=True)
    save_json(
        run / "native-status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=total,
            ended=stamp(),
            counts=dict(full_forwards=128, ordinary_true=32),
        ),
    )


def receipt_for(run, commit, branch):
    design = bindings(run)
    if design["published_literal_repetition"]:
        return dict(
            published_literal_repetition=True,
            fresh_confirmation=False,
            forecast_seal_sha256=sha(run / "forecast-seal.json"),
        )
    assert not design["synthetic"] and commit and branch
    assert (
        subprocess.check_output(
            ["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=ROOT, text=True
        ).split()[0]
        == commit
    )
    public_path = "results/carried-state-forecast-284/forecast-seal.json"
    published = subprocess.check_output(["git", "show", commit + ":" + public_path], cwd=ROOT)
    assert hashlib.sha256(published).hexdigest() == sha(run / "forecast-seal.json")
    return dict(
        public_forecast_commit=commit,
        branch=branch,
        public_seal_path=public_path,
        forecast_seal_sha256=sha(run / "forecast-seal.json"),
        remote_head_verified=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("forecast", "native"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--repeat-known-literals", action="store_true")
    p.add_argument("--public-forecast-commit")
    p.add_argument("--branch")
    a = p.parse_args()
    run = a.run.resolve()
    try:
        if a.stage == "forecast":
            forecast(run, repetition=a.repeat_known_literals)
        else:
            assert a.repeat_known_literals == bindings(run)["published_literal_repetition"]
            native(run, receipt_for(run, a.public_forecast_commit, a.branch))
    except Exception as exc:
        if run.exists() and not (run / ("failure-" + a.stage + ".json")).exists():
            save_json(run / ("failure-" + a.stage + ".json"), dict(error=repr(exc), time=stamp()))
        raise
