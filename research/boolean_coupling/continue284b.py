"""Memory-only native continuation; replay every sealed partial case before new ones."""

import argparse
import gc
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import read, table
from .cache272 import installed_sources, save_json
from .carried284 import NATIVE, ROOT, Memory, bindings, load_forecast
from .core225b import load_model
from .origin282 import Engine, pack_observations
from .run272 import stamp

SOURCES = (
    "research/boolean_coupling/continue284b.py",
    "research/boolean_coupling/audit284b.py",
    "research/boolean_coupling/preflight284b.py",
    "docs/experiments/284b-native-cache-continuation.md",
)


def release(model, events, key, arm):
    if model.device.type == "mps":
        torch.mps.synchronize()
        before = int(torch.mps.driver_allocated_memory())
        live = int(torch.mps.current_allocated_memory())
        torch.mps.empty_cache()
        torch.mps.synchronize()
        events.append(
            dict(
                key=key,
                arm=arm,
                device="mps",
                driver_before=before,
                driver_after=int(torch.mps.driver_allocated_memory()),
                live_before=live,
                live_after=int(torch.mps.current_allocated_memory()),
            )
        )
    else:
        events.append(
            dict(
                key=key,
                arm=arm,
                device=str(model.device),
                driver_before=0,
                driver_after=0,
                live_before=0,
                live_after=0,
            )
        )


def one_case(model, case, data, fmeta, path, events):
    engine = Engine(model, case)
    engine.anchors = [{n: data[f"F_node{site}_" + n] for n in ("gate", "phi")} for site in (1, 2)]
    engine.base_product = (data["B_node2_phi"] * data["B_node2_up"]).astype(np.float32)
    values, metadata = {}, {}
    for arm in NATIVE:
        with engine:
            packed, meta, nodes = engine.forward(arm)
        release(model, events, case["key"], arm)
        assert meta["prefix_hashes"] == fmeta["arms"]["F"]["prefix_hashes"]
        if arm != "T":
            assert np.array_equal(packed["source"][0], data["B_source"][0])
            for n in ("mid", "gate", "up"):
                assert np.array_equal(nodes[1][n], data["B_node2_" + n])
        values.update({arm + "_" + n: v for n, v in packed.items() if n != "phi"})
        metadata[arm] = meta
        if arm == "C1":
            for n in ("gate", "up", "phi"):
                values["C1_node2_" + n] = nodes[1][n]
            values["full_product"] = (nodes[1]["phi"] * nodes[1]["up"]).astype(np.float32)
            full = values["full_product"].astype(float) - engine.base_product.astype(float)
            values["full_remainder"] = full
            engine.product_deltas.update(U=full, N=full - data["response_increment"])
    for n in ("source", "final", "scores", "write"):
        assert np.array_equal(values["U_" + n], values["C1_" + n])
    aliases, logical = pack_observations(path, values, Memory(data))
    return dict(
        key=case["key"],
        arrays_sha256=sha(path),
        arms=metadata,
        array_aliases=aliases,
        logical_arrays=logical,
        forecast_arrays_sha256=fmeta["arrays_sha256"],
        forecast_metadata_sha256=sha(path.parent.parent / "forecasts" / (case["key"] + ".json")),
        full_control_exact=True,
    )


def main(original, out, failure_log):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 576 * 1024**2
    design = bindings(original)
    fs = read(original / "forecast-status.json")
    fa = read(original / "forecast-audit.json")
    assert fs["status"] == "complete" and fa["status"] == "passed" and len(fs["cases"]) == 32
    assert not (original / "native-status.json").exists()
    failure = read(original / "failure-native.json")
    assert failure["error"] == "AssertionError()"
    guard = "assert shutil.disk_usage(run).free >= 512 * 1024**2"
    assert guard in failure_log.read_text()
    partial = sorted((original / "native").glob("*.npz"))
    completed = {p.stem: read(p.with_suffix(".json")) for p in partial}
    expected_keys = [c["key"] for c in design["cases"]]
    assert len(completed) == 17 and set(completed) == set(expected_keys[:17])
    assert len(list((original / "native").glob("*.json"))) == 17
    for path in partial:
        assert sha(path) == completed[path.stem]["arrays_sha256"]
    start = read(original / "native-started.json")
    assert (
        start["forecast_seal_sha256"]
        == start["receipt"]["forecast_seal_sha256"]
        == sha(original / "forecast-seal.json")
    )
    if not design["synthetic"]:
        receipt = start["receipt"]
        proof = subprocess.check_output(
            ["git", "show", receipt["public_forecast_commit"] + ":" + receipt["public_seal_path"]],
            cwd=ROOT,
        )
        assert hashlib.sha256(proof).hexdigest() == sha(original / "forecast-seal.json")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        assert (
            subprocess.check_output(
                ["git", "ls-remote", "origin", "refs/heads/" + receipt["branch"]],
                cwd=ROOT,
                text=True,
            ).split()[0]
            == head
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", receipt["public_forecast_commit"], head],
            cwd=ROOT,
            check=True,
        )
    out.mkdir()
    for p in original.iterdir():
        if p.name in ("sources", "forecasts"):
            shutil.copytree(p, out / p.name, copy_function=os.link)
        elif p.is_file() and (
            p.name in ("design.json", "runtime.json", "weight-hashes.json", "native-started.json")
            or p.name.startswith("forecast")
        ):
            os.link(p, out / p.name)
    for name in ("native", "replay", "native-continuation-sources", "retained-native-stop"):
        (out / name).mkdir()
    for p in (original / "native").iterdir():
        os.link(p, out / "native" / p.name)
    for name in ("failure-native.json", "native-started.json"):
        shutil.copy2(original / name, out / "retained-native-stop" / name)
    source_hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "native-continuation-sources" / Path(n).name)
    retained = dict(
        cases=[
            dict(
                key=c["key"],
                arrays_sha256=sha(original / "native" / (c["key"] + ".npz")),
                metadata_sha256=sha(original / "native" / (c["key"] + ".json")),
                bytes=(original / "native" / (c["key"] + ".npz")).stat().st_size,
            )
            for c in design["cases"][:17]
        ],
        status="incomplete_native_storage_guard",
        full_forwards=68,
        no_orphan_case18=True,
        failure_log_sha256=sha(failure_log),
        guard_expression=guard,
        source_sha256=sha(ROOT / "research/boolean_coupling/run284.py"),
    )
    save_json(out / "retained-native-stop/inventory.json", retained)
    save_json(
        out / "native-continuation-design.json",
        dict(
            qualification="284b-unused-MPS-cache-and-exact-partial-replay",
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=source_hashes,
            design_sha256=sha(out / "design.json"),
            forecast_seal_sha256=sha(out / "forecast-seal.json"),
            original_failure_sha256=sha(original / "failure-native.json"),
            original_native_started_sha256=sha(original / "native-started.json"),
            original_partial_inventory_sha256=sha(out / "retained-native-stop/inventory.json"),
            previously_measured_native_cases=17,
            predictions_changed=False,
            scientific_thresholds_changed=False,
            numerical_validation_scope_changed=False,
            per_case_reserve_MiB=512,
            combined_raw_cap_MiB=96,
            replay_every_completed_case_before_new_native=True,
            policy="After each completed forward and before each case, synchronize and empty only unused MPS allocator cache; model parameters/live tensors unchanged. Record memory and exact outputs.",
            began=stamp(),
        ),
    )
    model = load_model(True)
    assert installed_sources() == design["installed_sources"]
    cases = {c["key"]: c for c in design["cases"]}
    events, replays, entries = [], [], []
    release(model, events, "initial", "loaded")
    for i, entry in enumerate(fs["cases"]):
        key = entry["key"]
        gc.collect()
        release(model, events, key, "before_case")
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        data, fmeta = load_forecast(out, entry)
        replay = i < 17
        path = out / ("replay" if replay else "native") / (key + ".npz")
        meta = one_case(model, cases[key], data, fmeta, path, events)
        # Case metadata remains byte-identical; cache policy lives separately.
        assert meta["forecast_metadata_sha256"] == entry["metadata_sha256"]
        save_json(path.with_suffix(".json"), meta)
        if replay:
            old = retained["cases"][i]
            assert (
                sha(path) == old["arrays_sha256"]
                and sha(path.with_suffix(".json")) == old["metadata_sha256"]
            )
            replays.append(
                dict(
                    key=key,
                    arrays_sha256=sha(path),
                    metadata_sha256=sha(path.with_suffix(".json")),
                    exact=True,
                )
            )
        else:
            assert len(replays) == 17 and all(r["exact"] for r in replays)
        actual = out / "native" / (key + ".npz")
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(actual),
                metadata_sha256=sha(actual.with_suffix(".json")),
                bytes=actual.stat().st_size,
            )
        )
        total = sum(
            p.stat().st_size
            for directory in ("forecasts", "native", "replay")
            for p in (out / directory).glob("*.npz")
        )
        assert total <= 96 * 1024**2
        print("284b", ("replayed" if replay else "new native"), i + 1, "/32", flush=True)
    table(out / "native-cache-events.csv", events)
    save_json(
        out / "native-status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=sum(r["bytes"] for r in entries),
            ended=stamp(),
            counts=dict(full_forwards=128, ordinary_true=32),
        ),
    )
    save_json(
        out / "native-continuation-status.json",
        dict(
            status="complete",
            replays=replays,
            all_replay_arrays_and_metadata_exact=True,
            original_partial_full_forwards=68,
            replay_full_forwards=68,
            new_unmeasured_full_forwards=60,
            total_native_producer_full_forwards=196,
            unique_native_cases=32,
            actual_native_True_producer_forwards=49,
            cache_events_sha256=sha(out / "native-cache-events.csv"),
            ended=stamp(),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--failure-log", type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.original.resolve(), a.out.resolve(), a.failure_log.resolve())
    except Exception as exc:
        if a.out.exists() and not (a.out / "failure-native-continuation.json").exists():
            save_json(
                a.out / "failure-native-continuation.json", dict(error=repr(exc), time=stamp())
            )
        raise
