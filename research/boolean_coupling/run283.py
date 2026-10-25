"""Retrospective283: predicted carry-conditioned gate/up increments, then controls."""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .carried283 import (
    ARMS,
    CANDIDATES,
    CONTROLS,
    ROOT,
    SOURCES,
    Observations,
    products,
    random_writes,
    response,
)
from .checkpoint238 import locate, weights
from .composition280 import digest
from .core225b import load_model
from .origin282 import Engine, local_pair, pack_observations, remainder, split_inputs
from .run272 import runtime, stamp

FIRST = ("mid", "gate", "up", "phi", "raw_write")


def main(reference, upstream, out):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 640 * 1024**2
    old, complete, prior_audit = (
        read(reference / (n + ".json")) for n in ("design", "status", "audit")
    )
    assert (
        old["experiment"] == 282
        and complete["status"] == "complete"
        and prior_audit["status"] == "passed"
    )
    assert prior_audit["analysis_qualification"] == "282c-paired-whole-secondary-unavailable"
    assert len(old["cases"]) == len(complete["cases"]) == 32
    for n, h in old["source_hashes"].items():
        assert sha(ROOT / n) == h
    for n, h in old["reference_hashes"].items():
        assert sha(upstream / n) == h
    folder, checkpoint = locate(ROOT)
    assert checkpoint == old["checkpoint"]
    out.mkdir()
    for name in ("cases", "sources"):
        (out / name).mkdir()
    source_hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in source_hashes:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=283,
            retrospective=True,
            cases=old["cases"],
            arms=ARMS,
            checkpoint=checkpoint,
            installed_sources=installed_sources(),
            source_hashes=source_hashes,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            began=stamp(),
            reference_hashes={
                n: sha(reference / n)
                for n in (
                    "design.json",
                    "status.json",
                    "audit.json",
                    "audit-b-design.json",
                    "audit-c-design.json",
                )
            },
            upstream_hashes={
                n: sha(upstream / n) for n in ("design.json", "status.json", "audit.json")
            },
            storage=dict(
                initial_free_MiB=640,
                per_case_free_MiB=512,
                new_array_cap_MiB=64,
                lossless_aliases=True,
            ),
            primary="P original native + full C1-B local/actual effect + paired local + incremental C1-K local25%/.95/material; >=24/32+>=6/8stratum.",
            numerical_scope="Declared staged Float64 derivative, not public Float32/whole-block Jacobian. Public finite shape replay exact; CPU64up bridge diagnostic+rounding envelope as282b. Other original1e-4 bridges/coefficients retained.",
            boundary="Old dependent32/16pairs, extra finiteCmap and staged tangent,26later source/allreceivers finite; no fresh/circuit/efficient/generalQB claim.",
        ),
    )
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 1, "mlp")
    save_json(out / "weight-hashes.json", {n: digest(v) for n, v in w.items()})
    model = load_model(True)
    save_json(out / "runtime.json", runtime(model))
    previous = {r["key"]: r for r in complete["cases"]}
    entries, total_bytes = [], 0
    for case in old["cases"]:
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        key, values, metadata = case["key"], {}, {}
        engine = Engine(model, case)

        def execute(arm):
            with engine:
                packed, meta, nodes = engine.forward(arm)
            if arm != "F":
                assert meta["prefix_hashes"] == metadata["F"]["prefix_hashes"]
            if arm not in ("F", "FZ", "T", "B"):
                assert np.array_equal(packed["source"][0], values["B_source"][0])
                for n in ("mid", "gate", "up"):
                    assert np.array_equal(nodes[1][n], values["B_node2_" + n]), (arm, n)
            for site in (1, 2):
                if arm in ("F", "B"):
                    for n in FIRST:
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
        carry = remainder(corner)
        engine.base_product = (bn[1]["phi"] * bn[1]["up"]).astype(np.float32)
        assert digest(engine.base_product) == metadata["B"]["product_hashes"][1]
        engine.product_deltas = products(carry, calculated)
        for arm in CANDIDATES:
            packed, _ = execute(arm)
            if arm == "P":
                engine.draws = random_writes(
                    case, packed["write"].astype(float) - values["K_write"].astype(float)
                )
                values.update({"draw_" + n: v for n, v in engine.draws.items()})
        for n in ("source", "final", "scores", "write"):
            assert np.array_equal(values["FZ_" + n], values["F_" + n]), ("zero", n)
        # Full source2 activation and native True are only evaluated after candidates.
        c1, c1nodes = execute(CONTROLS[0])
        full_product = (c1nodes[1]["phi"] * c1nodes[1]["up"]).astype(np.float32)
        full_remainder = full_product.astype(float) - engine.base_product.astype(float)
        values["full_product"] = full_product
        values["full_remainder"] = full_remainder
        engine.product_deltas.update(U=full_remainder, N=full_remainder - calculated["increment"])
        for arm in CONTROLS[1:]:
            execute(arm)
        for n in ("source", "final", "scores", "write"):
            assert np.array_equal(values["U_" + n], c1[n]), ("full", n)
        old_values = Observations(reference, upstream, previous[key])
        for arm in ("F", "B", "K", "C1", "T"):
            for n in ("source", "final", "scores", "write"):
                assert np.array_equal(values[arm + "_" + n], old_values[arm + "_" + n]), (
                    "archive",
                    arm,
                    n,
                )
        path = out / "cases" / (key + ".npz")
        aliases, logical = pack_observations(path, values, old_values)
        total_bytes += path.stat().st_size
        assert total_bytes <= 64 * 1024**2
        save_json(
            path.with_suffix(".json"),
            dict(
                key=key,
                arrays_sha256=sha(path),
                arms=metadata,
                array_aliases=aliases,
                logical_arrays=logical,
                reference_entry=previous[key],
                own_B_inputs=True,
                candidates_before_full_calibration=True,
                archive_replays_exact=True,
                staged_not_whole_model_derivative=True,
                zero_exact=True,
                full_control_exact=True,
            ),
        )
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(path),
                metadata_sha256=sha(path.with_suffix(".json")),
                bytes=path.stat().st_size,
            )
        )
        print("283", len(entries), "/32 complete", flush=True)
    save_json(
        out / "status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=total_bytes,
            ended=stamp(),
            counts=dict(
                full_forwards=len(ARMS) * len(entries),
                finite_C_corner_maps=len(entries),
                local_down_maps=2 * len(entries),
                staged_norm_jvps=len(entries),
                staged_gate_up_matrix_products=2 * len(entries),
                other_source_sites_finite=len(model.model.layers) - 2,
                all_receiving_operations_finite=True,
            ),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-282", type=Path, required=True)
    p.add_argument("--reference-281d", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.reference_282.resolve(), a.reference_281d.resolve(), a.out.resolve())
    except Exception as exc:
        if a.out.exists() and not (a.out / "failure.json").exists():
            save_json(a.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
