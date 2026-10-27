"""Saved-array audit of the frozen 286b full-shape numerical qualification."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .qualify286b import BOUNDS, ROOT
from .run272 import stamp
from .run286 import entry_data


def main(run, reference, readout):
    assert not (run / "audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == "286b" and status["status"] == "complete"
    assert design["numerical_bounds"] == BOUNDS
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    assert design["reference_hashes"] == {
        "285_design": sha(reference / "design.json"),
        "286_design": sha(readout / "design.json"),
        "286_status": sha(readout / "status.json"),
        "286_audit": sha(readout / "audit.json"),
    }
    fs = {entry["key"]: entry for entry in read(reference / "forecast-status.json")["cases"]}
    rs = {entry["key"]: entry for entry in read(readout / "status.json")["cases"]}
    es = {entry["key"]: entry for entry in status["cases"]}
    assert len(es) == len(status["cases"]) == len(design["cases"])
    rows, layer_rows, counts = [], [], []
    for case in design["cases"]:
        key = case["key"]
        data = entry_data(run, "cases", es[key])
        forecast, cached = (
            entry_data(reference, "forecasts", fs[key]),
            entry_data(readout, "cases", rs[key]),
        )
        meta = read(run / "cases" / (key + ".json"))
        assert meta["forecast_entry"] == fs[key] and meta["readout_entry"] == rs[key]
        assert meta["first_source_policy"]["False_primal_exact"]
        counts.append(meta["counts"])

        def check(name, actual, expected, bound):
            error = float(np.max(np.abs(np.asarray(actual, float) - np.asarray(expected, float))))
            rows.append(
                dict(
                    key=key,
                    name=name,
                    error=error,
                    bound=bound,
                    passes=bool(np.isfinite(error) and error <= bound),
                )
            )

        for name in ("source", "final", "scores"):
            check("ordinary_F_" + name, data["ordinary_F_" + name], forecast["F_" + name], 0.0)
        check("first_M_source", data["first_M_source"], forecast["M_source"], 0.0)
        for anchor in ("F", "M"):
            for name in ("final", "scores"):
                check(
                    anchor + "_full_replay_" + name,
                    data[anchor + "_" + name],
                    forecast[anchor + "_" + name],
                    BOUNDS["full_replay"],
                )
            check(anchor + "_causal_prefix", data[anchor + "_prefix_errors"], 0.0, 0.0)
            check(
                anchor + "_reverse_primal",
                data[anchor + "_reverse_primal_errors"],
                0.0,
                BOUNDS["full_replay"],
            )
            full, old = data[anchor + "_source_trace"], cached["g" + anchor + "_source_trace"]
            assert full.shape == old.shape
            check(anchor + "_all_source_adjoints", full, old, BOUNDS["source_adjoint"])
            full_projection = float(np.dot(full[0].astype(float), cached["delta"]))
            old_projection = float(np.dot(old[0].astype(float), cached["delta"]))
            check(
                anchor + "_projected_response",
                full_projection,
                old_projection,
                BOUNDS["projected_response"],
            )
            for index, (new, previous) in enumerate(zip(full, old, strict=True)):
                error = float(np.max(abs(new.astype(float) - previous.astype(float))))
                relative = float(
                    np.linalg.norm(new.astype(float) - previous.astype(float))
                    / max(np.linalg.norm(previous.astype(float)), 1e-30)
                )
                layer_rows.append(
                    dict(
                        key=key,
                        anchor=anchor,
                        receiving_block=index + 2,
                        maximum_coordinate_error=error,
                        relative_error=relative,
                        full_tail_norm=float(data[anchor + "_tail_norm_trace"][index]),
                        cached_tail_norm=float(cached["g" + anchor + "_tail_norm_trace"][index]),
                    )
                )
    for name, contents in (("checks", rows), ("layers", layer_rows)):
        with (run / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contents[0]))
            writer.writeheader()
            writer.writerows(contents)
    assert sum(entry["bytes"] for entry in status["cases"]) == status["raw_bytes"]
    assert status["counts"] == {name: sum(row[name] for row in counts) for name in counts[0]}
    result = dict(
        experiment="286b",
        numerical_only=True,
        synthetic=design["synthetic"],
        status="passed" if all(row["passes"] for row in rows) else "failed",
        checks=len(rows),
        failed_checks=sum(not row["passes"] for row in rows),
        maximum_source_adjoint_coordinate_error=max(
            row["maximum_coordinate_error"] for row in layer_rows
        ),
        maximum_source_adjoint_relative_error=max(row["relative_error"] for row in layer_rows),
        maximum_projected_response_error=max(
            row["error"] for row in rows if row["name"].endswith("_projected_response")
        ),
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"),
        csv_sha256={name + ".csv": sha(run / (name + ".csv")) for name in ("checks", "layers")},
        counts=status["counts"],
        ended=stamp(),
        boundary="All coordinates of the source-row reverse adjoint at each later block are compared between execution shapes; this is not a full Jacobian or an independent implementation of the library derivatives. Original286 raw coordinate failures remain failed.",
    )
    save_json(run / "audit.json", result)
    print(result, flush=True)
    assert result["status"] == "passed", "Original full-shape bounds failed; result preserved."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-285", type=Path, required=True)
    parser.add_argument("--readout-286", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.reference_285.resolve(), args.readout_286.resolve())
