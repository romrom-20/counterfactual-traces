"""Unchanged-rule replay after the preserved 292 disk-reserve stop."""

import argparse
from pathlib import Path

import numpy as np

from . import run292
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .joint292 import ROOT


def main(original, out, model=None):
    assert not out.exists()
    d = run292.binding(original, model)
    assert not (original / "forecast-status.json").exists()
    assert not (original / "native-started.json").exists() and not (original / "native").exists()
    assert (original / "failure-forecast.json").exists()
    files = list((original / "forecasts").glob("*.npz"))
    assert len(files) == 1 and files[0].stem == d["cases"][0]["key"]
    old = files[0]
    m = read(old.with_suffix(".json"))
    assert sha(old) == m["arrays_sha256"] and m["ordinary_true"] == 0
    assert all(meta["ordinary_true"] == 0 for meta in m["arms"].values())
    with np.load(old) as z:
        recorded = {n: z[n] for n in z.files}
    source = {n: sha(ROOT / n) for n in d["source_hashes"]}
    assert source == d["source_hashes"]
    archived = run292.archive
    replayed = []

    def checked(folder, key, data, metadata):
        if not replayed:
            assert key == files[0].stem and set(data) == set(recorded)
            for n, v in data.items():
                assert v.dtype == recorded[n].dtype and v.shape == recorded[n].shape
                assert np.array_equal(v, recorded[n]), ("Original first-case replay failed", n)
            assert metadata["arms"] == m["arms"]
            item = archived(folder, key, data, metadata)
            save_json(
                out / "continuation.json",
                dict(
                    experiment="292b",
                    method_unchanged=True,
                    ordinary_true_before=0,
                    first_case_replayed_before_any_remaining_case=True,
                    original_design_sha256=sha(original / "design.json"),
                    original_failure_sha256=sha(original / "failure-forecast.json"),
                    original_array_sha256=sha(old),
                    original_metadata_sha256=sha(old.with_suffix(".json")),
                    replay_entry=item,
                    replayed_arrays=len(data),
                    source_hashes=source,
                    executor_sha256=sha(Path(__file__)),
                    protocol_sha256=sha(ROOT / "docs/experiments/292b-resource-continuation.md"),
                    boundary="All 102 predictor bytes, inputs, numerical bounds and guards unchanged. Original stopped evidence preserved. No native True observed.",
                ),
            )
            replayed.append(key)
            return item
        return archived(folder, key, data, metadata)

    run292.archive = checked
    try:
        run292.forecast(out, model=model, cases=d["cases"])
    finally:
        run292.archive = archived
    assert replayed == [d["cases"][0]["key"]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.original.resolve(), a.out.resolve())
    except Exception as exc:
        if a.out.exists() and not (a.out / "failure-continuation.json").exists():
            save_json(a.out / "failure-continuation.json", dict(error=repr(exc)))
        raise
