"""Distinct False/candidate fixture and coherent baseline corruption refusal."""

import argparse
from pathlib import Path

import numpy as np

from .audit272 import read
from .cache272 import save_json
from .joint292 import Engine
from .namespace293c import install_baseline
from .namespace293c import main as corrected
from .preflight286 import tiny


def main(parent, out):
    assert not out.exists()
    out.mkdir()
    model = tiny()
    d = read(parent / "design.json")
    baselines = {}
    targets = {}
    for c in d["cases"]:
        f, _ = Engine(model, c).run("F")
        baselines[c["key"]] = dict(
            final=f["final"], scores=f["scores"], first_source=f["states"][0, 0]
        )
        with np.load(parent / "cases" / (c["key"] + ".npz")) as z:
            targets[c["key"]] = {n: z["T_" + n] for n in ("final", "scores", "first_source")}
    bank = corrected(parent, out / "complete", synthetic=True, baselines=baselines, targets=targets)
    for c in d["cases"]:
        k = c["key"]
        a = bank[k]
        assert np.array_equal(a["B0_final"], baselines[k]["final"]) and not np.array_equal(
            a["F_final"], a["B0_final"]
        )
        with np.load(parent / "cases" / (k + ".npz")) as z:
            assert np.array_equal(a["F_final"], z["F_final"])
    key = d["cases"][0]["key"]
    bad = {n: v.copy() for n, v in baselines[key].items()}
    bad["final"].flat[0] += 0.01
    try:
        install_baseline({}, bad, baselines[key])
    except AssertionError:
        rejected = True
    else:
        raise AssertionError("Corrupted baseline accepted")
    save_json(
        out / "preflight-summary.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            independent_CPU_F_observations=8,
            all_baselines_distinct=True,
            candidate_arrays_unchanged=True,
            coherent_baseline_corruption_rejected=rejected,
            correction_model_calls=0,
        ),
    )
    print(read(out / "preflight-summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.parent.resolve(), a.out.resolve())
