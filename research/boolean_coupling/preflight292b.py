"""Qualify first-case resource replay and coherent mismatch rejection on tiny CPU."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .continue292 import main as continuation
from .preflight286 import tiny


def stopped_fixture(complete, dest):
    dest.mkdir()
    for n in ("design.json", "runtime.json", "head.json", "head.npz"):
        shutil.copy2(complete / n, dest / n)
    shutil.copytree(complete / "sources", dest / "sources")
    (dest / "forecasts").mkdir()
    e = read(complete / "forecast-status.json")["cases"][0]
    for ext in (".npz", ".json"):
        shutil.copy2(
            complete / "forecasts" / (e["key"] + ext), dest / "forecasts" / (e["key"] + ext)
        )
    save_json(
        dest / "failure-forecast.json",
        dict(
            synthetic=True,
            error="Simulated stopped prefix for executor qualification, not a real resource claim.",
        ),
    )
    return e


def main(complete, out):
    assert not out.exists()
    out.mkdir()
    old = out / "stopped-fixture"
    first = stopped_fixture(complete, old)
    continuation(old, out / "replay", model=tiny())
    proof = read(out / "replay/continuation.json")
    assert proof["first_case_replayed_before_any_remaining_case"]
    assert read(out / "replay/forecast-status.json")["ordinary_true"] == 0
    bad = out / "mismatch-fixture"
    stopped_fixture(complete, bad)
    p = bad / "forecasts" / (first["key"] + ".npz")
    with np.load(p) as z:
        data = {n: z[n] for n in z.files}
    data["P_witness_phi"].flat[0] += 0.01
    np.savez_compressed(p, **data)
    meta = read(p.with_suffix(".json"))
    meta["arrays_sha256"] = sha(p)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    try:
        continuation(bad, out / "rejected-replay", model=tiny())
    except AssertionError:
        assert not list((out / "rejected-replay/forecasts").glob("*.npz"))
        assert not (out / "rejected-replay/forecast-status.json").exists()
        assert not (out / "rejected-replay/native-started.json").exists()
    else:
        raise AssertionError("Corrupted replay accepted")
    save_json(
        out / "preflight-summary.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            full_cases=4,
            first_case_arrays=proof["replayed_arrays"],
            coherent_replay_mismatch_rejected_before_archive=True,
            ordinary_true=0,
            executor_sha256=sha(Path(__file__).with_name("continue292.py")),
            preflight_source_sha256=sha(Path(__file__)),
        ),
    )
    print(read(out / "preflight-summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--complete-fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.complete_fixture.resolve(), a.out.resolve())
