"""Unequal-length tiny CPU controls, all-coordinate FDs and tamper tests."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import audit288, run288
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .preflight286 import tiny
from .readout286 import margin
from .routing288 import ARMS, Routing
from .run286 import entry_data


def main(out, reference, previous, qualification):
    assert not out.exists()
    out.mkdir()
    run288.main(reference, previous, qualification, out / "complete", model=tiny())
    audit288.main(out / "complete", reference, previous, qualification)
    entries = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    all_coordinates = []
    # Qwen primitives retain internal Float32 casts even with double weights.
    # Use the declared Float32 FD steps/bounds; do not call this Float64 validation.
    model = tiny()
    for case in read(previous / "design.json")["cases"]:
        f = entry_data(reference, "forecasts", entries[case["key"]])
        engine = Routing(model, case, f["M_write"])
        for arm in ARMS:
            ad = engine.gradient(arm)["gradient"]
            tri = engine.triangular(arm)["gradient"]
            assert np.max(abs(ad - tri)) <= 1e-4
            for c in range(len(ad)):
                direction = np.zeros_like(ad)
                direction[c] = 1
                estimates = []
                for step in (0.05, 0.025):
                    p = engine.finite(arm, direction * step)["lifted_scores"]
                    m = engine.finite(arm, -direction * step)["lifted_scores"]
                    fd = (margin(p) - margin(m)) / (2 * step)
                    err, bound = abs(float(ad[c]) - fd), 0.001 + 0.02 * abs(float(ad[c]))
                    assert err <= bound, (case["key"], arm, c, err, bound)
                    estimates.append(fd)
                assert abs(estimates[0] - estimates[1]) <= 0.002 + 0.03 * abs(float(ad[c]))
                all_coordinates.append(dict(key=case["key"], arm=arm, coordinate=c,
                                            AD=float(ad[c]), FD=estimates,
                                            halving=abs(estimates[0] - estimates[1])))
        engine.unchanged()
    save_json(out / "all-coordinates.json", dict(
        synthetic=True, checkpoint_loaded=False, checks=all_coordinates,
        absolute_bound=0.001, relative_bound=0.02, steps=[0.05, 0.025],
        halving_absolute=0.002, halving_relative=0.03,
    ))
    rejected = []
    for tag in ("bounds", "FD", "required"):
        dest = out / ("rejected-" + tag)
        shutil.copytree(out / "complete", dest, ignore=shutil.ignore_patterns("audit.json", "*.csv"))
        if tag == "bounds":
            path = dest / "design.json"
            d = read(path)
            d["numerical_bounds"]["boundary_adjoint"] = 0.01
            path.write_text(json.dumps(d, indent=2) + "\n")
        else:
            status = read(dest / "status.json")
            e = status["cases"][0]
            path = dest / "cases" / (e["key"] + ".json")
            meta = read(path)
            if tag == "FD":
                meta["derivatives"][0]["AD"] += 0.1
            else:
                meta["checks"][0]["bound"] = 0.01
            path.write_text(json.dumps(meta, indent=2) + "\n")
            e["metadata_sha256"] = sha(path)
            (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        try:
            audit288.main(dest, reference, previous, qualification)
        except AssertionError:
            rejected.append(tag)
        else:
            raise AssertionError(("tamper accepted", tag))
        assert not (dest / "hypotheses.csv").exists()
    summary = dict(passed=True, synthetic=True, checkpoint_loaded=False,
                   unequal_length_cases=4, arms=8, all_coordinate_FD_checks=len(all_coordinates),
                   tamper_rejected=rejected, scientific_outcomes_not_required=True,
                   counts=read(out / "complete/status.json")["counts"])
    save_json(out / "preflight-summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("out", "reference-285-fixture", "previous-287-fixture", "qualification-287b-fixture"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve(), a.reference_285_fixture.resolve(), a.previous_287_fixture.resolve(),
         a.qualification_287b_fixture.resolve())
