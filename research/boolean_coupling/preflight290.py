"""Tiny unequal-length finite-function/primitive checks with coherent negative fixtures."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import audit290, run290
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .preflight286 import tiny
from .readout286 import margin
from .receiver290 import ARMS, Receiver
from .run286 import entry_data


def main(out, reference, previous, head):
    assert not out.exists()
    out.mkdir()
    run290.main(reference, previous, head, out / "complete", model=tiny())
    audit290.main(out / "complete", reference, previous, head)
    entries = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    coordinates = []
    model = tiny()
    for case in read(previous / "design.json")["cases"]:
        f = entry_data(reference, "forecasts", entries[case["key"]])
        engine = Receiver(model, case, f["M_write"])
        for arm in ARMS:
            gradient = engine.gradient(arm)["gradient"]
            for c in range(len(gradient)):
                direction = np.zeros(len(gradient))
                direction[c] = 1.0
                est = []
                for step in (0.05, 0.025):
                    plus = engine.finite(arm, step * direction)["raw_scores"]
                    minus = engine.finite(arm, -step * direction)["raw_scores"]
                    fd = (margin(plus) - margin(minus)) / (2 * step)
                    assert abs(float(gradient[c]) - fd) <= 0.001 + 0.02 * abs(float(gradient[c]))
                    est.append(fd)
                assert abs(est[0] - est[1]) <= 0.002 + 0.03 * abs(float(gradient[c]))
                coordinates.append(
                    dict(key=case["key"], arm=arm, coordinate=c, AD=float(gradient[c]), FD=est)
                )
        engine.unchanged()
    save_json(
        out / "all-coordinates.json",
        dict(
            synthetic=True,
            checkpoint_loaded=False,
            checks=coordinates,
            steps=[0.05, 0.025],
            AD_absolute=0.001,
            AD_relative=0.02,
            halving_absolute=0.002,
            halving_relative=0.03,
        ),
    )
    rejected = []
    for tag in ("bounds", "FD", "primitive"):
        dest = out / ("rejected-" + tag)
        shutil.copytree(
            out / "complete", dest, ignore=shutil.ignore_patterns("audit.json", "*.csv")
        )
        if tag == "bounds":
            path = dest / "design.json"
            d = read(path)
            d["numerical_bounds"]["primitive"] = 0.01
            path.write_text(json.dumps(d, indent=2) + "\n")
        else:
            st = read(dest / "status.json")
            e = st["cases"][0]
            path = dest / "cases" / (e["key"] + ".json")
            m = read(path)
            if tag == "FD":
                m["derivatives"][0]["AD"] += 0.1
            else:
                m["primitive_checks"][0]["bound"] = 0.01
            path.write_text(json.dumps(m, indent=2) + "\n")
            e["metadata_sha256"] = sha(path)
            (dest / "status.json").write_text(json.dumps(st, indent=2) + "\n")
        try:
            audit290.main(dest, reference, previous, head)
        except AssertionError:
            rejected.append(tag)
        else:
            raise AssertionError(("Tamper accepted", tag))
        assert not (dest / "hypotheses.csv").exists()
    result = dict(
        passed=True,
        synthetic=True,
        checkpoint_loaded=False,
        unequal_length_cases=4,
        arms=len(ARMS),
        all_coordinate_FD_checks=len(coordinates),
        tamper_rejected=rejected,
        scientific_outcomes_not_required=True,
        counts=read(out / "complete/status.json")["counts"],
    )
    save_json(out / "preflight-summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("out", "reference-285-fixture", "previous-289-fixture", "head-287-fixture"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    main(
        a.out.resolve(),
        a.reference_285_fixture.resolve(),
        a.previous_289_fixture.resolve(),
        a.head_287_fixture.resolve(),
    )
