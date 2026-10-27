"""Unequal-length CPU qualification of all four crossed source/receiver states."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import audit287, run287
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .factor287 import CORNERS, Crossed, corners
from .preflight286 import full_gradient, tiny


def main(out, reference, readout, qualifier):
    assert not out.exists()
    out.mkdir()
    model = tiny()
    run287.main(reference, readout, qualifier, out / "complete", model=model)
    audit287.main(out / "complete", reference, readout, qualifier)
    checks = []
    for case in read(readout / "design.json")["cases"]:
        with np.load(reference / "forecasts" / (case["key"] + ".npz")) as data:
            mwrite = data["M_write"].copy()
        engine = Crossed(model, case)
        bases = corners(engine.false_tail, engine.first(mwrite))
        for corner in CORNERS:
            base = bases[corner]
            cached, scores, _, trace, _ = engine.gradient(base)
            full = engine.full_shape.gradient(base)
            tape, tape_scores, tape_trace = full_gradient(
                model, case, base, corner[0], engine.false_nodes["gate"], mwrite
            )
            errors = dict(
                source=float(np.max(abs(cached.astype(float) - tape))),
                source_trace=float(np.max(abs(trace.astype(float) - tape_trace))),
                blockwise_source_trace=float(
                    np.max(abs(full["source_trace"].astype(float) - tape_trace))
                ),
                scores=float(np.max(abs(scores.astype(float) - tape_scores))),
            )
            assert all(e <= 1e-4 for e in errors.values()), (corner, errors)
            checks.append(dict(key=case["key"], corner=corner, **errors))
        engine.unchanged()
    save_json(out / "full-tape-checks.json", dict(synthetic=True, checks=checks, bound=1e-4))
    # Audit rejections use coherent metadata/status hashes; no model call is made.
    rejected = []
    for tag in ("bounds", "FD"):
        dest = out / ("rejected-" + tag)
        shutil.copytree(
            out / "complete", dest, ignore=shutil.ignore_patterns("audit.json", "*.csv")
        )
        if tag == "bounds":
            path = dest / "design.json"
            value = read(path)
            value["numerical_bounds"]["source_adjoint"] = 0.01
            path.write_text(json.dumps(value, indent=2) + "\n")
        else:
            status = read(dest / "status.json")
            entry = status["cases"][0]
            path = dest / "cases" / (entry["key"] + ".json")
            meta = read(path)
            meta["derivatives"][0]["AD"] += 0.1
            path.write_text(json.dumps(meta, indent=2) + "\n")
            entry["metadata_sha256"] = sha(path)
            (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        try:
            audit287.main(dest, reference, readout, qualifier)
        except AssertionError:
            rejected.append(tag)
        else:
            raise AssertionError(("Tamper accepted", tag))
        assert not (dest / "audit.json").exists()
    status = read(out / "complete/status.json")
    assert status["counts"] == dict(
        full_F=4,
        first_block=4,
        suffix_forwards=224,
        backwards=16,
        full_shape_block_forwards=32,
        block_recomputations=32,
        block_VJPs=32,
        head_VJPs=16,
    )
    summary = dict(
        passed=True,
        synthetic=True,
        checkpoint_loaded=False,
        cases=4,
        corners=4,
        full_tape_cached_and_blockwise_all_source_coordinates=True,
        source_causality=True,
        FD_and_bound_tamper_rejected=rejected,
        scientific_outcomes_not_required=True,
        counts=status["counts"],
    )
    save_json(out / "preflight-summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-285-fixture", type=Path, required=True)
    p.add_argument("--readout-286-fixture", type=Path, required=True)
    p.add_argument("--qualifier-286b-fixture", type=Path, required=True)
    a = p.parse_args()
    main(
        a.out.resolve(),
        a.reference_285_fixture.resolve(),
        a.readout_286_fixture.resolve(),
        a.qualifier_286b_fixture.resolve(),
    )
