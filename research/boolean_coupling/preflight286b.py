"""Tiny full-tape comparison for the blockwise full-shape reverse calculation."""

import argparse
from pathlib import Path

import numpy as np

from . import audit286b, qualify286b
from .audit272 import read
from .cache272 import save_json
from .preflight286 import full_gradient, tiny


def main(out, reference, readout):
    assert not out.exists()
    out.mkdir()
    model = tiny()
    qualify286b.main(reference, readout, out / "complete", model=model)
    audit286b.main(out / "complete", reference, readout)
    checks = []
    for case in read(readout / "design.json")["cases"]:
        with np.load(reference / "forecasts" / (case["key"] + ".npz")) as data:
            mwrite = data["M_write"].copy()
        engine = qualify286b.FullShape(model, case)
        bases = {"F": engine.false_tail, "M": engine.first(mwrite)}
        for anchor, base in bases.items():
            blockwise = engine.gradient(base)
            full, scores, trace = full_gradient(
                model, case, base, anchor, engine.nodes["gate"], mwrite
            )
            errors = dict(
                source=float(np.max(abs(full.astype(float) - blockwise["source_trace"][0]))),
                source_trace=float(np.max(abs(trace.astype(float) - blockwise["source_trace"]))),
                scores=float(np.max(abs(scores.astype(float) - blockwise["scores"]))),
            )
            assert all(error <= 1e-4 for error in errors.values()), errors
            checks.append(dict(key=case["key"], anchor=anchor, **errors))
    save_json(out / "full-tape-checks.json", dict(synthetic=True, checks=checks))
    save_json(
        out / "preflight-summary.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            cases=len(checks) // 2,
            full_tape_vs_blockwise_source_adjoints=True,
            bound=0.0001,
            saved_array_audit_passed=True,
        ),
    )
    print("286b synthetic full-tape qualification passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-285-fixture", type=Path, required=True)
    parser.add_argument("--readout-286-fixture", type=Path, required=True)
    args = parser.parse_args()
    main(
        args.out.resolve(), args.reference_285_fixture.resolve(), args.readout_286_fixture.resolve()
    )
