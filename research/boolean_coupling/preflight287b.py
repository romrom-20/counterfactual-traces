"""Array-only acceptance/rejection tests for the narrower 287b numerical gate."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .cache272 import save_json
from .qualify287b import qualify, records


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(fixture, out):
    assert not out.exists()
    out.mkdir()
    _, boundary, fd, _, failures = qualify(fixture)
    assert not failures and all(r["passes"] for r in boundary + fd)
    rejected = []
    for kind in ("boundary", "full_shape_FD", "other_required"):
        target = out / kind
        shutil.copytree(fixture, target)
        audit = read(target / "audit.json")
        if kind in ("boundary", "full_shape_FD"):
            status = read(target / "status.json")
            entry = status["cases"][0]
            path = target / "cases" / (entry["key"] + ".npz")
            with np.load(path) as f:
                data = {k: f[k] for k in f.files}
            meta_path = path.with_suffix(".json")
            meta = read(meta_path)
            coordinate = (
                0
                if kind == "boundary"
                else int(
                    next(
                        r["direction"][11:]
                        for r in meta["derivatives"]
                        if r["direction"].startswith("coordinate_")
                    )
                )
            )
            data["FF_gradient"][coordinate] += 0.001 if kind == "boundary" else 0.1
            data["FF_source_trace"][0] = data["FF_gradient"]
            if kind == "full_shape_FD":
                data["FF_full_source_trace"][0, coordinate] = data["FF_gradient"][coordinate]
            np.savez_compressed(path, **data)
            meta["arrays_sha256"] = sha(path)
            write(meta_path, meta)
            entry["metadata_sha256"] = sha(meta_path)
            # The status binds both arrays and metadata independently.
            entry["arrays_sha256"] = sha(path)
            entry["bytes"] = path.stat().st_size
            status["raw_bytes"] = sum(e["bytes"] for e in status["cases"])
            write(target / "status.json", status)
            audit["status_sha256"] = sha(target / "status.json")
        else:
            rows = records(target / "checks.csv")
            rows[0].update(error="1.0", passes="False")
            table(target / "checks.csv", rows)
            audit.update(failed_checks=1, status="failed")
            audit["csv_sha256"]["checks.csv"] = sha(target / "checks.csv")
        write(target / "audit.json", audit)
        try:
            qualify(target)
        except AssertionError:
            rejected.append(kind)
        else:
            raise AssertionError(("Invalid fixture accepted", kind))
    save_json(
        out / "preflight-summary.json",
        dict(
            passed=True,
            synthetic=True,
            boundary_checks=len(boundary),
            full_shape_FD_checks=len(fd),
            rejected=rejected,
            model_calls=0,
            scientific_outcomes_not_read=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.fixture.resolve(), a.out.resolve())
