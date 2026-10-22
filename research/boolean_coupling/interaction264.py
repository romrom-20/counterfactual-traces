"""All-case retrospective interaction of the completed263 normalization policies."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import checked, sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(value, target):
    size, wanted = (float(np.linalg.norm(v)) for v in (value, target))
    dot = float(np.vdot(value, target))
    return dict(norm=size, relative_target_norm=size/max(wanted, 1e-30),
        signed_target_share=dot/max(wanted*wanted, 1e-30),
        cosine=dot/max(size*wanted, 1e-30))


def main(run, out):
    assert not out.exists()
    audit, design, status = (read(run/name) for name in ("audit.json", "design.json", "status.json"))
    assert audit["experiment"] == 263 and audit["status"] == "passed"
    for name in ("design", "status"):
        assert sha(run/(name+".json")) == audit[name+"_sha256"]
    for name, digest in audit["csv_sha256"].items():
        assert sha(run/name) == digest
    out.mkdir(parents=True)
    checks, rows, margins, repairs, records = [], [], [], [], []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        key = case["key"]
        path = run/"cases"/(key+".npz")
        assert sha(path) == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert all(meta["statuses"][a]["status"] == "complete" for a in ("F", "T", "B", "L", "P", "D", "H", "E"))
        with np.load(path) as values:
            states = {a:values[a+"_trajectory"].astype(float) for a in ("F", "T", "L", "P", "D", "H", "E")}
            truth = states["T"]-states["F"]
            contrasts = dict(direction_alone=states["D"]-states["L"],
                direction_after_pattern=states["H"]-states["P"],
                pattern_alone=states["P"]-states["L"],
                pattern_after_direction=states["H"]-states["D"],
                interaction=states["H"]-states["P"]-states["D"]+states["L"],
                complete_after_energy=states["H"]-states["E"])
            checked(checks, "vector_factorial_identity", contrasts["direction_after_pattern"]-contrasts["direction_alone"], contrasts["interaction"], 1e-9)
            for layer in range(1, truth.shape[0]):
                for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                    target = contrasts["direction_after_pattern"][layer, pick]
                    for name, v in contrasts.items():
                        rows.append(dict(key=key, **case["group"], layer=layer, scope=scope,
                            contrast=name, **compare(v[layer, pick], target),
                            relative_native_norm=float(np.linalg.norm(v[layer, pick]))/max(float(np.linalg.norm(truth[layer, pick])), 1e-30)))
                    for arm in ("P", "E"):
                        old_error = states["T"][layer, pick]-states[arm][layer, pick]
                        new_error = states["T"][layer, pick]-states["H"][layer, pick]
                        correction = states["H"][layer, pick]-states[arm][layer, pick]
                        reduction = float(np.vdot(old_error, old_error)-np.vdot(new_error, new_error))
                        expected = float(2*np.vdot(old_error, correction)-np.vdot(correction, correction))
                        bound = 1e-9+1e-12*max(float(np.vdot(old_error, old_error)), 1.)
                        checked(checks, "squared_error_change_identity", reduction, expected, bound)
                        repairs.append(dict(key=key, **case["group"], layer=layer, scope=scope, background=arm,
                            squared_error_reduction=reduction, error_reduction_fraction=reduction/max(float(np.vdot(old_error, old_error)), 1e-30),
                            correction_norm=float(np.linalg.norm(correction)),
                            signed_error_projection=float(np.vdot(old_error, correction))/max(float(np.vdot(old_error, old_error)), 1e-30)))
            def margin(arm):
                return float(np.diff(values[arm+"_scores"].astype(float))[0])
            m = {a:margin(a) for a in ("T", "L", "P", "D", "H", "E")}
            margins.append(dict(key=key, **case["group"],
                direction_alone=m["D"]-m["L"], direction_after_pattern=m["H"]-m["P"],
                interaction=m["H"]-m["P"]-m["D"]+m["L"], complete_after_energy=m["H"]-m["E"],
                P_error=m["P"]-m["T"], E_error=m["E"]-m["T"], H_error=m["H"]-m["T"]))
            vectors = out/(key+".npz")
            np.savez_compressed(vectors, **{name:value[-1] for name, value in contrasts.items()})
            records.append(dict(key=key, input_arrays_sha256=entry["arrays_sha256"],
                input_metadata_sha256=entry["metadata_sha256"], vectors_sha256=sha(vectors)))
    for name, data in (("layerwise.csv", rows), ("margins.csv", margins), ("error-changes.csv", repairs)):
        table(out/name, data)
    write_json(out/"audit.json", dict(experiment=264, status="passed", retrospective=True,
        model_calls=0, new_native_measurements=0, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"docs/experiments/264-normalization-component-interaction.md"),
        input_audit_sha256=sha(run/"audit.json"), checks=checks, records=records,
        csv_sha256={name:sha(out/name) for name in ("layerwise.csv", "margins.csv", "error-changes.csv")},
        interpretation="Global own-state policy contrasts, not a same-input primitive decomposition, fresh test or learned circuit."))
    print(json.dumps(dict(status="passed", checks=len(checks), rows=len(rows), margins=margins), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
