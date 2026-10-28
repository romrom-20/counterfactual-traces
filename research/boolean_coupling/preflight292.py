"""Tiny unequal-length complete pipeline and fail-closed native measurement gates."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import audit292, run292
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .preflight285 import cases
from .preflight286 import tiny


def main(out):
    assert not out.exists()
    out.mkdir()
    selected = [cases()[j] for j in (0, 9, 18, 31)]
    model = tiny()
    complete = out / "complete"
    run292.forecast(complete, model=model, cases=selected)
    audit292.qualification(complete, model=model)
    run292.seal(complete, model=model)
    rejected = []
    for tag in ("bounds", "qualification", "last_array", "receipt"):
        dest = out / ("rejected-" + tag)
        shutil.copytree(complete, dest)
        if tag == "bounds":
            p = dest / "design.json"
            d = read(p)
            d["bounds"]["head"] = 0.5
            p.write_text(json.dumps(d, indent=2) + "\n")
        if tag == "qualification":
            p = dest / "forecast-audit.json"
            a = read(p)
            a["status"] = "failed"
            p.write_text(json.dumps(a, indent=2) + "\n")
        if tag == "last_array":
            e = read(dest / "forecast-status.json")["cases"][-1]
            p = dest / "forecasts" / (e["key"] + ".npz")
            with p.open("ab") as f:
                f.write(b"coherent-gate-test")
        receipt = "synthetic292:" + sha(dest / "forecast-seal.json")
        if tag == "receipt":
            receipt += "wrong"
        try:
            run292.native(dest, receipt, "synthetic292", model=model)
        except AssertionError:
            assert not (dest / "native-started.json").exists() and not (dest / "native").exists()
            rejected.append(tag)
        else:
            raise AssertionError(("Native gate accepted corruption", tag))
    # Coherently update outer hashes after changing a primitive witness.
    dest = out / "rejected-primitive"
    shutil.copytree(
        complete,
        dest,
        ignore=shutil.ignore_patterns("forecast-audit.json", "forecast-seal.json", "*.csv"),
    )
    st = read(dest / "forecast-status.json")
    e = st["cases"][0]
    p = dest / "forecasts" / (e["key"] + ".npz")
    with np.load(p) as z:
        a = {n: z[n] for n in z.files}
    a["P_witness_phi"].flat[0] += 0.01
    np.savez_compressed(p, **a)
    m = read(p.with_suffix(".json"))
    m["arrays_sha256"] = sha(p)
    p.with_suffix(".json").write_text(json.dumps(m, indent=2) + "\n")
    e.update(
        arrays_sha256=sha(p), metadata_sha256=sha(p.with_suffix(".json")), bytes=p.stat().st_size
    )
    st["raw_bytes"] = sum(e["bytes"] for e in st["cases"])
    (dest / "forecast-status.json").write_text(json.dumps(st, indent=2) + "\n")
    try:
        audit292.qualification(dest, model=model)
    except AssertionError:
        assert read(dest / "forecast-audit.json")["status"] == "failed"
        assert not (dest / "native-started.json").exists()
        rejected.append("primitive")
    else:
        raise AssertionError("Coherent primitive tamper accepted")
    run292.native(
        complete,
        "synthetic292:" + sha(complete / "forecast-seal.json"),
        "synthetic292",
        model=model,
    )
    audit292.outcome(complete, model=model)
    result = dict(
        passed=True,
        synthetic=True,
        checkpoint_loaded=False,
        unequal_length_cases=4,
        cases_frozen_before_forecasts=True,
        forecast_audit=read(complete / "forecast-audit.json")["required_checks"],
        native_audit=read(complete / "native-audit.json")["required_checks"],
        rejected=rejected,
        scientific_outcomes_not_required=True,
    )
    save_json(out / "preflight-summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
