"""Posthoc local pipeline-repeat and exact first-block checks; no model calls."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .audit229 import sha
from .core225b import write_json

ROOT = Path(__file__).resolve().parents[2]
NAMES = {251:"fixed-receiver-attention-251", 252:"source-driven-attention-252"}


def read(path):
    return json.loads(path.read_text())


def main(originals, replica, out):
    assert not out.exists()
    pipeline = read(replica/"pipeline.json")
    assert pipeline["status"] == "complete" and pipeline["wrapper_sha256"] == sha(ROOT/"research/boolean_coupling/reproduce_attention.py")
    records, first_blocks, array_count = [], [], 0
    for num, original in originals.items():
        repeated = replica/f"experiment{num}"
        assert read(original/"audit.json")["status"] == read(repeated/"audit.json")["status"] == "passed"
        assert read(original/"runtime.json") == read(repeated/"runtime.json")
        for phase, status_name in (("forecasts", "forecast-status.json"), ("observations", "status.json")):
            for old, new in zip(read(original/status_name)["cases"], read(repeated/status_name)["cases"], strict=True):
                key = old["key"]
                assert key == new["key"]
                op, rp = original/phase/(key+".npz"), repeated/phase/(key+".npz")
                assert sha(op) == old["arrays_sha256"] and sha(rp) == new["arrays_sha256"]
                with np.load(op) as a, np.load(rp) as b:
                    assert set(a.files) == set(b.files)
                    for name in a.files:
                        assert np.array_equal(a[name], b[name]), (num, key, name)
                        array_count += 1
                    records.append(dict(experiment=num, phase=phase, key=key, arrays=len(a.files), original_sha256=sha(op), repeated_sha256=sha(rp), max_error=0.))
        if num == 252:
            for entry in read(original/"forecast-status.json")["cases"]:
                key = entry["key"]
                with np.load(original/"forecasts"/(key+".npz")) as f, np.load(original/"observations"/(key+".npz")) as t:
                    for arm in ("A", "Q", "K"):
                        assert np.array_equal(f[arm+"_trajectory"][0], t["T_trajectory"][0])
                        first_blocks.append(dict(key=key, arm=arm, max_error=0.))
    out.mkdir(parents=True)
    write_json(out/"validation.json", dict(status="passed", posthoc=True, model_calls=0, local_same_team_replication=True,
        source_sha256=sha(__file__), pipeline_sha256=sha(replica/"pipeline.json"), arrays_compared=array_count, records=records,
        first_block_252=first_blocks, original_audits={str(n):sha(p/"audit.json") for n, p in originals.items()},
        interpretation="Local repeatability and exact initial receiving state; not independent replication or a stronger raw-logit accuracy claim."))
    shutil.copy2(replica/"pipeline.json", out/"pipeline.json")
    (out/"README.md").write_text("# Local end-to-end pipeline validation\n\nThe published reproduction wrapper completed both model experiments and their audits again, using explicit local seals. Every saved numeric array matches the original runs exactly. This is same-team local repeatability, not independent replication. Raw-logit absolute failures remain failed.\n\nThe posthoc first-block check also verifies that every252 A/Q/K forecast starts with the exact native True first-block tail, before its later feedback policy diverges. No model calls were used to compare the saved arrays. See `validation.json` for all case hashes/checks and `pipeline.json` for the executed commands.\n")
    for name in NAMES.values():
        manifest_path = ROOT/"results"/name/"manifest.json"
        manifest = read(manifest_path)
        for path in [Path(__file__), *out.iterdir()]:
            manifest["files"][str(path.relative_to(ROOT))] = sha(path)
        write_json(manifest_path, manifest)
    print(f"Local pipeline repeat: {array_count} arrays identical;96 exact252 first-block checks; no comparison model calls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-251", "input-252", "replica", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    main({251:args.input_251.resolve(), 252:args.input_252.resolve()}, args.replica.resolve(), args.out.resolve())
