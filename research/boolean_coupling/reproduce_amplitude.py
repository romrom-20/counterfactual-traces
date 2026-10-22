"""Local published-literal269 replay; never push Git or claim fresh confirmation."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def native_replay(run):
    import numpy as np
    import torch

    from boolean_coupling.audit229 import sha
    from boolean_coupling.core225b import array, load_model, observe
    from boolean_coupling.fields267 import digest, emit, read
    from boolean_coupling.forecast269 import sources_check, stamp
    from boolean_coupling.mlp258 import runtime

    design = sources_check(run)
    assert not (run/"native-started.json").exists() and not (run/"forecast-seal.json").exists()
    assert read(run/"forecast-audit.json")["status"] == "passed"
    emit(run, run/"replication-scope.json", dict(published_literal_replay=True, fresh_confirmation=False,
        public_receipt=False, source_sha256=sha(__file__), limitation="Local numerical replication; does not satisfy original fresh-run remote-commit guard."))
    emit(run, run/"forecast-seal.json", dict(experiment=269, globally_pre_native=False,
        ordinary_true=0, created=stamp(), scope="local published-literal numerical replication",
        design_sha256=sha(run/"design.json"), status_sha256=sha(run/"forecast-status.json"),
        audit_sha256=sha(run/"forecast-audit.json"), cases=read(run/"forecast-status.json")["cases"]))
    emit(run, run/"native-started.json", dict(started=stamp(), public_commit=None, ordinary_true_before=0,
        local_replication=True, forecast_seal_sha256=sha(run/"forecast-seal.json"),
        scope_sha256=sha(run/"replication-scope.json")))
    (run/"observations").mkdir()
    model = load_model(True)
    assert runtime(model) == read(run/"runtime.json")
    assert shutil.disk_usage(run).free >= 512*1024**2
    entries = []
    for index, case in enumerate(design["cases"], 1):
        start, key = case["position"], case["key"]
        states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        path = run/"observations"/(key+".npz")
        final, score = array(states[-1][0, start:]), array(scores)
        assert np.isfinite(final).all() and np.isfinite(score).all()
        np.savez_compressed(path, T_final=final, T_scores=score)
        records = []
        for layer, state in enumerate(states, 1):
            value = array(state[0])
            records.append(dict(layer=layer, block_digest=digest(value[start+1:]),
                source_digest=digest(value[start]), prefix_digest=digest(value[:start])))
        emit(run, path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), records=records, measured=stamp(), local_replication=True))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        assert shutil.disk_usage(run).free >= 512*1024**2
        print("Local269 numerical replay native", index, "/32", flush=True)
    emit(run, run/"native-status.json", dict(status="complete", cases=entries, ordinary_true=32,
        forecast_seal_sha256=sha(run/"forecast-seal.json"), native_started_sha256=sha(run/"native-started.json"),
        ended=stamp(), local_replication=True))


def main(args):
    if args.verify_only:
        subprocess.run([sys.executable, "-S", str(ROOT/"research/boolean_coupling/verify_amplitude.py")], check=True)
        return
    assert args.out is not None
    out = args.out.resolve()
    assert not out.exists(), "Use a new output directory; preserve all previous arrays/failures."
    assert not out.is_relative_to(ROOT/"results") and not ROOT.is_relative_to(out)
    run = out/"experiment269"
    previous = ROOT/"results/amplitude-direction-268-269/amplitude-direction-268"
    commands = [
        [sys.executable, "-m", "boolean_coupling.forecast269", "forecast", "--previous", str(previous), "--run", str(run)],
        [sys.executable, "-m", "boolean_coupling.audit269", "pre", "--run", str(run)],
        [sys.executable, str(Path(__file__).resolve()), "--native-local", str(run)],
        [sys.executable, "-m", "boolean_coupling.audit269", "final", "--run", str(run)],
    ]
    plan = dict(scope="Published-literal local numerical replication, not fresh confirmation; no Git push.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), commands=commands,
        complete_model_forwards=320, requires="Pinned MPS environment and cached checkpoint; no prior activation arrays.")
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True)
    (out/"local-replication-plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT/"research"), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}
    for index, command in enumerate(commands, 1):
        with (out/f"stage{index}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Completed local269 numerical replay; inspect every outcome and original-bound raw failure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--native-local", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.native_local is not None:
        native_replay(args.native_local.resolve())
    else:
        main(args)
