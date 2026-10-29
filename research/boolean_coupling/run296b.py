"""Progress-path-only qualification; preserve frozen296 and replay two old cases."""

import argparse
import tempfile
from pathlib import Path

from . import run296 as original
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json

ROOT = original.ROOT
SOURCES = ("research/boolean_coupling/run296b.py", "docs/experiments/296b-append-only-progress.md")


def routed(path, value):
    if path.name in ("forecast-progress.json", "native-progress.json"):
        path = path.with_name(path.stem + f"-{value['cases']:02d}.json")
    save_json(path, value)


def same_bytes(a, b):
    assert sha(a) == sha(b), "Exact replay mismatch: " + a.name


def fixture(out):
    assert not out.exists()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        routed(p / "forecast-progress.json", dict(cases=1))
        routed(p / "forecast-progress.json", dict(cases=2))
        assert (
            not (p / "forecast-progress.json").exists()
            and read(p / "forecast-progress-01.json") == dict(cases=1)
            and read(p / "forecast-progress-02.json") == dict(cases=2)
        )
        routed(p / "native-progress.json", dict(cases=1))
        routed(p / "native-progress.json", dict(cases=2))
        routed(p / "other.json", dict(value=1))
        try:
            routed(p / "other.json", dict(value=2))
        except AssertionError:
            pass
        else:
            raise AssertionError("Unrelated immutable writer weakened")
        (p / "a.bin").write_bytes(b"original")
        (p / "b.bin").write_bytes(b"original")
        same_bytes(p / "a.bin", p / "b.bin")
        (p / "b.bin").write_bytes(b"changed")
        try:
            same_bytes(p / "a.bin", p / "b.bin")
        except AssertionError:
            pass
        else:
            raise AssertionError("Replay corruption not rejected")
    save_json(
        out,
        dict(
            passed=True,
            progress_payloads_preserved=4,
            unrelated_immutable_write_preserved=True,
            changed_replay_rejected=True,
            model_forwards=0,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
        ),
    )


def main(phase, run, old, qualification):
    import hashlib
    import subprocess

    q = read(qualification)
    assert q["passed"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for n in SOURCES:
        assert (
            q["source_hashes"][n]
            == sha(ROOT / n)
            == hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
        )
    original.save_json = routed
    if phase == "forecast":
        assert old is not None and not run.exists()
        od = read(old / "design.json")
        keys = [c["key"] for c in od["cases"][:2]]
        assert len(list((old / "forecasts").glob("*.npz"))) == 2 and not list(
            (old / "observations").glob("*")
        )
        assert (old / "failure-forecast.json").exists()
        proof = dict(
            method_unchanged=True,
            progress_paths_only=True,
            source_commit=commit,
            source_hashes=q["source_hashes"],
            original_design_sha256=sha(old / "design.json"),
            original_failure_sha256=sha(old / "failure-forecast.json"),
            original_completed_forecasts=16,
            replayed_cases=keys,
            fixture=q,
        )
        save_json(run.with_name(run.name + "-execution-before.json"), proof)
        original_save = original.save_case
        replayed = []

        def save_case(*args, **kwargs):
            result = original_save(*args, **kwargs)
            if args[1] == "forecast" and len(replayed) < 2:
                key = args[2]["key"]
                assert key == keys[len(replayed)]
                for folder in ["forecasts", "witnesses-forecast"]:
                    p = old / folder / (key + ".npz")
                    if p.exists():
                        same_bytes(p, run / folder / p.name)
                p = old / "forecasts" / (key + ".json")
                same_bytes(p, run / "forecasts" / p.name)
                replayed.append(key)
                if len(replayed) == 2:
                    save_json(
                        run / "exact-original-replay.json",
                        dict(
                            passed=True,
                            keys=replayed.copy(),
                            arrays_witness_metadata_exact=True,
                            before_third_case=True,
                        ),
                    )
            return result

        original.save_case = save_case
        original.forecast(run)
        assert replayed == keys
        save_json(run / "execution-qualification.json", proof)
    else:
        proof = read(run / "execution-qualification.json")
        assert (
            proof["source_hashes"] == q["source_hashes"]
            and read(run / "exact-original-replay.json")["passed"]
        )
        if phase == "seal":
            original.seal(run)
        else:
            original.native(run, old)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["fixture", "forecast", "seal", "native"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--original", type=Path)
    p.add_argument("--qualification", type=Path)
    a = p.parse_args()
    if a.phase == "fixture":
        fixture(a.out.resolve())
    else:
        main(
            a.phase,
            a.out.resolve(),
            None if a.original is None else a.original.resolve(),
            a.qualification.resolve(),
        )
