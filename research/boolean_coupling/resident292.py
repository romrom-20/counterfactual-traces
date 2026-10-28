"""Resource-only resident execution; unchanged 292 calculations and native gates."""

import argparse
import ctypes
import gc
import shutil
import time
from pathlib import Path

import torch

from . import audit292, run292
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .continue292 import main as continuation
from .joint292 import ROOT
from .run285 import release

MIB = 1024**2


def wait_space(root, minimum, timeout=180, stat=None, sleeper=None, clock=None):
    assert minimum >= 384 * MIB
    stat = stat or (lambda: shutil.disk_usage(root).free)
    sleeper = sleeper or time.sleep
    clock = clock or time.monotonic
    started = clock()
    samples = []
    while True:
        free = stat()
        samples.append(free)
        if free >= minimum:
            return dict(minimum=minimum, samples=samples, elapsed=clock() - started, passed=True)
        assert clock() - started < timeout, "Headroom did not recover; original guards not weakened"
        print("292c waiting for host headroom", free, "needed", minimum, flush=True)
        sleeper(5)


def pressure_relief(model):
    release(model)
    gc.collect()
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    relief = library.malloc_zone_pressure_relief
    relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    relief.restype = ctypes.c_size_t
    return int(relief(None, 0))


class Resident:
    def __init__(
        self, factory, root, minimum=640 * MIB, maintenance=pressure_relief, require_mps=True
    ):
        self.factory, self.root, self.minimum = factory, root, minimum
        self.maintenance, self.require_mps = maintenance, require_mps
        self.model = None
        self.loads = 0
        self.records = []

    @staticmethod
    def snapshot(model):
        return {
            n: (id(p), p._version, str(p.dtype), str(p.device), tuple(p.shape))
            for n, p in model.named_parameters()
        }

    def __call__(self, offline):
        assert offline is True
        if self.model is None:
            self.model = self.factory(offline)
            self.loads += 1
            assert not self.model.training and not any(
                p.requires_grad for p in self.model.parameters()
            )
            if self.require_mps:
                assert self.model.device.type == "mps" and self.model.dtype == torch.float32
            self.parameters = self.snapshot(self.model)
            released = self.maintenance(self.model)
            record = wait_space(self.root, self.minimum)
            record["allocator_bytes_released"] = released
            self.records.append(record)
        assert self.snapshot(self.model) == self.parameters
        return self.model


def main(original, out, receipt, timeout=3600):
    assert not out.exists()
    loader = Resident(run292.load_model, out.parent)
    old_run, old_audit = run292.load_model, audit292.load_model
    run292.load_model = audit292.load_model = loader
    try:
        continuation(original, out)
        assert loader.loads == 1
        pressure_relief(loader.model)
        wait_space(out, 400 * MIB)
        audit292.qualification(out)
        run292.seal(out)
        pressure_relief(loader.model)
        save_json(
            out / "resident-awaiting-receipt.json",
            dict(
                experiment="292c",
                ordinary_true=0,
                forecast_seal_sha256=sha(out / "forecast-seal.json"),
                model_loads=loader.loads,
                initial_headroom=loader.records,
                parameter_objects_versions_unchanged=True,
                executor_sha256=sha(Path(__file__)),
                protocol_sha256=sha(ROOT / "docs/experiments/292c-resident-execution.md"),
            ),
        )
        print("292c QUALIFIED_AND_SEALED; awaiting exact public receipt", str(receipt), flush=True)
        started = time.monotonic()
        while not receipt.exists():
            assert time.monotonic() - started < timeout, (
                "No public receipt; no native measurement authorized"
            )
            time.sleep(5)
        proof = read(receipt)
        remaining = 256 * MIB - read(out / "forecast-status.json")["raw_bytes"]
        wait_space(out, 384 * MIB + remaining)
        assert loader.snapshot(loader.model) == loader.parameters
        run292.native(out, proof["public_commit"], proof["public_branch"])
        pressure_relief(loader.model)
        audit292.outcome(out)
        assert loader.loads == 1 and loader.snapshot(loader.model) == loader.parameters
        save_json(
            out / "resident-complete.json",
            dict(
                experiment="292c",
                model_loads=1,
                parameter_objects_versions_unchanged=True,
                calculation_source_hashes=read(out / "design.json")["source_hashes"],
                native_gate_unchanged=True,
                executor_sha256=sha(Path(__file__)),
            ),
        )
        print("292c COMPLETE", flush=True)
    finally:
        run292.load_model, audit292.load_model = old_run, old_audit


def selftest(out):
    assert not out.exists()
    from .preflight286 import tiny

    calls = []
    model = tiny()
    loader = Resident(
        lambda offline: calls.append(offline) or model,
        out.parent,
        minimum=384 * MIB,
        maintenance=lambda m: 0,
        require_mps=False,
    )
    assert loader(True) is loader(True) is model and calls == [True] and loader.loads == 1
    readings = iter([300 * MIB, 383 * MIB, 640 * MIB])
    elapsed = [0]
    record = wait_space(
        out.parent,
        640 * MIB,
        stat=lambda: next(readings),
        sleeper=lambda n: elapsed.__setitem__(0, elapsed[0] + n),
        clock=lambda: elapsed[0],
    )
    assert record["samples"] == [300 * MIB, 383 * MIB, 640 * MIB] and record["elapsed"] == 10
    try:
        wait_space(
            out.parent,
            384 * MIB,
            timeout=5,
            stat=lambda: 0,
            sleeper=lambda n: elapsed.__setitem__(0, elapsed[0] + n),
            clock=lambda: elapsed[0],
        )
    except AssertionError:
        rejected = True
    else:
        raise AssertionError("Timeout ignored")
    out.mkdir()
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_loaded=False,
            one_factory_call=True,
            same_parameter_objects_versions=True,
            threshold_not_weakened=True,
            timeout_rejected=rejected,
            executor_sha256=sha(Path(__file__)),
        ),
    )
    print(read(out / "qualification.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--original", type=Path)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    try:
        if a.selftest:
            selftest(a.out.resolve())
        else:
            assert a.original is not None and a.receipt is not None
            main(a.original.resolve(), a.out.resolve(), a.receipt.resolve())
    except Exception as exc:
        if a.out.exists() and not (a.out / "failure-resident.json").exists():
            save_json(a.out / "failure-resident.json", dict(error=repr(exc)))
        raise
