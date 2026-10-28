"""Explicit four-literal resource qualification; immutable293 science and replay."""

import argparse
import ast
import hashlib
import types
from pathlib import Path

import numpy as np

from . import audit293
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .subspace293 import ROOT


def modules():
    runfile = ROOT / "research/boolean_coupling/run293.py"
    auditfile = ROOT / "research/boolean_coupling/audit293.py"

    def clone(path, changes, expected):
        source = path.read_text()
        modified = source
        for old, new in changes:
            assert modified.count(old) == 1, (path, old)
            modified = modified.replace(old, new)
        before = ast.parse(source)
        after = ast.parse(modified)
        bn = [n for n in ast.walk(before) if isinstance(n, ast.Constant)]
        an = [n for n in ast.walk(after) if isinstance(n, ast.Constant)]
        assert len(bn) == len(an)
        differences = [
            (x.value, y.value) for x, y in zip(bn, an, strict=True) if x.value != y.value
        ]
        assert differences == [(256, 384)] * expected, differences
        for x, y in zip(bn, an, strict=True):
            y.value = x.value
        assert ast.dump(before, include_attributes=False) == ast.dump(
            after, include_attributes=False
        )
        m = types.ModuleType("boolean_coupling.resource293b_" + path.stem)
        m.__file__ = str(path)
        m.__package__ = "boolean_coupling"
        exec(compile(modified, str(path) + ":293b-resource-only", "exec"), m.__dict__)
        return m, dict(
            original=sha(path),
            executed=hashlib.sha256(modified.encode()).hexdigest(),
            resource_literals=expected,
        )

    run, rs = clone(
        runfile,
        [
            ('d["guards_MiB"] == [512, 384, 256]', 'd["guards_MiB"] == [512, 384, 384]'),
            ("guards_MiB=[512, 384, 256]", "guards_MiB=[512, 384, 384]"),
            ("total <= 256 * 1024**2", "total <= 384 * 1024**2"),
        ],
        3,
    )
    audit, ass = clone(auditfile, [("<= 256 * 1024**2", "<= 384 * 1024**2")], 1)
    audit.binding = run.binding
    return run, audit, dict(run=rs, audit=ass)


def replay(original, key, data, metadata):
    p = original / "cases" / (key + ".npz")
    if not p.exists():
        return 0
    old = read(p.with_suffix(".json"))
    assert sha(p) == old["arrays_sha256"]
    with np.load(p) as z:
        assert set(z.files) == set(data)
        for n in z.files:
            assert (
                z[n].shape == data[n].shape
                and z[n].dtype == data[n].dtype
                and np.array_equal(z[n], data[n])
            ), ("Replay mismatch", key, n)
        count = len(z.files)
    assert {k: v for k, v in old.items() if k not in ("key", "arrays_sha256")} == metadata
    return count


def main(original, out, reference=None, **kwargs):
    assert not out.exists()
    module, audit, proof = modules()
    oldarchive = module.archive
    oldaudit = audit293.main
    oldcases = [p.stem for p in (original / "cases").glob("*.npz")]
    replayed = {}
    assert len(oldcases) in (7, 8)

    def archive(folder, key, data, metadata):
        count = replay(original, key, data, metadata)
        if count:
            replayed[key] = count
        else:
            assert set(replayed) == set(oldcases), (
                "Remaining original cases must replay before new case"
            )
        return oldarchive(folder, key, data, metadata)

    module.archive = archive
    audit293.main = audit.main
    try:
        model = module.main(out, reference, **kwargs)
        assert set(replayed) == set(oldcases)
        save_json(
            out / "resource-qualification.json",
            dict(
                experiment="293b",
                method_unchanged=True,
                ordinary_checkpoint_true=0,
                original_design_sha256=sha(original / "design.json"),
                original_failure_sha256=sha(original / "failure.json")
                if (original / "failure.json").exists()
                else None,
                replayed_arrays=replayed,
                original_total_cap_MiB=256,
                qualified_total_cap_MiB=384,
                initial_and_percase_free_MiB=[512, 384],
                compiled_sources=proof,
                executor_sha256=sha(Path(__file__)),
                protocol_sha256=sha(ROOT / "docs/experiments/293b-resource-continuation.md"),
            ),
        )
        return model
    finally:
        audit293.main = oldaudit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    try:
        main(a.original.resolve(), a.out.resolve(), a.reference.resolve())
    except Exception as e:
        if a.out.exists() and not (a.out / "failure-resource.json").exists():
            save_json(a.out / "failure-resource.json", dict(error=repr(e)))
        raise
