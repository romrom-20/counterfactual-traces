"""Frozen retrospective message interventions; no complete transformer forward."""

import argparse
import hashlib
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .cache272 import installed_sources, save_json
from .messages298 import BOUNDS, PRIMARY, ROOT, SOURCES, load_terminal, read_parent, run_case
from .publish285 import read, sha


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main(out, parent):
    assert not out.exists()
    assert shutil.disk_usage(out.parent).free >= 8 * 1024**3
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    assert len(hashes) == len(SOURCES) == 140 and len({Path(n).name for n in SOURCES}) == 140
    for n, h in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
            ).hexdigest()
            == h
        ), n
    cpu = read(ROOT.parent / "298-cpu/qualification.json")
    assert cpu["passed"] and cpu["source_hashes"] == hashes
    parents = read(ROOT / "research/boolean_coupling/parents298.json")
    assert parents["retrospective"] and len(parents["cases"]) == 8
    assert sha(parent / "design.json") == parents["source_design_sha256"]
    assert sha(parent / "native-status.json") == parents["source_native_status_sha256"]
    assert sha(parent / "audit.json") == parents["source_native_audit_sha256"]
    out.mkdir()
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    pd = read(parent / "design.json")
    assert installed_sources() == pd["installed_sources"], "Parent numerical runtime changed"
    save_json(
        out / "design.json",
        dict(
            experiment=298,
            retrospective=True,
            fresh_inputs=False,
            native_inputs_supplied=True,
            source_commit=commit,
            source_hashes=hashes,
            installed_sources=installed_sources(),
            cases=parents["cases"],
            primary=PRIMARY,
            bounds=BOUNDS,
            parent_source_sha256=sha(ROOT / "research/boolean_coupling/parents298.json"),
            checkpoint=pd["checkpoint"],
            parent_design_sha256=sha(parent / "design.json"),
            created_utc=stamp(),
            cpu_qualification_sha256=sha(ROOT.parent / "298-cpu/qualification.json"),
            storage_cap_bytes=64 * 1024**2,
            reserve_bytes=8 * 1024**3,
            boundary="Fixed known native terminal inputs, chosen position groups from posthoc readouts; no prospective counterfactual or learned-circuit claim.",
        ),
    )
    terminal = load_terminal()
    assert terminal.checkpoint == pd["checkpoint"]
    save_json(
        out / "runtime.json",
        dict(
            device=str(terminal.device),
            dtype=str(terminal.wo.dtype),
            torch=str(torch.__version__),
            terminal_layers_loaded=1,
            full_transformer_loaded=False,
            full_transformer_forwards=0,
            checkpoint=terminal.checkpoint,
        ),
    )
    entries, counts, total = [], Counter(), 0
    for selection in parents["cases"]:
        assert shutil.disk_usage(out).free >= 8 * 1024**3
        payload, checks, used, complete = run_case(
            terminal, read_parent(parent, selection), selection
        )
        counts.update(used)
        p = out / "cases" / (selection["key"] + ".npz")
        assert not p.exists()
        np.savez_compressed(p, **payload)
        total += p.stat().st_size
        assert total <= 64 * 1024**2
        save_json(
            p.with_suffix(".json"),
            dict(
                key=selection["key"],
                group=selection["group"],
                arrays_sha256=sha(p),
                checks=checks,
                counts=used,
                complete=complete,
                parent_hashes=selection["parent_hashes"],
            ),
        )
        entries.append(
            dict(
                key=selection["key"],
                arrays_sha256=sha(p),
                metadata_sha256=sha(p.with_suffix(".json")),
                bytes=p.stat().st_size,
            )
        )
        save_json(
            out / f"progress-{len(entries):04}.json",
            dict(cases=len(entries), counts=dict(counts), raw_bytes=total),
        )
        assert complete, (
            "Native calibration failed; first case is saved, further interventions withheld"
        )
        print("298 terminal", len(entries), "/8", flush=True)
    assert counts == dict(
        full_shape_output_projections=16, terminal_MLPs=160, full_vocabulary_heads=160
    ), dict(counts)
    save_json(
        out / "status.json",
        dict(
            status="complete",
            experiment=298,
            cases=entries,
            counts=dict(counts),
            raw_bytes=total,
            completed_utc=stamp(),
            full_transformer_forwards=0,
            prospective_native_forecasts=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.out.resolve(), args.parent.resolve())
    except Exception as error:
        if args.out.exists():
            save_json(
                args.out.resolve() / "failure.json",
                dict(error=repr(error), utc=stamp(), scientific_outcomes_withheld=True),
            )
        raise
