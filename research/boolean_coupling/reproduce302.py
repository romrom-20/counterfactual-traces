"""Numerically repeat the retrospective 302 diagnostic using shared offline weights."""

import argparse
import gc
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import audit302, run302
from .alias301 import ROOT, sites
from .alias301 import SOURCES as PARENT_SOURCES
from .alias301 import execute as parent_execute
from .audit229 import sha
from .audit272 import read, table
from .audit301 import bundle, native_sources
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .preflight302 import main as tiny
from .run301 import save_case


def bootstrap(parent):
    """Generate only the eight old witness parents, not the original full study."""
    parent.mkdir()
    for name in ("forecasts", "observations", "witnesses-forecast", "witnesses-native"):
        (parent / name).mkdir()
    cases = [
        c
        for c in read(ROOT / "research/boolean_coupling/cases301.json")
        if c["group"]["reverse"] == c["group"]["target"] == 0
    ]
    _, checkpoint = locate(ROOT)
    model = load_model(True)
    head = dict(
        norm=array(model.model.norm.weight).astype(float),
        eps=np.array(model.config.rms_norm_eps),
        head=array(model.lm_head.weight[[895, 830]]).astype(float),
    )
    np.savez_compressed(parent / "head.npz", **head)
    maps = {str(j): array(model.model.layers[j].self_attn.o_proj.weight) for j in sites(model)}
    np.savez_compressed(parent / "output-maps.npz", **maps)
    save_json(
        parent / "design.json",
        dict(
            experiment=301,
            numerical_repetition=True,
            source_hashes={n: sha(ROOT / n) for n in PARENT_SOURCES},
            installed_sources=installed_sources(),
            checkpoint=checkpoint,
            cases=cases,
            fresh_confirmation=False,
        ),
    )
    checks, raw, fe, ne = [], [], [], []
    fc, nc = Counter(), Counter()
    for case in cases:
        assert shutil.disk_usage(parent).free >= 8 * 1024**3
        f, fw, ck, used, fm = parent_execute(model, case, "forecast")
        e = save_case(parent, "forecast", case, f, fw, ck, used, fm)
        fe.append(e)
        fc.update(used)
        n, nw, nk, counts, nm = parent_execute(model, case, "native", f, fm)
        ne.append(
            save_case(
                parent,
                "native",
                case,
                n,
                nw,
                nk,
                counts,
                nm,
                started="numerical-repetition-known-input",
                parent_entry=e,
            )
        )
        nc.update(counts)
        assert sum(p.stat().st_size for p in parent.rglob("*.npz")) <= 320 * 1024**2
        fm["checks"], nm["checks"] = ck, nk
        bundle(case, f, fw, fm, head, maps, "forecast", checks, raw, f)
        bundle(case, n, nw, nm, head, maps, "native", checks, raw, f)
        native_sources(case, f, n, fm, nm, checks)
    assert all(r["passes"] for r in checks), "Reconstructed parents failed qualification"
    save_json(
        parent / "forecast-status.json", dict(cases=fe, counts=dict(fc), numerical_repetition=True)
    )
    save_json(
        parent / "native-status.json", dict(cases=ne, counts=dict(nc), numerical_repetition=True)
    )
    table(parent / "checks.csv", checks)
    table(parent / "raw.csv", raw)
    save_json(
        parent / "audit.json",
        dict(
            status="passed",
            numerical_repetition=True,
            cases=8,
            full_forwards=112,
            forecast_counts=dict(fc),
            native_counts=dict(nc),
            required_checks=len(checks),
            raw_failed=sum(not r["passes"] for r in raw),
            no_original_global_seal_claim=True,
        ),
    )
    del model
    gc.collect()
    torch.mps.empty_cache()


def main(out, parent, create, tiny_fixture):
    assert not out.exists(), "Use a new output directory"
    if tiny_fixture:
        tiny(out)
        return
    assert shutil.disk_usage(out.parent).free >= 8 * 1024**3
    out.mkdir()
    # The frozen producer expects qualification beside ROOT. A new symlinked
    # checkout view changes only that output location, never sources or weights.
    checkout = out / "checkout"
    checkout.symlink_to(ROOT, target_is_directory=True)
    tiny(out / "302-cpu")
    if create:
        parent = out / "parents301"
        bootstrap(parent)
    assert parent is not None and (parent / "design.json").exists()
    run302.ROOT = audit302.ROOT = checkout
    run = out / "run"
    run302.calibrate(run, parent)
    audit302.main(run, parent, "calibration")
    run302.intervene(run, parent)
    audit302.main(run, parent, "complete")
    save_json(
        out / "repetition.json",
        dict(
            numerical_repetition=True,
            fresh_confirmation=False,
            original_native_inputs_already_public=True,
            parent_bootstrapped=create,
            full_checkpoint_forwards=496 if create else 384,
            shared_checkpoint=True,
            model_copies=0,
            tiny_forwards=140,
            retrospective=True,
            parent_counts=(read(parent / "audit.json") if create else None),
            bootstrap_extra_full_QK_softmax_PV_each=2240 if create else 0,
            bootstrap_witness_QK=288 if create else 0,
            bootstrap_extra_full_norms_and_heads_each=16 if create else 0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--bootstrap-parent", action="store_true")
    p.add_argument("--tiny-fixture", action="store_true")
    p.add_argument("--offline", action="store_true", help="Checkpoint loading is always offline")
    a = p.parse_args()
    if not a.tiny_fixture and (a.parent is not None) == a.bootstrap_parent:
        p.error("Choose --parent or --bootstrap-parent")
    main(
        a.out.absolute(),
        a.parent.resolve() if a.parent else None,
        a.bootstrap_parent,
        a.tiny_fixture,
    )
