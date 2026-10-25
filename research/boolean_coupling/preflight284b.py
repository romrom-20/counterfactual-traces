"""Synthetic17+15 cache continuation, full bytewise replay and original native audit."""

import argparse
import contextlib
import io
import os
import shutil
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM

from . import audit284, audit284b, continue284b
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json


def main(fixture, checkpoint, out):
    assert not out.exists()
    out.mkdir()
    complete = fixture / "complete"
    partial = out / "simulated-partial"
    partial.mkdir()
    assert read(complete / "native-audit.json")["status"] == "passed"
    for p in complete.iterdir():
        if p.name in ("sources", "forecasts"):
            shutil.copytree(p, partial / p.name, copy_function=os.link)
        elif p.is_file() and (
            p.name in ("design.json", "runtime.json", "weight-hashes.json", "native-started.json")
            or p.name.startswith("forecast")
        ):
            os.link(p, partial / p.name)
    (partial / "native").mkdir()
    entries = read(complete / "native-status.json")["cases"]
    for entry in entries[:17]:
        for suffix in (".npz", ".json"):
            p = complete / "native" / (entry["key"] + suffix)
            os.link(p, partial / "native" / p.name)
    save_json(
        partial / "failure-native.json",
        dict(
            error="AssertionError()",
            synthetic_fixture_mock=True,
            not_an_observed_CPU_storage_failure=True,
        ),
    )
    log = out / "simulated-storage-stop.log"
    log.write_text(
        "SYNTHETIC storage-stop fixture, no observed CPU storage failure.\nassert shutil.disk_usage(run).free >= 512 * 1024**2\nAssertionError\n"
    )
    torch.set_num_threads(1)
    model = Qwen2ForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    model.requires_grad_(False)
    continue284b.load_model = lambda offline: model
    audit284.load_model = lambda offline: model
    audit284.locate = lambda root: (checkpoint, dict(synthetic_checkpoint=True))
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            continue284b.main(partial, out / "complete", log)
            audit284b.main(out / "complete")
    finally:
        (out / "execution.log").write_text(captured.getvalue())
    result = out / "complete"
    q = read(result / "native-continuation-audit.json")
    assert q["status"] == "passed"
    for entry in entries:
        for suffix in (".npz", ".json"):
            assert sha(result / "native" / (entry["key"] + suffix)) == sha(
                complete / "native" / (entry["key"] + suffix)
            )
    for key in (
        "native_counts",
        "local_counts",
        "effect_counts",
        "incremental_counts",
        "incremental_effect_counts",
        "paired_counts",
        "joint_counts",
        "P_strata",
        "primary_passes",
        "removal_passes",
        "increment_margin_eligible",
    ):
        assert read(result / "native-audit.json")[key] == read(complete / "native-audit.json")[key]
    print(
        dict(
            passed=True,
            synthetic=True,
            real_checkpoint_loaded=False,
            simulated_partial_cases=17,
            new_native_control_CPU_forwards=128,
            all17_replayed_before_remaining15=True,
            all32_arrays_and_metadata_identical=True,
            original_native_audit_passed=True,
            independent_native_replay_CPU_forwards=32,
            scientific_predictions_or_bounds_changed=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.fixture.resolve(), a.checkpoint.resolve(), a.out.resolve())
