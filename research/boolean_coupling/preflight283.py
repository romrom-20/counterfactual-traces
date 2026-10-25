"""Whole283 producer/audit on the saved unequal-length tinyCPU reference only."""

import argparse
import contextlib
import io
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM

from . import audit283, run283
from .audit272 import read


def main(fixture, out):
    assert not out.exists()
    out.mkdir()
    reference, upstream = fixture / "complete", fixture / "reference"
    assert read(reference / "audit.json")["status"] == "passed"
    checkpoint = fixture / "tiny-checkpoint"
    torch.set_num_threads(1)
    model = Qwen2ForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    model.requires_grad_(False)
    fake = dict(synthetic_checkpoint=True)
    for module in (run283, audit283):
        module.load_model = lambda offline: model
        module.locate = lambda root: (checkpoint, fake)
    run283.runtime = lambda current: dict(synthetic=True, device="cpu", layers=3)
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            run283.main(reference, upstream, out / "complete")
            audit283.main(out / "complete", reference, upstream)
    finally:
        (out / "execution.log").write_text(capture.getvalue())
    run = out / "complete"
    status, audit = (read(run / (n + ".json")) for n in ("status", "audit"))
    assert status["status"] == "complete" and audit["status"] == "passed"
    assert len(status["cases"]) == 32 and status["counts"]["full_forwards"] == 448
    assert audit["staged_derivative_anchors"] == 32 and audit["False_anchors"] == 64
    assert audit["finite_head_maps"] == 448 and audit["public_corner_maps"] == 32
    assert len(audit["paired_counts"]) == 12 and audit["native_counts"]["T"] == 32
    for field in (
        "native",
        "local",
        "effect",
        "incremental",
        "incremental_effect",
        "paired",
        "joint",
    ):
        assert audit[field + "_counts"]["U"] == audit[field + "_counts"]["C1"]
    assert all(
        any(
            "reference" in a
            for a in read(run / "cases" / (e["key"] + ".json"))["array_aliases"].values()
        )
        for e in status["cases"]
    )
    print(
        dict(
            passed=True,
            synthetic=True,
            real_checkpoint_loaded=False,
            entire_producer_and_audit=True,
            new_CPU_full_forwards=448,
            finite_corner_maps=32,
            staged_derivative_anchors=32,
            numerical_checks=audit["numerical_checks"],
            unequal_receiving_lengths=True,
            archive_zero_full_exact=True,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.fixture.resolve(), a.out.resolve())
