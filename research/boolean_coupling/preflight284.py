"""Fresh unequal-length CPU fixture tests every globally separated284 stage."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import torch
from transformers import Qwen2ForCausalLM

from . import audit284, run284
from .audit229 import sha
from .audit272 import read
from .carried284 import input_hash


def main(fixture, out):
    assert not out.exists()
    out.mkdir()
    old = read(fixture / "complete/design.json")["cases"]
    cases = []
    for previous in old:
        c = json.loads(json.dumps(previous))
        c["group"]["block"] = "fresh-" + c["group"]["block"]
        for n in ("false_ids", "true_ids"):
            c[n][0] += 200
        c["key"] = hashlib.sha256(json.dumps(c["group"], sort_keys=True).encode()).hexdigest()
        c["input_hashes"] = [input_hash(c[n]) for n in ("false_ids", "true_ids")]
        cases.append(c)
    history = dict(
        hashes=[input_hash(c[n]) for c in old for n in ("false_ids", "true_ids")],
        provenance=[],
        synthetic=True,
    )
    checkpoint = fixture / "tiny-checkpoint"
    torch.set_num_threads(1)
    model = Qwen2ForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    model.requires_grad_(False)
    fake = dict(synthetic_checkpoint=True)
    for module in (run284, audit284):
        module.load_model = lambda offline: model
        module.locate = lambda root: (checkpoint, fake)
    run284.runtime = lambda current: dict(synthetic=True, device="cpu", layers=3)
    run = out / "complete"
    captured = io.StringIO()
    premature = False
    try:
        with contextlib.redirect_stdout(captured):
            run284.forecast(run, cases=cases, historical=history, synthetic=True)
            try:
                run284.native(run, dict(synthetic=True))
            except (FileNotFoundError, AssertionError):
                premature = True
            else:
                raise AssertionError("Native accepted unqualified/unsealed forecasts")
            assert not (run / "native-started.json").exists() and not (run / "native").exists()
            audit284.forecast(run)
            assert not (run / "native-started.json").exists()
            try:
                run284.native(run, dict(synthetic=True, forecast_seal_sha256="0" * 64))
            except AssertionError:
                pass
            else:
                raise AssertionError("Native accepted the wrong forecast seal")
            assert not (run / "native-started.json").exists()
            receipt = dict(
                synthetic=True,
                not_a_public_receipt=True,
                forecast_seal_sha256=sha(run / "forecast-seal.json"),
            )
            run284.native(run, receipt)
            audit284.native(run)
    finally:
        (out / "execution.log").write_text(captured.getvalue())
    fa, na, fs, ns = (
        read(run / (n + ".json"))
        for n in ("forecast-audit", "native-audit", "forecast-status", "native-status")
    )
    assert (
        fa["status"] == na["status"] == "passed"
        and fs["counts"]["full_forwards"] == 320
        and ns["counts"]["full_forwards"] == 128
    )
    assert (
        fa["ordinary_true"] == fa["full_current_source2_primal"] == 0
        and na["ordinary_True_replay_forwards"] == 32
    )
    assert (
        fa["staged_derivative_anchors"] == 32
        and fa["finite_head_maps"] == 320
        and na["finite_head_maps"] == 128
    )
    assert na["native_counts"]["T"] == 32 and na["joint_counts"]["U"] == na["joint_counts"]["C1"]
    print(
        dict(
            passed=True,
            synthetic=True,
            real_checkpoint_loaded=False,
            all_four_stages=True,
            forecast_CPU_forwards=320,
            native_CPU_forwards=128,
            independent_native_replay_CPU_forwards=32,
            premature_native_rejected=premature,
            wrong_seal_native_rejected=True,
            unequal_token_lengths=True,
            forecast_checks=fa["numerical_checks"],
            native_checks=na["numerical_checks"],
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.fixture.resolve(), a.out.resolve())
