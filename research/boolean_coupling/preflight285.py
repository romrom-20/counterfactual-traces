"""Full unequal-layout tiny CPU pipeline, including the global native gate.

This creates an explicitly synthetic 279-shaped donor bank, not model evidence.
It checks execution/qualification and never requires random tiny-model outcomes
or margin eligibility to pass scientific prediction criteria.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from . import audit285, run285
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .composition280 import digest
from .structure285 import ARMS


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cases():
    result = []
    for layout in range(4):
        for schema in ("generic", "descriptive"):
            for reverse in (0, 1):
                for target in (0, 1):
                    group = dict(
                        assignment=0,
                        block="synthetic-layout-" + str(layout),
                        reverse=reverse,
                        schema=schema,
                        target=target,
                    )
                    prefix = [
                        13,
                        18 + layout,
                        23 + int(schema == "descriptive"),
                        27 + reverse,
                        31 + target,
                    ]
                    prefix += [38 + layout] * layout
                    suffix = [42 + target] * (2 + int(schema == "descriptive")) + [48 + layout]
                    f, t = prefix + [895] + suffix, prefix + [830] + suffix
                    hashes = [
                        hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                        for ids in (f, t)
                    ]
                    result.append(
                        dict(
                            key=key(group),
                            group=group,
                            position=len(prefix),
                            false_ids=f,
                            true_ids=t,
                            input_hashes=hashes,
                        )
                    )
    assert len(result) == 32 and len({len(c["false_ids"]) for c in result}) > 1
    assert len({c["position"] for c in result}) == 4
    return result


def reference_bank(out, width):
    """Mock only the published forecast file schema, with explicit synthetic flags."""
    out.mkdir()
    (out / "forecasts").mkdir()
    rng = np.random.default_rng(285)
    old_cases, entries = [], []
    for assignment in (0, 1):
        for schema in ("generic", "descriptive"):
            for reverse in (0, 1):
                for target in (0, 1):
                    for block in ("synthetic-donor-a", "synthetic-donor-b"):
                        group = dict(
                            assignment=assignment,
                            block=block,
                            reverse=reverse,
                            schema=schema,
                            target=target,
                        )
                        donor = key(group)
                        false = rng.normal(0, 0.05, width).astype(np.float32)
                        cubic = (false.astype(float) + rng.normal(0, 0.1, width)).astype(np.float32)
                        path = out / "forecasts" / (donor + ".npz")
                        np.savez_compressed(path, false_write=false, writes_C=cubic)
                        save_json(
                            path.with_suffix(".json"),
                            dict(
                                key=donor, synthetic=True, arrays_sha256=sha(path), ordinary_true=0
                            ),
                        )
                        old_cases.append(dict(key=donor, group=group))
                        entries.append(
                            dict(
                                key=donor,
                                arrays_sha256=sha(path),
                                metadata_sha256=sha(path.with_suffix(".json")),
                                bytes=path.stat().st_size,
                            )
                        )
    save_json(out / "design.json", dict(experiment=279, synthetic=True, cases=old_cases))
    save_json(
        out / "forecast-status.json",
        dict(
            status="complete",
            ordinary_true=0,
            synthetic=True,
            cases=entries,
            raw_bytes=sum(e["bytes"] for e in entries),
        ),
    )
    save_json(
        out / "forecast-audit.json",
        dict(
            status="passed",
            synthetic=True,
            ordinary_true=0,
            design_sha256=sha(out / "design.json"),
            status_sha256=sha(out / "forecast-status.json"),
        ),
    )
    save_json(
        out / "forecast-seal.json",
        dict(
            experiment=279,
            synthetic=True,
            ordinary_true=0,
            cases=entries,
            design_sha256=sha(out / "design.json"),
            status_sha256=sha(out / "forecast-status.json"),
            audit_sha256=sha(out / "forecast-audit.json"),
        ),
    )
    indexed = {e["key"]: e for e in entries}
    groups = []
    for schema in ("generic", "descriptive"):
        for reverse in (0, 1):
            for target in (0, 1):
                group = dict(schema=schema, reverse=reverse, target=target, assignment=0)
                donors = [
                    indexed[c["key"]]
                    for c in old_cases
                    if all(c["group"][n] == value for n, value in group.items())
                ]
                assert len(donors) == 2
                groups.append(dict(group=group, donors=donors))
    return dict(
        experiment=285,
        reference_experiment=279,
        synthetic=True,
        match_fields=["schema", "reverse", "target", "assignment"],
        groups=groups,
        reference_hashes={
            n: sha(out / n)
            for n in (
                "design.json",
                "forecast-status.json",
                "forecast-audit.json",
                "forecast-seal.json",
            )
        },
        boundary="Explicit synthetic random-vector bank for CPU program qualification, not a checkpoint forecast or empirical279 finding.",
    )


def expect_rejected(call, run):
    try:
        call()
    except (AssertionError, FileNotFoundError):
        assert not (run / "native-started.json").exists() and not (run / "native").exists()
        return
    raise AssertionError("Native measurement was allowed without the required global seal")


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.manual_seed(285)
    torch.set_num_threads(1)
    cfg = Qwen2Config(
        vocab_size=1000,
        hidden_size=16,
        intermediate_size=23,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg).eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim == 2:
                p.normal_(0, 0.2)
    reference = out / "synthetic-reference279"
    bank = reference_bank(reference, cfg.hidden_size)
    run = out / "complete"
    run285.forecast(reference, run, model=model, cases=cases(), bankconfig=bank)
    expect_rejected(
        lambda: run285.native(
            run, reference, public_commit="premature", public_branch="synthetic285", model=model
        ),
        run,
    )
    audit285.forecast(run, reference, model=model)
    assert read(run / "forecast-audit.json")["status"] == "passed"
    expect_rejected(
        lambda: run285.native(
            run,
            reference,
            public_commit="synthetic285:wrong",
            public_branch="synthetic285",
            model=model,
        ),
        run,
    )
    # These are synthetic copies, kept as explicit rejected attempts. The
    # original complete forecast files remain untouched throughout the tests.
    for kind in ("audit", "csv", "array", "metadata"):
        damaged = out / ("rejected-tampered-" + kind)
        shutil.copytree(run, damaged)
        if kind == "audit":
            path = damaged / "forecast-audit.json"
        elif kind == "csv":
            filename = next(iter(read(run / "forecast-audit.json")["csv_sha256"]))
            path = damaged / filename
        else:
            first = read(run / "forecast-status.json")["cases"][0]["key"]
            path = damaged / "forecasts" / (first + (".npz" if kind == "array" else ".json"))
        with path.open("ab") as stream:
            stream.write(b" ")
        calls = []
        hook = model.register_forward_pre_hook(lambda *args: calls.append(True))
        try:
            expect_rejected(
                lambda: run285.native(
                    damaged,
                    reference,
                    public_commit="synthetic285:" + sha(damaged / "forecast-seal.json"),
                    public_branch="synthetic285",
                    model=model,
                ),
                damaged,
            )
            assert not calls, kind
        finally:
            hook.remove()
    seal = sha(run / "forecast-seal.json")
    run285.native(
        run,
        reference,
        public_commit="synthetic285:" + seal,
        public_branch="synthetic285",
        model=model,
    )
    audit285.native(run, reference, model=model)
    native = read(run / "native-audit.json")
    assert native["status"] == "passed" and native["ordinary_True_replay_forwards"] == 32
    assert read(run / "forecast-status.json")["counts"]["full_forwards"] == 224
    assert read(run / "native-status.json")["counts"]["full_forwards"] == 32
    for entry in read(run / "forecast-status.json")["cases"]:
        meta = read(run / "forecasts" / (entry["key"] + ".json"))
        assert set(meta["arms"]) == set(ARMS)
        assert meta["ordinary_true"] == meta["full_current_first_source_SiLU"] == 0
        assert meta["False_zero_exact"] and meta["all_prefixes_exact"]
        with np.load(run / "forecasts" / (entry["key"] + ".npz")) as saved:
            assert digest(saved["F_final"]) == digest(saved["FZ_final"])
    summary = dict(
        passed=True,
        synthetic=True,
        checkpoint_loaded=False,
        cases=32,
        unequal_lengths=True,
        four_source_positions=True,
        all_forecasts_before_native=True,
        premature_native_rejected=True,
        wrong_seal_rejected=True,
        corrupted_audit_csv_array_metadata_rejected_before_any_forward=True,
        source_True_activation_not_evaluated_for_forecasts=True,
        unhooked_True_audit_replays=32,
        outcome_passes_not_required=True,
        forecast_checks=read(run / "forecast-audit.json")["numerical_checks"],
        native_checks=native["numerical_checks"],
        full_forecasts=224,
        native_producer=32,
        audit_full_True=32,
    )
    save_json(out / "preflight-summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
