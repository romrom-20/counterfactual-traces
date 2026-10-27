"""Full-sequence-shape, blockwise reverse-mode check of 286 cached adjoints.

This is a numerical qualification on the already observed cohort, not a new
forecast. Causal prefix states are constant; only the suffix carries adjoints.
Every block is recomputed at the original full sequence shape. Keeping one
block's reverse tape at a time avoids a full-model full-sequence reverse tape.
"""

import argparse
import gc
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .core225b import array, load_model
from .readout286 import ROOT, release, source_write
from .readout286 import SOURCES as PREVIOUS
from .run272 import stamp
from .run285 import model_signature, runtime
from .run286 import entry_data

SOURCES = (
    *PREVIOUS,
    "research/boolean_coupling/qualify286b.py",
    "research/boolean_coupling/audit286b.py",
    "research/boolean_coupling/preflight286b.py",
    "docs/experiments/286b-full-shape-adjoints.md",
)
BOUNDS = dict(full_replay=1e-4, source_adjoint=1e-4, projected_response=1e-4)


class FullShape:
    def __init__(self, model, case):
        self.model, self.case = model, case
        self.start, self.total = case["position"], len(case["false_ids"])
        self.inputs, self.context, self.nodes = [], {}, {}
        handles = []

        def context(module, args, kwargs):
            self.context = {
                "attention_mask": kwargs["attention_mask"].detach().clone(),
                "position_ids": kwargs["position_ids"].detach().clone(),
                "cache_position": kwargs["cache_position"].detach().clone(),
                "position_embeddings": tuple(
                    v.detach().clone() for v in kwargs["position_embeddings"]
                ),
                "use_cache": False,
            }

        def state(module, args):
            self.inputs.append(args[0].detach().clone())

        def last(module, args, output):
            self.false_last = output.detach().clone()
            self.false_final = array(output[0, self.start :])

        def node(name):
            def hook(module, args, output):
                self.nodes[name] = array(output[0, self.start])

            return hook

        handles.append(model.model.layers[0].register_forward_pre_hook(context, with_kwargs=True))
        handles.extend(layer.register_forward_pre_hook(state) for layer in model.model.layers)
        handles.append(model.model.layers[-1].register_forward_hook(last))
        handles.extend(
            module.register_forward_hook(node(name))
            for module, name in (
                (model.model.layers[0].mlp.gate_proj, "gate"),
                (model.model.layers[0].mlp.act_fn, "phi"),
            )
        )
        ids = torch.tensor([case["false_ids"]], device=model.device)
        try:
            with torch.no_grad():
                result = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=1,
                    position_ids=torch.arange(self.total, device=model.device)[None],
                    cache_position=torch.arange(self.total, device=model.device),
                )
            self.false_scores = array(result.logits[0, -1, [895, 830]])
        finally:
            for handle in handles:
                handle.remove()
        assert len(self.inputs) == len(model.model.layers)
        self.false_tail = self.inputs[1][:, self.start :].detach().clone()
        self.forward_calls = self.reverse_calls = self.head_reverse_calls = 0

    def first(self, write):
        ids = torch.tensor([self.case["true_ids"]], device=self.model.device)
        with (
            torch.no_grad(),
            source_write(
                self.model, self.start, self.nodes["gate"], write, self.nodes["phi"]
            ) as witness,
        ):
            result = self.model.model.layers[0](self.model.model.embed_tokens(ids), **self.context)
        assert witness["activation_calls"] == 1 and witness["False_primal_exact"]
        self.first_policy = witness
        return result[:, self.start :].detach().clone()

    def block(self, index, tail):
        full = torch.cat((self.inputs[index][:, : self.start], tail), dim=1)
        return self.model.model.layers[index](full, **self.context)

    def gradient(self, base):
        tails, prefix_errors = [base.detach().clone()], []
        with torch.no_grad():
            for index in range(1, len(self.inputs)):
                output = self.block(index, tails[-1])
                self.forward_calls += 1
                expected = (
                    self.inputs[index + 1] if index + 1 < len(self.inputs) else self.false_last
                )
                prefix_errors.append(
                    float((output[:, : self.start] - expected[:, : self.start]).abs().max().cpu())
                )
                tails.append(output[:, self.start :].detach().clone())
        with torch.enable_grad():
            point = tails[-1].detach().clone().requires_grad_(True)
            scores = self.model.lm_head(self.model.model.norm(point)[:, -1:])[0, -1, [895, 830]]
            adjoint = torch.autograd.grad(scores[1] - scores[0], point)[0].detach()
        self.head_reverse_calls += 1
        saved_scores, final = array(scores), array(tails[-1][0])
        traces, norms, reverse_errors = [], [], []
        for index in reversed(range(1, len(self.inputs))):
            with torch.enable_grad():
                point = tails[index - 1].detach().clone().requires_grad_(True)
                output = self.block(index, point)[:, self.start :]
                reverse_errors.append(float((output.detach() - tails[index]).abs().max().cpu()))
                adjoint = torch.autograd.grad(output, point, grad_outputs=adjoint)[0].detach()
            self.reverse_calls += 1
            traces.append(array(adjoint[0, 0]))
            norms.append(float(torch.linalg.vector_norm(adjoint).cpu()))
            del point, output
            release(self.model)
        return dict(
            source_trace=np.stack(traces[::-1]),
            tail_norm_trace=np.asarray(norms[::-1]),
            final=final,
            scores=saved_scores,
            prefix_errors=np.asarray(prefix_errors),
            reverse_primal_errors=np.asarray(reverse_errors[::-1]),
        )


def main(reference, readout, out, model=None):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 768 * 1024**2
    old, design = read(reference / "design.json"), read(readout / "design.json")
    assert old["experiment"] == 285 and design["experiment"] == 286
    assert read(readout / "audit.json")["status"] == "passed"
    assert old["installed_sources"] == design["installed_sources"] == installed_sources()
    synthetic = model is not None
    assert synthetic is old["synthetic"] is design["synthetic"]
    if synthetic:
        assert model_signature(model) == old["checkpoint"]
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == h
    out.mkdir()
    (out / "cases").mkdir()
    (out / "sources").mkdir()
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in hashes:
        shutil.copy2(ROOT / name, out / "sources" / Path(name).name)
    save_json(
        out / "design.json",
        dict(
            experiment="286b",
            numerical_only=True,
            retrospective=True,
            synthetic=synthetic,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=hashes,
            installed_sources=installed_sources(),
            reference_hashes={
                "285_design": sha(reference / "design.json"),
                "286_design": sha(readout / "design.json"),
                "286_status": sha(readout / "status.json"),
                "286_audit": sha(readout / "audit.json"),
            },
            cases=design["cases"],
            checkpoint=old["checkpoint"],
            numerical_bounds=BOUNDS,
            began=stamp(),
            boundary="Alternative full-shape reverse execution of the same library primitives, not an independent derivative implementation or full Jacobian. No new science endpoint or threshold relaxation.",
        ),
    )
    model = load_model(True) if model is None else model
    assert not any(p.requires_grad for p in model.parameters())
    save_json(out / "runtime.json", runtime(model, synthetic))
    fs = {entry["key"]: entry for entry in read(reference / "forecast-status.json")["cases"]}
    rs = {entry["key"]: entry for entry in read(readout / "status.json")["cases"]}
    entries, counts, raw_bytes = [], [], 0
    for case in design["cases"]:
        release(model)
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        key = case["key"]
        forecast = entry_data(reference, "forecasts", fs[key])
        old_arrays = entry_data(readout, "cases", rs[key])
        engine = FullShape(model, case)
        data = dict(ordinary_F_scores=engine.false_scores, ordinary_F_final=engine.false_final)
        data["ordinary_F_source"] = array(engine.false_tail[0, 0])
        bases = {"F": engine.false_tail, "M": engine.first(forecast["M_write"])}
        data["first_M_source"] = array(bases["M"][0, 0])
        for anchor, base in bases.items():
            result = engine.gradient(base)
            data.update({anchor + "_" + name: value for name, value in result.items()})
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **data)
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 32 * 1024**2
        count = dict(
            full_F=1,
            first_block=1,
            full_shape_block_forwards=engine.forward_calls,
            block_recomputations=engine.reverse_calls,
            block_VJPs=engine.reverse_calls,
            head_VJPs=engine.head_reverse_calls,
        )
        save_json(
            path.with_suffix(".json"),
            dict(
                key=key,
                arrays_sha256=sha(path),
                forecast_entry=fs[key],
                readout_entry=rs[key],
                first_source_policy=engine.first_policy,
                counts=count,
                reference_array_sha256=rs[key]["arrays_sha256"],
                reference_delta_sha256=sha(readout / "cases" / (key + ".npz")),
                old_arrays_finite=all(np.isfinite(value).all() for value in old_arrays.values()),
            ),
        )
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(path),
                metadata_sha256=sha(path.with_suffix(".json")),
                bytes=path.stat().st_size,
            )
        )
        counts.append(count)
        print("286b full-shape adjoints saved", len(entries), "/", len(design["cases"]), flush=True)
        del engine, bases, forecast, old_arrays, data, result
        gc.collect()
        release(model)
    save_json(
        out / "status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=raw_bytes,
            ended=stamp(),
            counts={name: sum(row[name] for row in counts) for name in counts[0]},
            outcomes_not_yet_audited=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-285", type=Path, required=True)
    parser.add_argument("--readout-286", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.reference_285.resolve(), args.readout_286.resolve(), args.out.resolve())
