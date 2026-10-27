"""Retrospective False-readout sensitivity diagnostic on the qualified 285 cohort."""

import argparse
import gc
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .cache272 import installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model
from .readout286 import ROOT, SOURCES, Suffix, directions, finite_difference, release, split
from .run272 import stamp
from .run285 import model_signature, runtime

REF_FILES = (
    "runtime.json",
    "design.json",
    "forecast-status.json",
    "forecast-audit.json",
    "forecast-seal.json",
    "native-status.json",
    "native-audit.json",
    "native-started.json",
)
COMPONENTS = ("parallel_F", "orthogonal_F", "parallel_M", "orthogonal_M", "random0", "random1")


def entry_data(reference, folder, entry):
    p = reference / folder / (entry["key"] + ".npz")
    assert sha(p) == entry["arrays_sha256"] and p.stat().st_size == entry["bytes"]
    assert sha(p.with_suffix(".json")) == entry["metadata_sha256"]
    with np.load(p) as d:
        return {n: d[n] for n in d.files}


def head_data(model):
    return dict(
        norm=array(model.model.norm.weight),
        weight=array(model.lm_head.weight[[895, 830]]),
        eps=np.asarray(model.config.rms_norm_eps, dtype=np.float64),
    )


def cpu_head(head, final):
    h = final[-1].astype(float)
    return (
        h
        / np.sqrt(np.mean(h * h) + float(head["eps"]))
        * head["norm"].astype(float)
        @ head["weight"].astype(float).T
    )


def main(reference, out, model=None, sentinel=False):
    assert not out.exists() and shutil.disk_usage(out.parent).free >= 768 * 1024**2
    old = read(reference / "design.json")
    assert old["experiment"] == 285
    assert old["installed_sources"] == installed_sources()
    assert (
        read(reference / "forecast-audit.json")["status"]
        == read(reference / "native-audit.json")["status"]
        == "passed"
    )
    synthetic = model is not None
    assert old["synthetic"] is synthetic
    if synthetic:
        assert model.device.type == "cpu" and model_signature(model) == old["checkpoint"]
    else:
        assert locate(ROOT)[1] == old["checkpoint"]
    for n, h in old["source_hashes"].items():
        assert sha(ROOT / n) == h
    cases = old["cases"]
    if sentinel:
        cases = [
            c
            for c in cases
            if c["group"]["schema"] == "generic"
            and c["group"]["reverse"] == c["group"]["target"] == 0
        ]
        assert len(cases) == 4
    else:
        assert len(cases) == 32
    out.mkdir()
    (out / "sources").mkdir()
    (out / "cases").mkdir()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    for n in hashes:
        shutil.copy2(ROOT / n, out / "sources" / Path(n).name)
    save_json(
        out / "design.json",
        dict(
            experiment=286,
            retrospective=True,
            synthetic=synthetic,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=hashes,
            installed_sources=installed_sources(),
            reference_hashes={n: sha(reference / n) for n in REF_FILES},
            cases=cases,
            checkpoint=old["checkpoint"],
            began=stamp(),
            numerical_bounds=dict(
                replay_coordinate=0.002,
                replay_relative=0.0001,
                replay_score=0.001,
                projected_AD_FD_absolute=0.001,
                projected_AD_FD_relative=0.02,
                FD_halving_absolute=0.002,
                FD_halving_relative=0.03,
                CPU64_head=0.005,
            ),
            FD_steps=[0.05, 0.025],
            sentinel=sentinel,
            observed="One ordinary False full KV observation; source residual after block1.",
            rule="Predict native T-M margin by dot(False-state source sensitivity, C_source-M_source). M-gradient is an extra forecast-state control.",
            boundary="Old285 outcomes already known; no prospective/fresh or full native-vector claim. Original failed controls/derivatives unchanged. All affected suffix operations finite during derivative/causal measurement; no learned circuit/efficiency/priority/generalQB claim.",
        ),
    )
    model = load_model(True) if model is None else model
    assert not any(p.requires_grad for p in model.parameters())
    run_runtime = runtime(model, synthetic)
    run_runtime.update(gradient_dtype="model_float32", gradient_node="source residual after block1")
    save_json(out / "runtime.json", run_runtime)
    release(model)
    head = head_data(model)
    np.savez_compressed(out / "head.npz", **head)
    save_json(
        out / "head.json",
        dict(
            arrays_sha256=sha(out / "head.npz"),
            dtype="float32 model weights; Float64 epsilon",
            runtime_device=str(model.device),
        ),
    )
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ns = {e["key"]: e for e in read(reference / "native-status.json")["cases"]}
    entries, counts, raw_bytes = [], [], 0
    for case in cases:
        release(model)
        assert shutil.disk_usage(out).free >= 512 * 1024**2
        key = case["key"]
        f = entry_data(reference, "forecasts", fs[key])
        engine = Suffix(model, case)
        checks, fd = [], []

        def check(name, a, b, bound=0.0):
            err = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
            row = dict(
                name=name, error=err, bound=bound, passes=bool(np.isfinite(err) and err <= bound)
            )
            checks.append(row)
            assert row["passes"], row

        def replay(name, final, scores, expected_final, expected_scores):
            check(name + "_raw_coordinate", final, expected_final, 0.002)
            relative = np.linalg.norm(final.astype(float) - expected_final.astype(float)) / max(
                np.linalg.norm(expected_final.astype(float)), 1e-30
            )
            check(name + "_relative", relative, 0.0, 0.0001)
            check(name + "_score", scores, expected_scores, 0.001)
            check(name + "_CPU64_head", cpu_head(head, final), scores, 0.005)

        check("ordinary_F_source", array(engine.false_tail[0, 0]), f["F_source"])
        check("ordinary_F_final", engine.false_final, f["F_final"])
        check("ordinary_F_scores", engine.false_scores, f["F_scores"])
        for n in ("gate", "phi", "write"):
            check("ordinary_F_" + n, engine.false_nodes[n], f["F_" + n])
        mbase = engine.first(f["M_write"])
        check("first_M_source", array(mbase[0, 0]), f["M_source"])
        delta = f["C_source"].astype(float) - f["M_source"].astype(float)
        data = dict(
            delta=delta, F_source=f["F_source"], M_source=f["M_source"], C_source=f["C_source"]
        )
        bases = {"F": engine.false_tail, "M": mbase}
        for anchor, base in bases.items():
            grad, scores, final, trace, trace_norms = engine.gradient(base)
            release(model)
            replay("gradient_" + anchor, final, scores, f[anchor + "_final"], f[anchor + "_scores"])
            data["g" + anchor] = grad
            data["g" + anchor + "_source_trace"] = trace
            data["g" + anchor + "_tail_norm_trace"] = trace_norms
            data[anchor + "_scores"] = scores
        pf, of = split(data["gF"], delta)
        pm, om = split(data["gM"], delta)
        components = dict(parallel_F=pf, orthogonal_F=of, parallel_M=pm, orthogonal_M=om)
        rng = np.random.default_rng(
            int(hashlib.sha256((key + ":286-random").encode()).hexdigest()[:16], 16)
        )
        for i in range(2):
            r = rng.normal(size=len(delta))
            components["random" + str(i)] = r * np.linalg.norm(pf) / max(np.linalg.norm(r), 1e-30)
        check("component_sum_F", pf + of, delta, 1e-12)
        check("component_sum_M", pm + om, delta, 1e-12)
        check("gradient_orthogonal_F", np.dot(data["gF"].astype(float), of), 0.0, 1e-10)
        check("gradient_orthogonal_M", np.dot(data["gM"].astype(float), om), 0.0, 1e-10)
        for anchor, base in bases.items():
            for label, d in directions(key, delta, pf, of).items():
                result = finite_difference(engine, base, data["g" + anchor], d, label)
                fd.extend(dict(anchor=anchor, **r) for r in result)
                if not all(r["passes"] for r in result):
                    save_json(
                        out / "failure-projected-derivative.json",
                        dict(key=key, checks=checks, derivatives=fd),
                    )
                    raise AssertionError((key, anchor, label, result))
        # Defined predictions and causal directions exist before native arrays
        # are opened. This order does not make the OLD cohort prospective.
        for name, component in components.items():
            final, scores = engine.finite(mbase, f["M_source"].astype(float) + component)
            release(model)
            check(name + "_CPU64_head", cpu_head(head, final), scores, 0.005)
            data.update(
                {name + "_component": component, name + "_final": final, name + "_scores": scores}
            )
        final, scores = engine.finite(engine.false_tail)
        release(model)
        replay("cached_F", final, scores, f["F_final"], f["F_scores"])
        data.update(F_final=final, F_scores=scores)
        final, scores = engine.finite(mbase)
        release(model)
        replay("cached_M", final, scores, f["M_final"], f["M_scores"])
        data.update(M_final=final, M_scores=scores)
        final, scores = engine.finite(mbase, f["C_source"])
        release(model)
        replay("cached_C", final, scores, f["C_final"], f["C_scores"])
        data.update(C_final=final, C_scores=scores)
        t = entry_data(reference, "native", ns[key])
        final, scores = engine.finite(mbase, t["T_source"])
        release(model)
        replay("oracle_cached_T", final, scores, t["T_final"], t["T_scores"])
        data.update(T_source=t["T_source"], T_final=final, T_scores=scores)
        # Preserve ordinary full-shape scores as scientific targets.
        for arm, ref in (("F", f), ("M", f), ("C", f), ("T", t)):
            data["reference_" + arm + "_scores"] = ref[arm + "_scores"]
        engine.unchanged()
        path = out / "cases" / (key + ".npz")
        np.savez_compressed(path, **data)
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 32 * 1024**2
        meta = dict(
            key=key,
            arrays_sha256=sha(path),
            checks=checks,
            derivatives=fd,
            prefix_seals=engine.seals,
            first_source_policy=engine.first_policy,
            prefix_unchanged=True,
            forecast_entry=fs[key],
            native_entry=ns[key],
            counts=dict(
                full_F=1,
                first_block=engine.first_blocks,
                suffix_forwards=engine.calls,
                backwards=engine.backwards,
            ),
            old_native_opened_only_after_defined_predictions_and_components=True,
        )
        save_json(path.with_suffix(".json"), meta)
        entries.append(
            dict(
                key=key,
                arrays_sha256=sha(path),
                metadata_sha256=sha(path.with_suffix(".json")),
                bytes=path.stat().st_size,
            )
        )
        counts.append(meta["counts"])
        print("286 numerically qualified", len(entries), "/", len(cases), flush=True)
        del engine, bases, mbase, f, t, data
        gc.collect()
        release(model)
    save_json(
        out / "status.json",
        dict(
            status="complete",
            cases=entries,
            raw_bytes=raw_bytes,
            counts={n: sum(c[n] for c in counts) for n in counts[0]},
            ended=stamp(),
            outcomes_not_yet_audited=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-285", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sentinel", action="store_true")
    a = p.parse_args()
    try:
        main(a.reference_285.resolve(), a.out.resolve(), sentinel=a.sentinel)
    except Exception as exc:
        if a.out.exists() and not (a.out / "failure.json").exists():
            save_json(a.out / "failure.json", dict(error=repr(exc), time=stamp()))
        raise
