"""Globally pre-native forecasts, public receipt, then fresh native measurements277."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import sha
from .audit272 import read
from .cache272 import digest, installed_sources, save_json
from .checkpoint238 import locate, weights
from .core225b import array, load_model, observe
from .firstmlp276 import finite, tangent
from .run272 import runtime, stamp
from .source273 import paired
from .source277 import ARMS, SOURCES, First, Forecast, prepare, proposals

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = Path("results/first-source-mlp-forecast-277/pre-native")


def validate(run):
    design = read(run / "design.json")
    assert design["experiment"] == 277 and design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    assert locate(ROOT)[1] == design["checkpoint"]
    return design


def forecasts(reference, run):
    assert not run.exists() and shutil.disk_usage(run.parent).free >= 1280 * 1024**2
    previous = read(reference / "audit.json")
    assert previous["experiment"] == 276 and previous["status"] == "passed" and previous["primary_passes"]
    assert not previous["gain_explanation_passes"]  # retain, never upgrade the failed276 claim
    cases = read(ROOT / "research/boolean_coupling/cases277.json")
    assert len(cases) == 32
    run.mkdir(parents=True)
    for name in ("forecasts", "sources"):
        (run / name).mkdir()
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in hashes:
        shutil.copy2(ROOT / name, run / "sources" / Path(name).name)
    folder, checkpoint = locate(ROOT)
    partners = {a["key"]: b["key"] for pair in paired(cases) for a, b in (pair, pair[::-1])}
    save_json(run / "design.json", dict(experiment=277, cases=cases, partners=partners, arms=ARMS,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, installed_sources=installed_sources(), checkpoint=checkpoint,
        reference_hashes={name: sha(reference / name) for name in ("design.json", "status.json", "audit.json")},
        began=stamp(), globally_pre_native=True, ordinary_true=0,
        primary="Unchanged276 S: original native receiving/verdict5%/.99/.10nat AND paired first-source/verdict25%/.95/material/finite-margin, >=24/32 and>=6/8 each stratum. Separate gain account unchanged and cannot rescue failed276.",
        boundary="Fresh literals in dependent templates. Known edit + False observation + weights. First source MLP approximated; all other dense model operations finite. No compact full predictor, efficiency, learned circuit or general Question B solution."))
    model = load_model(True)
    save_json(run / "runtime.json", runtime(model))
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, read(folder / "config.json"), 0, "mlp")
    entries, total = [], 0
    for case in cases:
        assert shutil.disk_usage(run).free >= 768 * 1024**2
        key, start, length = case["key"], case["position"], len(case["false_ids"])
        with First(model, case) as capture:
            states, context, scores = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        false = capture.data
        prefixes = [digest(array(s[0, :start])) for s in states]
        values = dict(F_source=np.stack([array(s[0, start]) for s in states]),
                      F_final=array(states[-1][0, start:]), F_scores=array(scores),
                      rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]))
        prepared, points, components = prepare(model, case, context, false)
        with torch.no_grad():
            nodes = {name: finite(model.model.layers[0], points[name], length, start) for name in ("F", "C", "R")}
        for name in ("z", "gate", "up", "phi", "product", "write"):
            assert np.array_equal(nodes["F"][name], false[name])
        derivatives = {name: tangent(w, points[name], components["read"]) for name in ("F", "C")}
        writes, gain, random = proposals(nodes, derivatives, key)
        for prefix, data in (("false", false), ("prepared", prepared), ("point", points),
                             ("component", components), ("proposal", writes), ("random", random)):
            values.update({prefix + "_" + n: v for n, v in data.items()})
        values["gain"] = gain
        for name, data in nodes.items():
            values.update({"node_" + name + "_" + n: v for n, v in data.items()})
        for name, data in derivatives.items():
            values.update({"derivative_" + name + "_" + n: v for n, v in data.items()})
        with Forecast(model, case, points, nodes, writes) as engine, torch.no_grad():
            for arm in ARMS:
                states, scores, record = engine.forward(arm)
                assert [digest(array(s[0, :start])) for s in states] == prefixes
                values[arm + "_source"] = np.stack([array(s[0, start]) for s in states])
                values[arm + "_final"], values[arm + "_scores"] = array(states[-1][0, start:]), scores
                values.update({arm + "_" + n: v for n, v in record.items()})
        for node in ("source", "final", "scores"):
            assert np.array_equal(values["FZ_" + node], values["F_" + node])
        path = run / "forecasts" / (key + ".npz")
        np.savez_compressed(path, **values)
        save_json(path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), measured=stamp(),
            false_prefix_hashes=prefixes, all_forecast_prefixes_exact=True, source_only_install=True,
            source_True_MLP_primal_evaluations=0, prepared_attention_only=True, ordinary_true=0))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        total += path.stat().st_size
        assert total <= 175 * 1024**2
        print("277 pre-native forecast", len(entries), "/32", flush=True)
    save_json(run / "forecast-status.json", dict(status="complete", cases=entries, raw_bytes=total,
        ordinary_true=0, ended=stamp(), counts=dict(full_false=32, full_forecasts=288, source_first_attention=32,
        corner_PV=64, corner_output_projections=64, finite_first_MLP_maps=96,
        CPU64_MLP_tangents=64, CPU64_tangent_dense_matvec=320, native_True_source_MLP_primals=0)))


def receipt(run):
    design = validate(run)
    assert not (run / "native-started.json").exists()
    audit = read(run / "forecast-audit.json")
    assert audit["status"] == "passed" and audit["ordinary_true"] == 0
    assert audit["status_sha256"] == sha(run / "forecast-status.json")
    save_json(run / "forecast-seal.json", dict(experiment=277, globally_pre_native=True, ordinary_true=0,
        created=stamp(), source_commit=design["source_commit"], design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "forecast-status.json"), audit_sha256=sha(run / "forecast-audit.json"),
        cases=read(run / "forecast-status.json")["cases"],
        csv_sha256={p.name: sha(p) for p in run.glob("forecast-*.csv")}))
    destination = ROOT / RECEIPT
    assert not destination.exists()
    destination.mkdir(parents=True)
    names = ["design.json", "runtime.json", "forecast-status.json", "forecast-audit.json", "forecast-seal.json"]
    for path in [*(run / n for n in names), *run.glob("forecast-*.csv")]:
        shutil.copy2(path, destination / path.name)
    print("277 forecasts and qualifications globally sealed; publish this receipt BEFORE native True.", flush=True)


def natives(run, proof_path):
    design = validate(run)
    assert not (run / "native-started.json").exists() and not (run / "observations").exists()
    proof = read(proof_path)
    commit = proof["public_commit"]
    assert proof["forecast_seal_sha256"] == sha(run / "forecast-seal.json")
    public_bytes = subprocess.check_output(["git", "show", commit + ":" + str(RECEIPT / "forecast-seal.json")], cwd=ROOT)
    assert hashlib.sha256(public_bytes).hexdigest() == proof["forecast_seal_sha256"]
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=ROOT, text=True).split()[0]
    assert remote == commit and read(run / "forecast-audit.json")["status"] == "passed"
    assert read(run / "forecast-seal.json")["ordinary_true"] == 0
    save_json(run / "native-started.json", dict(started=stamp(), public_commit=commit,
        proof_sha256=sha(proof_path), forecast_seal_sha256=sha(run / "forecast-seal.json"), ordinary_true_before=0))
    (run / "observations").mkdir()
    model = load_model(True)
    assert runtime(model) == read(run / "runtime.json")
    entries = []
    for case in design["cases"]:
        start, key = case["position"], case["key"]
        with First(model, case) as capture:
            states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        with np.load(run / "forecasts" / (key + ".npz")) as forecast:
            for name in ("before", "norm", "q", "k", "v", "full_k", "full_v", "p", "mask", "scaling", "y", "attwrite", "mid"):
                assert np.array_equal(capture.data[name], forecast["prepared_" + name])
            assert [digest(array(s[0, :start])) for s in states] == read(run / "forecasts" / (key + ".json"))["false_prefix_hashes"]
        values = dict(T_source=np.stack([array(s[0, start]) for s in states]),
                      T_final=array(states[-1][0, start:]), T_scores=array(scores),
                      **{"native_" + name: value for name, value in capture.data.items()})
        path = run / "observations" / (key + ".npz")
        np.savez_compressed(path, **values)
        save_json(path.with_suffix(".json"), dict(key=key, measured=stamp(), arrays_sha256=sha(path),
            prepared_first_attention_exact=True, prefix_exact=True))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        assert shutil.disk_usage(run).free >= 512 * 1024**2
        print("277 sealed native measurement", len(entries), "/32", flush=True)
    save_json(run / "native-status.json", dict(status="complete", ordinary_true=32, cases=entries, ended=stamp(),
        forecast_seal_sha256=sha(run / "forecast-seal.json"), native_started_sha256=sha(run / "native-started.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "receipt", "native"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-276", type=Path)
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            forecasts(args.reference_276.resolve(), args.run.resolve())
        elif args.phase == "receipt":
            receipt(args.run.resolve())
        else:
            natives(args.run.resolve(), args.proof.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / ("failure-" + args.phase + ".json")).exists():
            save_json(args.run / ("failure-" + args.phase + ".json"), dict(error=repr(exc), time=stamp()))
        raise
