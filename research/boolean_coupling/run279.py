"""Freeze/seal new-literal source activation forecasts before native measurements279."""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit272 import read
from .cache272 import digest, installed_sources, save_json
from .checkpoint238 import locate
from .core225b import array, load_model, observe
from .run272 import runtime, stamp
from .source273 import paired
from .source277 import First
from .source279 import ARMS, SOURCES, Forecast, candidates, prepare

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = Path("results/source-activation-forecast-279/pre-native")


def validate(run):
    design = read(run / "design.json")
    assert design["experiment"] == 279 and design["installed_sources"] == installed_sources()
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h
    assert locate(ROOT)[1] == design["checkpoint"]
    return design


def forecasts(reference, run):
    assert not run.exists() and shutil.disk_usage(run.parent).free >= 1280 * 1024**2
    previous = read(reference / "audit.json")
    assert previous["experiment"] == 278 and previous["status"] == "passed"
    assert not previous["primary_passes"] and not previous["prespecified_secondary_candidates"]
    cases = read(ROOT / "research/boolean_coupling/cases279.json")
    run.mkdir(parents=True)
    for name in ("forecasts", "sources"):
        (run / name).mkdir()
    hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in hashes:
        shutil.copy2(ROOT / name, run / "sources" / Path(name).name)
    partners = {a["key"]: b["key"] for pair in paired(cases) for a, b in (pair, pair[::-1])}
    save_json(run / "design.json", dict(experiment=279, cases=cases, partners=partners, arms=ARMS,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, installed_sources=installed_sources(), checkpoint=locate(ROOT)[1], began=stamp(),
        reference_hashes={name: sha(reference / name) for name in ("design.json", "status.json", "audit.json")},
        globally_pre_native=True, ordinary_true=0,
        primary="Cubic C joint native5%/.99/.10nat + sourcewrite10%/.995 + material whole-firstMLP T-Z effect25%/.95/margin + paired source/verdict25%/.95/margin, >=24/32 and>=6/8 each stratum. S ranked ceil1% exact is a separate preregistered candidate; stronger C->native correction test with matched controls remains separate.",
        boundary="New literals in dependent templates; known edit+False computation+weights. One first-source MLP activation approximated. Finite first attention/norm/projections/product and every later operation. No useful whole-network speedup or learned circuit/general QB solution."))
    model = load_model(True)
    save_json(run / "runtime.json", runtime(model))
    column_norm = np.linalg.norm(array(model.model.layers[0].mlp.down_proj.weight).astype(float), axis=0)
    entries, raw_bytes = [], 0
    for case in cases:
        assert shutil.disk_usage(run).free >= 768 * 1024**2
        key, start = case["key"], case["position"]
        with First(model, case) as captured:
            states, context, scores = observe(model, torch.tensor([case["false_ids"]], device="mps"))
        false = captured.data
        prefixes = [digest(array(s[0, :start])) for s in states]
        false_tails = [digest(array(s[0, start:])) for s in states]
        values = dict(F_source=array(states[0][0, start]), F_final=array(states[-1][0, start:]), F_scores=array(scores),
                      rope_cos=array(context["rope"][0]), rope_sin=array(context["rope"][1]), column_norm=column_norm)
        prepared = prepare(model, case, context, false)
        plan = candidates(model.model.layers[0], case, false, prepared, column_norm)
        for prefix, data in (("false", false), ("prepared", prepared)):
            values.update({prefix + "_" + n: v for n, v in data.items()})
        values.update(dg=plan["dg"], mask_score=plan["score"])
        for prefix, data in plan.items():
            if isinstance(data, dict):
                values.update({prefix + "_" + n: v for n, v in data.items()})
        with Forecast(model, case, false, prepared, plan) as engine, torch.no_grad():
            for arm in ARMS:
                states, scores, record = engine.forward(arm)
                assert [digest(array(s[0, :start])) for s in states] == prefixes
                if arm == "FZ":
                    assert [digest(array(s[0, start:])) for s in states] == false_tails
                values.update({arm + "_source": array(states[0][0, start]), arm + "_final": array(states[-1][0, start:]), arm + "_scores": scores})
                values.update({arm + "_actual_" + n: v for n, v in record.items()})
        for name in ("source", "final", "scores"):
            assert np.array_equal(values["FZ_" + name], values["F_" + name])
        path = run / "forecasts" / (key + ".npz")
        np.savez_compressed(path, **values)
        save_json(path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), measured=stamp(),
            ordinary_true=0, all_prefixes_exact=True, prefix_hashes=prefixes, False_zero_exact=True,
            only_first_source_MLP_write_changed=True, full_True_source_activation_primal_count=0,
            controller_current_coordinates_per_mask=len(plan["masks"]["S"]), False_source_primal_for_all_forecasts=True))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        raw_bytes += path.stat().st_size
        assert raw_bytes <= 192 * 1024**2
        print("279 pre-native forecast", len(entries), "/32", flush=True)
    save_json(run / "forecast-status.json", dict(status="complete", ordinary_true=0, cases=entries, raw_bytes=raw_bytes, ended=stamp(),
        counts=dict(full_false=32, full_forecasts=288, first_attention_only=32, source_norm_gate_up_maps=32,
            masked_full_shaped_SiLU_maps=128, current_source_SiLU_coordinates_per_sparse_arm=90,
            current_source_SiLU_coordinates_all_masks=11520, source_product_down_maps=224,
            full_True_source_activation_primal_count=0)))


def receipt(run):
    design = validate(run)
    assert not (run / "native-started.json").exists()
    audit = read(run / "forecast-audit.json")
    assert audit["status"] == "passed" and audit["ordinary_true"] == 0
    assert audit["status_sha256"] == sha(run / "forecast-status.json")
    save_json(run / "forecast-seal.json", dict(experiment=279, globally_pre_native=True, ordinary_true=0, created=stamp(),
        source_commit=design["source_commit"], design_sha256=sha(run / "design.json"), status_sha256=sha(run / "forecast-status.json"),
        audit_sha256=sha(run / "forecast-audit.json"), cases=read(run / "forecast-status.json")["cases"],
        csv_sha256={p.name: sha(p) for p in run.glob("forecast-*.csv")}))
    dest = ROOT / RECEIPT
    assert not dest.exists()
    dest.mkdir(parents=True)
    names = ("design.json", "runtime.json", "forecast-status.json", "forecast-audit.json", "forecast-seal.json")
    for p in [*(run / n for n in names), *run.glob("forecast-*.csv")]:
        shutil.copy2(p, dest / p.name)
    print("279 all forecasts/qualification sealed; commit/push receipt BEFORE native True.", flush=True)


def natives(run, proof_path):
    design = validate(run)
    assert not (run / "native-started.json").exists() and not (run / "observations").exists()
    proof = read(proof_path)
    commit, seal_hash = proof["public_commit"], proof["forecast_seal_sha256"]
    assert seal_hash == sha(run / "forecast-seal.json")
    content = subprocess.check_output(["git", "show", commit + ":" + str(RECEIPT / "forecast-seal.json")], cwd=ROOT)
    assert hashlib.sha256(content).hexdigest() == seal_hash
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=ROOT, text=True).split()[0]
    assert remote == commit and read(run / "forecast-audit.json")["status"] == "passed"
    save_json(run / "native-started.json", dict(started=stamp(), public_commit=commit, proof_sha256=sha(proof_path), forecast_seal_sha256=seal_hash, ordinary_true_before=0))
    (run / "observations").mkdir()
    model = load_model(True)
    entries = []
    for case in design["cases"]:
        start, key = case["position"], case["key"]
        with First(model, case) as captured:
            states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        with np.load(run / "forecasts" / (key + ".npz")) as f:
            for name in ("before", "norm", "q", "k", "v", "full_k", "full_v", "p", "mask", "scaling", "y", "attwrite", "mid", "z", "gate", "up"):
                assert np.array_equal(captured.data[name], f["prepared_" + name]), name
            for arm in ("S", "R0", "R1", "R2"):
                assert np.array_equal(captured.data["phi"][f["masks_" + arm]], f["selected_phi_" + arm])
            assert [digest(array(s[0, :start])) for s in states] == read(run / "forecasts" / (key + ".json"))["prefix_hashes"]
        path = run / "observations" / (key + ".npz")
        np.savez_compressed(path, T_source=array(states[0][0, start]), T_final=array(states[-1][0, start:]), T_scores=array(scores),
                            **{"native_" + n: v for n, v in captured.data.items()})
        save_json(path.with_suffix(".json"), dict(key=key, measured=stamp(), arrays_sha256=sha(path),
            prepared_attention_norm_gate_up_exact=True, selected_current_phi_exact=True, prefix_exact=True))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        assert shutil.disk_usage(run).free >= 512 * 1024**2
        print("279 sealed native measurement", len(entries), "/32", flush=True)
    save_json(run / "native-status.json", dict(status="complete", ordinary_true=32, cases=entries, ended=stamp(),
        native_started_sha256=sha(run / "native-started.json"), forecast_seal_sha256=seal_hash))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "receipt", "native"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-278", type=Path)
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            forecasts(args.reference_278.resolve(), args.run.resolve())
        elif args.phase == "receipt":
            receipt(args.run.resolve())
        else:
            natives(args.run.resolve(), args.proof.resolve())
    except Exception as exc:
        if args.run.exists() and not (args.run / ("failure-" + args.phase + ".json")).exists():
            save_json(args.run / ("failure-" + args.phase + ".json"), dict(error=repr(exc), time=stamp()))
        raise
