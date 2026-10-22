"""Globally sealed new-literal signed-amplitude forecasts and native measurements269."""

import argparse
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .amplitude268 import ROOT, freeze_fields, normalization, policy
from .amplitude268 import SOURCES as PREVIOUS_SOURCES
from .audit229 import checked, sha
from .checkpoint238 import locate
from .core225b import array, load_model, observe
from .fields267 import digest, emit, node_seal, read
from .input261b import Instability, polynomial
from .mlp258 import runtime

ACTIVE = None
ARMS = ("F", "Z", "L", "H", "A11", "A10", "A01", "A00", "Ar")
SOURCES = (*PREVIOUS_SOURCES, "research/boolean_coupling/forecast269.py",
    "research/boolean_coupling/audit269.py", "research/boolean_coupling/fixtures269.py",
    "research/boolean_coupling/cases269.json", "research/boolean_coupling/historical-input-hashes-269.json",
    "docs/experiments/269-sealed-amplitude-forecast.md")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def execute(model, case, out):
    start, length, key = case["position"], len(case["false_ids"]), case["key"]
    bank, stars, live, saved = {}, {}, {}, {}
    records, checks, statuses, handles, counts = [], [], {}, [], Counter()
    arm = "F"
    eps = float(model.config.rms_norm_eps)
    qgamma = [array(layer.input_layernorm.weight) for layer in model.model.layers]
    gamma = [array(layer.post_attention_layernorm.weight) for layer in model.model.layers]

    def install(index, layer):
        number = index+1
        matrices = {n: getattr(layer.mlp, n+"_proj").weight for n in ("gate", "up", "down")}

        def query_raw(module, args):
            live[index] = dict(qx=array(args[0][0, start+1:]))
            if arm == "F":
                bank[index] = dict(qx=live[index]["qx"])

        def query_input(module, args):
            old = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["qz"] = old
            changed = args[0]
            if arm not in ("F", "T"):
                lin, curv = polynomial(bank[index]["qx"].astype(float),
                    live[index]["qx"].astype(float), bank[index]["qz"].astype(float),
                    qgamma[index].astype(float), eps)
                value = (lin+curv).astype(np.float32)
                changed = args[0].clone()
                changed[:, start+1:] = torch.tensor(value[None], device="mps")
                checked(checks, "query_installation", array(changed[0, start+1:]), value, 0.)
                checked(checks, "query_prefix_source", array(changed[0, :start+1]), array(args[0][0, :start+1]), 0.)
                counts["cubic_query_maps"] += 1
            live[index].update(qoriginal_digest=digest(old),
                               qinstalled_digest=digest(array(changed[0, start+1:])))
            return (changed,)

        def mlp_raw(module, args):
            live[index]["x"] = array(args[0][0, start+1:])
            if arm == "F":
                bank[index]["x"] = live[index]["x"]

        def normalized(module, args, output):
            data = live[index]
            old = array(output[0, start+1:])
            data["original_z"] = old
            if arm == "F":
                bank[index]["z"] = old
            data["parts"] = normalization(bank[index]["x"], data["x"],
                                          bank[index]["z"], gamma[index], eps)
            changed = output
            if arm not in ("F", "T", "H"):
                changed = output.clone()
                value = data["parts"]["linear"].astype(np.float32)
                changed[:, start+1:] = torch.tensor(value[None], device="mps")
                checked(checks, "backbone_norm_installation", array(changed[0, start+1:]), value, 0.)
                checked(checks, "normalization_prefix_source", array(changed[0, :start+1]), array(output[0, :start+1]), 0.)
            data["installed_z"] = array(changed[0, start+1:])
            return changed

        def branch_output(name):
            def hook(module, args, output):
                live[index][name] = array(output[0, start+1:])
            return hook

        def product(module, args):
            data = live[index]
            data["product"] = array(args[0][0, start+1:])
            checked(checks, "finite_product", data["product"], data["phi"]*data["up"], 0.)

        def write(module, args, output):
            data = live[index]
            raw = array(output[0, start+1:])
            if arm == "L":
                stars[index], field_nodes = freeze_fields(data["parts"], matrices, length, start, key, number)
                data["frozen_field_nodes"] = node_seal(field_nodes)
                data["frozen_field_seal"] = {k: digest(v) for k, v in stars[index].items()}
                counts["L_field_caches"] += 1
            expected, nodes = policy(arm, data["parts"], data["original_z"],
                                     stars.get(index), matrices, length, start)
            for name in ("z", "gate", "up", "phi", "product"):
                actual = data["installed_z"] if name == "z" else data[name]
                checked(checks, "public_backbone_"+name, nodes["base"][name], actual, 0.)
            checked(checks, "public_backbone_write", nodes["base"]["write"], raw, 0.)
            changed = output
            if arm not in ("F", "T", "H", "L"):
                changed = output.clone()
                changed[:, start+1:] = torch.tensor(expected[None], device="mps")
                checked(checks, "write_prefix_source", array(changed[0, :start+1]), array(output[0, :start+1]), 0.)
            checked(checks, "exact_write_installation", array(changed[0, start+1:]), expected, 0.)
            data.update(installed_write=expected, nodes=node_seal(nodes))
            counts["finite_policy_points"] += 1
            counts["extra_public_MLP_evaluations"] += len(nodes)+(3 if arm == "L" else 0)
            return changed

        def guard(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            tail = array(value[0, start:])
            data = live[index]
            prefix = digest(array(value[0, :start]))
            if arm == "F":
                bank[index].update(prefix=prefix, tail_norm=float(np.linalg.norm(tail.astype(float))))
            assert prefix == bank[index]["prefix"]
            ratio = float(np.linalg.norm(tail.astype(float))/max(bank[index]["tail_norm"], .001))
            record = dict(arm=arm, layer=number, nodes=data["nodes"],
                qoriginal_digest=data["qoriginal_digest"], qinstalled_digest=data["qinstalled_digest"],
                original_z_digest=digest(data["original_z"]), installed_z_digest=digest(data["installed_z"]),
                write_digest=digest(data["installed_write"]), block_digest=digest(tail[1:]),
                source_digest=digest(tail[0]), prefix_digest=prefix, tail_ratio=ratio,
                frozen_field_nodes=data.get("frozen_field_nodes"), frozen_field_seal=data.get("frozen_field_seal"),
                geometry_seal={n: digest(data["parts"][n]) for n in ("alpha", "amplitude", "unit", "weighted_norm", "overlap", "energy")})
            records.append(record)
            for name in ("x", "qx"):
                saved.setdefault(arm+"_"+name, []).append(data[name])
            if not np.isfinite(tail).all() or ratio > 100:
                raise Instability(dict(arm=arm, layer=number, ratio=ratio))
            checked(checks, "exact_receiving_residual_addition", data["x"]+data["installed_write"], tail[1:], 0.)

        return ((layer.input_layernorm, query_raw, True), (layer.self_attn.q_proj, query_input, True),
            (layer.post_attention_layernorm, mlp_raw, True), (layer.post_attention_layernorm, normalized, False),
            (layer.mlp.gate_proj, branch_output("gate"), False),
            (layer.mlp.up_proj, branch_output("up"), False),
            (layer.mlp.act_fn, branch_output("phi"), False),
            (layer.mlp.down_proj, product, True), (layer.mlp.down_proj, write, False),
            (layer, guard, False))

    for index, layer in enumerate(model.model.layers):
        for module, function, before in install(index, layer):
            handles.append((module.register_forward_pre_hook if before else module.register_forward_hook)(function))
    try:
        for arm in ARMS:
            began = stamp()
            try:
                ids = case["false_ids" if arm in ("F", "Z") else "true_ids"]
                states, _, scores = observe(model, torch.tensor([ids], device="mps"))
                trajectory = np.stack([array(state[0, start:]) for state in states])
                saved[arm+"_final"] = trajectory[-1]
                saved[arm+"_scores"] = array(scores)
                assert np.isfinite(saved[arm+"_scores"]).all()
                if arm == "Z":
                    checked(checks, "zero_final", saved[arm+"_final"], saved["F_final"], 0.)
                    checked(checks, "zero_scores", saved[arm+"_scores"], saved["F_scores"], 0.)
                    for name in ("x", "qx"):
                        checked(checks, "zero_all_"+name, np.stack(saved["Z_"+name]), np.stack(saved["F_"+name]), 0.)
                statuses[arm] = dict(status="complete", began=began, ended=stamp())
            except Instability as exc:
                assert arm not in ("F", "Z", "L", "H", "T")
                statuses[arm] = dict(status="guard_failed", began=began, ended=stamp(), reason=str(exc))
            counts["full_forwards_attempted"] += 1
            print("269 forecast", case["group"]["schema"], case["group"]["reverse"], arm, statuses[arm]["status"], flush=True)
    except Exception:
        partial = {k: np.stack(v) if isinstance(v, list) else v for k, v in saved.items()}
        np.savez_compressed(out/("partial-"+key+".npz"), **partial)
        emit(out, out/("partial-"+key+".json"), dict(records=records, checks=checks, statuses=statuses))
        raise
    finally:
        for handle in handles:
            handle.remove()
    lookup = {(r["arm"], r["layer"]): r for r in records}
    for record in records:
        number, name = record["layer"], record["arm"]
        if name == "Z":
            for field in ("block_digest", "source_digest"):
                assert record[field] == lookup["F", number][field]
        elif name != "F":
            assert record["source_digest"] == lookup["H", number]["source_digest"]
    saved = {k: np.stack(v) if isinstance(v, list) else v for k, v in saved.items()}
    for name in ("z", "qz"):
        saved["F_"+name] = np.stack([bank[i][name] for i in range(28)])
    return saved, dict(checks=checks, records=records, statuses=statuses, counts=dict(counts))


def sources_check(run):
    design = read(run/"design.json")
    assert design["experiment"] == 269
    for name, value in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == value
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    return design


def forecasts(previous, out):
    global ACTIVE
    assert not out.exists() and read(previous/"audit.json")["status"] == "passed"
    assert shutil.disk_usage(out.parent).free >= 1200*1024**2
    out.mkdir(parents=True)
    ACTIVE = out
    (out/"forecasts").mkdir()
    (out/"sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/Path(name).name)
    cases = read(ROOT/"research/boolean_coupling/cases269.json")
    assert len(cases) == 32
    _, checkpoint = locate(ROOT)
    emit(out, out/"design.json", dict(experiment=269, retrospective=False, cases=cases, arms=ARMS,
        source_hashes={name: sha(ROOT/name) for name in SOURCES}, began=stamp(), checkpoint=checkpoint,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        qwen_source_sha256=sha(qwen.__file__), motivation_audit_sha256=sha(previous/"audit.json"),
        motivation_primary_passed=read(previous/"audit.json")["primary_amplitudes_sufficient"],
        budget="One observed False + known edit + L seed + forecast, with declared finite primitives; control arms and duplicate audit computations extra."))
    model = load_model(True)
    emit(out, out/"runtime.json", runtime(model))
    assert runtime(model) == read(ROOT/"results/receiver-mlp-curvature-258/runtime.json")
    entries, counts, total = [], Counter(), 0
    for number, case in enumerate(cases, 1):
        assert shutil.disk_usage(out).free >= 768*1024**2
        saved, meta = execute(model, case, out)
        path = out/"forecasts"/(case["key"]+".npz")
        np.savez_compressed(path, **saved)
        emit(out, path.with_suffix(".json"), dict(key=case["key"], arrays_sha256=sha(path), **meta))
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        counts.update(meta["counts"])
        total += path.stat().st_size
        assert total <= 600*1024**2 and shutil.disk_usage(out).free >= 512*1024**2
        print("269 globally pre-native forecasts", number, "/32", flush=True)
    emit(out, out/"forecast-status.json", dict(status="complete", cases=entries, counts=dict(counts), raw_bytes=total, ended=stamp(), ordinary_true=0))


def receipt(run):
    design = sources_check(run)
    audit = read(run/"forecast-audit.json")
    assert audit["status"] == "passed" and audit["ordinary_true"] == 0
    assert audit["status_sha256"] == sha(run/"forecast-status.json")
    seal = dict(experiment=269, globally_pre_native=True, ordinary_true=0, created=stamp(),
        source_commit=design["source_commit"], design_sha256=sha(run/"design.json"),
        status_sha256=sha(run/"forecast-status.json"), audit_sha256=sha(run/"forecast-audit.json"),
        cases=read(run/"forecast-status.json")["cases"],
        raw_projection_sha256=sha(run/"forecast-raw-projections.csv"),
        scores_sha256=sha(run/"forecast-scores.csv"))
    emit(run, run/"forecast-seal.json", seal)
    destination = ROOT/"results/signed-amplitude-forecast-269/pre-native"
    assert not destination.exists()
    destination.mkdir(parents=True)
    for name in ("design.json", "runtime.json", "forecast-status.json", "forecast-audit.json",
                 "forecast-seal.json", "forecast-scores.csv", "forecast-raw-projections.csv", "forecast-raw-head.csv"):
        shutil.copy2(run/name, destination/name)
    print("269 global forecast receipt ready to commit/push BEFORE any unmodified True measurement.", flush=True)


def natives(run, proof_path):
    global ACTIVE
    design = sources_check(run)
    assert not (run/"native-started.json").exists() and not (run/"native-status.json").exists()
    proof = read(proof_path)
    assert proof["forecast_seal_sha256"] == sha(run/"forecast-seal.json")
    commit = proof["public_commit"]
    relative = "results/signed-amplitude-forecast-269/pre-native/forecast-seal.json"
    public_bytes = subprocess.check_output(["git", "show", commit+":"+relative], cwd=ROOT)
    import hashlib

    assert hashlib.sha256(public_bytes).hexdigest() == proof["forecast_seal_sha256"]
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/research/amplitude-direction-268"], cwd=ROOT, text=True).split()[0]
    assert remote == commit, (remote, commit)
    assert read(run/"forecast-seal.json")["globally_pre_native"]
    assert read(run/"forecast-audit.json")["status"] == "passed"
    assert len(read(run/"forecast-status.json")["cases"]) == 32
    ACTIVE = run
    (run/"observations").mkdir()
    emit(run, run/"native-started.json", dict(started=stamp(), proof_sha256=sha(proof_path),
        public_commit=commit, forecast_seal_sha256=sha(run/"forecast-seal.json"), ordinary_true_before=0))
    model = load_model(True)
    assert runtime(model) == read(run/"runtime.json")
    entries = []
    for number, case in enumerate(design["cases"], 1):
        start, key = case["position"], case["key"]
        states, _, scores = observe(model, torch.tensor([case["true_ids"]], device="mps"))
        final = array(states[-1][0, start:])
        score = array(scores)
        assert np.isfinite(final).all() and np.isfinite(score).all()
        records = []
        for layer, state in enumerate(states, 1):
            value = array(state[0])
            assert np.isfinite(value).all()
            records.append(dict(layer=layer, block_digest=digest(value[start+1:]),
                source_digest=digest(value[start]), prefix_digest=digest(value[:start])))
        path = run/"observations"/(key+".npz")
        assert not path.exists()
        np.savez_compressed(path, T_final=final, T_scores=score)
        emit(run, path.with_suffix(".json"), dict(key=key, arrays_sha256=sha(path), records=records, measured=stamp()))
        entries.append(dict(key=key, arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        assert shutil.disk_usage(run).free >= 512*1024**2
        print("269 sealed native measurement", number, "/32", flush=True)
    emit(run, run/"native-status.json", dict(status="complete", cases=entries, ordinary_true=32,
        forecast_seal_sha256=sha(run/"forecast-seal.json"), native_started_sha256=sha(run/"native-started.json"), ended=stamp()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "receipt", "native"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "forecast":
            assert args.previous is not None
            forecasts(args.previous.resolve(), args.run.resolve())
        elif args.phase == "receipt":
            receipt(args.run.resolve())
        else:
            assert args.proof is not None
            natives(args.run.resolve(), args.proof.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/("failure-"+args.phase+".json"), dict(error=repr(exc), time=stamp()))
        raise
