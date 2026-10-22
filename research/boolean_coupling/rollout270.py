"""Run the adaptive-backbone primitive factorial270 on four old contexts."""

import argparse
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .amplitude268 import ROOT, freeze_fields, normalization
from .audit229 import checked, sha
from .backbone270 import ARMS, bind, cache_nodes, policy
from .core225b import array, load_model, observe
from .fields267 import digest, emit, node_seal, read
from .input261b import Instability, polynomial
from .mlp258 import runtime

ACTIVE = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def execute(model, case, original, out):
    start, length, key = case["position"], len(case["false_ids"]), case["key"]
    bank, stars, live, saved = {}, {}, {}, {}
    records, checks, statuses, handles, counts = [], [], {}, [], Counter()
    arm = "F"
    eps = float(model.config.rms_norm_eps)
    qgamma = [array(layer.input_layernorm.weight) for layer in model.model.layers]
    gamma = [array(layer.post_attention_layernorm.weight) for layer in model.model.layers]
    fentries = {e["key"]: e for e in read(original/"forecast-status.json")["cases"]}
    tentries = {e["key"]: e for e in read(original/"status.json")["cases"]}

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
                cache_nodes(stars[index], field_nodes)
                data["frozen_field_nodes"] = node_seal(field_nodes)
                data["frozen_field_seal"] = {k: digest(v) for k, v in stars[index].items()}
                counts["L_field_caches"] += 1
            expected, nodes, mechanism = policy(arm, data["parts"], data["original_z"],
                                     stars.get(index), matrices, length, start, key, number)
            data["mechanism_seal"] = {k: digest(v) for k, v in mechanism.items()}
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
            counts["extra_down_projections"] += int(arm in ("J", "G", "X", "R"))
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
                mechanism_seal=data["mechanism_seal"],
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
                if arm in ("F", "H", "T"):
                    sub, entry, target = ("observations", tentries[key], "T") if arm == "T" else ("forecasts", fentries[key], "C" if arm == "H" else "F")
                    path = original/sub/(key+".npz")
                    assert sha(path) == entry["arrays_sha256"]
                    assert sha(path.with_suffix(".json")) == entry["record_sha256"]
                    with np.load(path) as old:
                        checked(checks, "original258_"+arm+"_trajectory", trajectory, old[target+"_trajectory"], 0.)
                        checked(checks, "original258_"+arm+"_scores", saved[arm+"_scores"], old[target+"_scores"], 0.)
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
            print("270", case["group"]["schema"], case["group"]["reverse"], arm, statuses[arm]["status"], flush=True)
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
            assert record["source_digest"] == lookup["T", number]["source_digest"]
    saved = {k: np.stack(v) if isinstance(v, list) else v for k, v in saved.items()}
    for name in ("z", "qz"):
        saved["F_"+name] = np.stack([bank[i][name] for i in range(28)])
    return saved, dict(checks=checks, records=records, statuses=statuses, counts=dict(counts))


def main(original, reference, previous, out):
    global ACTIVE
    assert not out.exists() and read(reference/"audit.json")["status"] == "passed"
    assert read(original/"audit.json")["status"] == "passed"
    assert read(previous/"audit.json")["status"] == "passed"
    assert sha(qwen.__file__) == read(original/"design.json")["qwen_source_sha256"]
    assert shutil.disk_usage(out.parent).free >= 800*1024**2
    out.mkdir(parents=True)
    ACTIVE = out
    (out/"cases").mkdir()
    sources = bind(out)
    cases = read(reference/"design.json")["cases"]
    assert cases == read(previous/"design.json")["cases"]
    assert len(cases) == 4 and len({(c["group"]["schema"], c["group"]["reverse"]) for c in cases}) == 4
    emit(out, out/"design.json", dict(experiment=270, retrospective=True, source_hashes=sources,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        cases=cases, arms=ARMS, began=stamp(), qwen_source_sha256=sha(qwen.__file__),
        original_hashes={n: sha(original/n) for n in ("design.json", "forecast-status.json", "status.json", "audit.json")},
        reference_hashes={n: sha(reference/n) for n in ("design.json", "status.json", "audit.json", "norm-weights.npz")},
        previous_hashes={n: sha(previous/n) for n in ("design.json", "status.json", "audit.json")}))
    model = load_model(True)
    assert runtime(model) == read(original/"runtime.json")
    emit(out, out/"runtime.json", runtime(model))
    entries, counts, total = [], Counter(), 0
    previous_entries = {r["key"]: r for r in read(previous/"status.json")["cases"]}
    for case in cases:
        assert shutil.disk_usage(out).free >= 768*1024**2
        saved, meta = execute(model, case, original, out)
        path = out/"cases"/(case["key"]+".npz")
        np.savez_compressed(path, **saved)
        try:
            old_path = previous/"cases"/(case["key"]+".npz")
            entry = previous_entries[case["key"]]
            assert sha(old_path) == entry["arrays_sha256"]
            assert sha(old_path.with_suffix(".json")) == entry["metadata_sha256"]
            old_records = {(r["arm"], r["layer"]): r for r in read(old_path.with_suffix(".json"))["records"]}
            mappings = {a: a for a in ("F", "Z", "L", "H", "T")}
            mappings["B"] = "A10"
            with np.load(old_path) as old:
                for arm, old_arm in mappings.items():
                    for field in ("x", "qx", "final", "scores"):
                        checked(meta["checks"], "exact268_"+arm+"_"+field, saved[arm+"_"+field], old[old_arm+"_"+field], 0.)
                for record in meta["records"]:
                    if record["arm"] in mappings:
                        before = old_records[mappings[record["arm"]], record["layer"]]
                        assert record["block_digest"] == before["block_digest"]
        except Exception as exc:
            emit(out, out/("failure-case-"+case["key"]+".json"), dict(error=repr(exc), arrays_sha256=sha(path), **meta))
            raise
        total += path.stat().st_size
        emit(out, path.with_suffix(".json"), dict(key=case["key"], arrays_sha256=sha(path), **meta))
        assert total <= 160*1024**2 and shutil.disk_usage(out).free >= 512*1024**2
        entries.append(dict(key=case["key"], arrays_sha256=sha(path), metadata_sha256=sha(path.with_suffix(".json")), bytes=path.stat().st_size))
        counts.update(meta["counts"])
    emit(out, out/"status.json", dict(status="complete", cases=entries, counts=dict(counts), raw_bytes=total, ended=stamp()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.original.resolve(), args.reference.resolve(), args.previous.resolve(), args.out.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/"failure.json", dict(error=repr(exc), time=stamp()))
        raise
