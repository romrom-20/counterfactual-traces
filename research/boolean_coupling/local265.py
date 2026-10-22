"""Same-input primitive factorial on all saved263 H/L MLP inputs."""

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import checked, sha
from .audit261b import expansion, metrics
from .checkpoint238 import locate, read_head
from .core225b import write_json
from .transmission265 import components, corners, project

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("research/boolean_coupling/transmission265.py", "research/boolean_coupling/local265.py",
    "research/boolean_coupling/rollout265.py", "research/boolean_coupling/audit265.py",
    "docs/experiments/265-local-mixing-and-state-feedback.md", "research/boolean_coupling/input263.py",
    "research/boolean_coupling/input261b.py", "research/boolean_coupling/audit261b.py",
    "research/boolean_coupling/curvature255.py",
    "research/boolean_coupling/silu258.py", "research/boolean_coupling/checkpoint238.py",
    "research/boolean_coupling/core225b.py", "research/boolean_coupling/audit229.py",
    "research/boolean_coupling/mlp258.py", "research/boolean_coupling/settings.json")
ACTIVE = None


def read(path):
    return json.loads(path.read_text())


def emit(out, path, value):
    assert path.resolve().is_relative_to(out.resolve()) and not path.exists(), path
    write_json(path, value)


def table(path, data):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def bind(out):
    (out/"sources").mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name, out/"sources"/Path(name).name)
    return {name:sha(ROOT/name) for name in SOURCES}


def anchor(values, layer):
    return {name:values[f"F_{layer}_"+field] for name, field in
        (("x", "x"), ("z", "original_z"), ("gate", "gate"), ("phi", "phi"))}


def finite_scores(state, head):
    state = state.astype(float)
    z = state/np.sqrt(np.mean(state*state)+head["final_eps"])*head["final_norm"]
    return z@head["final_head"].T


def main(reference, out):
    global ACTIVE
    assert not out.exists()
    out.mkdir(parents=True)
    ACTIVE = out
    source_hashes = bind(out)
    design, status, previous = (read(reference/n) for n in ("design.json", "status.json", "audit.json"))
    assert previous["status"] == "passed" and previous["experiment"] == 263
    for name in ("design", "status"):
        assert sha(reference/(name+".json")) == previous[name+"_sha256"]
    folder, checkpoint = locate(ROOT)
    assert checkpoint == previous["checkpoint"]
    head = read_head(folder)
    with np.load(reference/"norm-weights.npz") as w:
        columns, eps = w["columns"], float(w["eps"])
    values_by_case = []
    saved = {}
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        path = reference/"cases"/(case["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        values_by_case.append((case, np.load(path)))
        saved[case["key"]] = {}
    checks, raw, rows, margins, nonadditivities, counts = [], [], [], [], [], Counter()
    emit(out, out/"design.json", dict(experiment=265, stage="local", retrospective=True,
        source_hashes=source_hashes, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        reference_design_sha256=sha(reference/"design.json"), reference_status_sha256=sha(reference/"status.json"),
        reference_audit_sha256=sha(reference/"audit.json"), checkpoint=checkpoint, cases=design["cases"]))
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            def weight(name):
                return handle.get_tensor(f"model.layers.{layer-1}."+name).float().numpy()
            gamma = weight("post_attention_layernorm.weight")
            weights = {branch:weight("mlp."+branch+"_proj.weight") for branch in ("gate", "up", "down")}
            matrices = {name:torch.tensor(value, device="mps") for name, value in weights.items()}
            down = weights["down"].astype(float)
            for case, values in values_by_case:
                key, start, length = case["key"], case["position"], len(case["false_ids"])
                base = anchor(values, layer)
                for background in ("H", "L"):
                    prefix = f"{background}_{layer}_"
                    x = values[prefix+"x"]
                    expansion(base["x"].astype(float), x.astype(float), base["z"].astype(float), gamma.astype(float), eps, checks, counts)
                    nodes = corners(base, x, gamma, eps, matrices, columns[layer-1], key, layer, length, start)
                    own = nodes[background]
                    for name in ("gate", "up", "phi"):
                        checked(checks, "exact_old_corner_"+name, own[name], values[prefix+name], 0.)
                    for branch in ("gate", "up"):
                        checked(checks, "exact_old_normalized_input", own["z"], values[prefix+branch+"_z"], 0.)
                    checked(checks, "exact_old_product", own["product_adaptive"], values[prefix+"product"], 0.)
                    writes = {}
                    for mode in ("fixed", "adaptive"):
                        writes[mode] = {name:project(v["product_"+mode], matrices["down"], length, start) for name, v in nodes.items()}
                    checked(checks, "exact_old_down_write", writes["adaptive"][background], values[prefix+"write"], 0.)
                    counts["exact_old_full_shape_MLP_replays"] += 1
                    for name, node in nodes.items():
                        for branch in ("gate", "up"):
                            error = float(np.max(abs(node["z"].astype(float)@weights[branch].astype(float).T-node[branch])))
                            raw.append(dict(key=key, background=background, layer=layer, corner=name, node=branch,
                                mode="shared", error=error, original_bound=.0001, passes=error <= .0001))
                        for mode in ("fixed", "adaptive"):
                            error = float(np.max(abs(node["product_"+mode].astype(float)@down.T-writes[mode][name])))
                            raw.append(dict(key=key, background=background, layer=layer, corner=name, node="down",
                                mode=mode, error=error, original_bound=.0001, passes=error <= .0001))
                    product_terms, ideal, product = components(nodes)
                    checked(checks, "fixed_product_component_identity", sum(product_terms[n] for n in ("gate_side", "gate_up_cross", "up_nonadditivity")), ideal, 1e-9)
                    checked(checks, "product_rounding_identity", sum(product_terms.values()), product, 1e-9)
                    vectors = {name:value@down.T for name, value in product_terms.items()}
                    public = {mode:w["H"].astype(float)-w["P"].astype(float)-w["D"].astype(float)+w["L"].astype(float) for mode, w in writes.items()}
                    smooth = product@down.T
                    checked(checks, "projected_component_identity", sum(vectors.values()), smooth, 1e-8)
                    vectors["projection_rounding"] = public["fixed"]-smooth
                    checked(checks, "public_fixed_mixed_closure", sum(vectors.values()), public["fixed"], 1e-8)
                    vectors["mask_effect"] = public["adaptive"]-public["fixed"]
                    vectors.update(fixed=public["fixed"], adaptive=public["adaptive"])
                    for name, vector in vectors.items():
                        saved[key][f"{background}_{layer}_"+name] = vector
                        for scope, pick in (("all", slice(None)), ("verdict", -1)):
                            m = metrics(vector[pick], public["fixed"][pick])
                            rows.append(dict(key=key, **case["group"], background=background, layer=layer, scope=scope,
                                component=name, eligible=m["norm"] >= .001, **m,
                                passes=m["norm"] >= .001 and m["relative"] <= .25 and m["cosine"] >= .95))
                    for scope, pick in (("all", slice(None)), ("verdict", -1)):
                        for name, field in (("gate", "gate"), ("up", "up")):
                            interaction = nodes["H"][field].astype(float)-nodes["P"][field].astype(float)-nodes["D"][field].astype(float)+nodes["L"][field].astype(float)
                            counts["nonadditivity_records"] += 1
                            saved[key][f"{background}_{layer}_{name}_nonadditivity_norm_"+scope] = np.array(np.linalg.norm(interaction[pick]))
                            nonadditivities.append(dict(key=key, background=background, layer=layer, scope=scope,
                                branch=name, norm=float(np.linalg.norm(interaction[pick])), maximum=float(np.max(abs(interaction[pick])))))
                    if layer == 28:
                        for mode, w in writes.items():
                            scores = {name:finite_scores((x+value).astype(np.float32)[-1], head) for name, value in w.items()}
                            checked(checks, "old_finite_head", scores[background] if mode == "adaptive" else finite_scores((x+writes["adaptive"][background]).astype(np.float32)[-1], head), values[background+"_scores"], .005)
                            for name, score in scores.items():
                                margins.append(dict(key=key, **case["group"], background=background, mode=mode,
                                    corner=name, margin=float(score[1]-score[0])))
            print("same-input primitive factorial", layer, "/28", flush=True)
    assert len(rows) == 4*2*28*8*2
    records = []
    for case, values in values_by_case:
        key = case["key"]
        path = out/(key+".npz")
        np.savez_compressed(path, **saved[key])
        records.append(dict(key=key, arrays_sha256=sha(path), bytes=path.stat().st_size))
        values.close()
    names = ("components.csv", "raw-projections.csv", "margins.csv", "nonadditivity.csv")
    for name, data in zip(names, (rows, raw, margins, nonadditivities), strict=True):
        table(out/name, data)
    decisions = {}
    for background in ("H", "L"):
        group = [r for r in rows if r["background"] == background and r["component"] == "gate_up_cross" and r["scope"] == "all" and r["eligible"]]
        decisions[background] = dict(eligible=len(group), passes=sum(r["passes"] for r in group),
            primary_pass=bool(group and sum(r["passes"] for r in group) >= .9*len(group)))
    emit(out, out/"audit.json", dict(status="passed", experiment=265, stage="local", retrospective=True, model_calls=0,
        checks=checks, counts=dict(counts), source_sha256=sha(__file__), design_sha256=sha(out/"design.json"), records=records,
        decisions=decisions, raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        csv_sha256={name:sha(out/name) for name in names},
        limitation="Defined same-input Float32 corner maps and CPU64 component projection; measured arithmetic differences retained, no whole-block derivative or learned-feature claim."))
    print(json.dumps(dict(decisions=decisions, raw_failed=sum(not r["passes"] for r in raw), checks=len(checks)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.reference.resolve(), args.out.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/"failure.json", dict(error=repr(exc)))
        raise
