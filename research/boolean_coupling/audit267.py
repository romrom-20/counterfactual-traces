"""Qualify267's finite field interventions and evaluate its frozen factorial."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .audit229 import checked, sha
from .audit261b import expansion, metrics
from .checkpoint238 import locate, read_head
from .fields267 import (
    ARMS,
    FAMILIES,
    ROOT,
    PublicHead,
    digest,
    emit,
    freeze_fields,
    node_seal,
    normalization,
    policy,
    public_norm,
    read,
)
from .local265 import finite_scores

ACTIVE = None


def table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def margin(scores):
    return float(np.diff(scores.astype(float))[0])


def main(original, reference, run):
    global ACTIVE
    assert not (run/"audit.json").exists() and not (run/"failure-audit.json").exists()
    design, status = read(run/"design.json"), read(run/"status.json")
    assert design["experiment"] == 267 and status["status"] == "complete"
    ACTIVE = run
    for name, value in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == value
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    for folder, name in ((original, "original_hashes"), (reference, "reference_hashes")):
        for file, value in design[name].items():
            assert sha(folder/file) == value
    folder, checkpoint = locate(ROOT)
    assert checkpoint == read(reference/"audit.json")["checkpoint"]
    config = read(folder/"config.json")
    eps = config["rms_norm_eps"]
    cpu_head = read_head(folder)
    inputs = []
    checks, counts, raw, outcomes, native_rows, contrasts, head_raw = [], Counter(), [], [], [], [], []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        assert case["key"] == entry["key"]
        path = run/"cases"/(case["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert meta["arrays_sha256"] == entry["arrays_sha256"]
        inputs.append((case, np.load(path), meta, {(r["arm"], r["layer"]): r for r in meta["records"]}))
        checks.extend(meta["checks"])
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            prefix = f"model.layers.{layer-1}."
            gamma = handle.get_tensor(prefix+"post_attention_layernorm.weight").float().numpy()
            qgamma = handle.get_tensor(prefix+"input_layernorm.weight").float().numpy()
            weights = {n: handle.get_tensor(prefix+"mlp."+n+"_proj.weight").float().numpy()
                       for n in ("gate", "up", "down")}
            matrices = {n: torch.tensor(w, device="mps") for n, w in weights.items()}
            doubles = {n: w.astype(float) for n, w in weights.items()}
            for case, values, meta, records in inputs:
                key, start, length = case["key"], case["position"], len(case["false_ids"])
                i = layer-1
                x0, z0, qx0, qz0 = (values["F_"+n][i] for n in ("x", "z", "qx", "qz"))
                lparts = normalization(x0, values["L_x"][i], z0, gamma, eps)
                star, field_nodes = freeze_fields(lparts, matrices, length, start, key, layer)
                assert node_seal(field_nodes) == records["L", layer]["frozen_field_nodes"]
                assert {k: digest(v) for k, v in star.items()} == records["L", layer]["frozen_field_seal"]
                for tag in ("p", "d"):
                    target = np.linalg.norm(star["c"+tag], axis=-1)
                    error = abs(np.linalg.norm(star["r"+tag].astype(float), axis=-1)-target)/np.maximum(target, 1e-8)
                    checked(checks, "random_field_matched_write_norm", error, np.zeros_like(error), 1e-5)
                for arm in ARMS:
                    if (arm, layer) not in records:
                        assert meta["statuses"][arm]["status"] == "guard_failed"
                        continue
                    record = records[arm, layer]
                    x, qx = (values[arm+"_"+n][i] for n in ("x", "qx"))
                    parts = normalization(x0, x, z0, gamma, eps)
                    expansion(x0.astype(float), x.astype(float), z0.astype(float), gamma.astype(float), eps, checks, counts)
                    qlin, qcurv, _ = expansion(qx0.astype(float), qx.astype(float), qz0.astype(float), qgamma.astype(float), eps, checks, counts)
                    qo = public_norm(qx, qgamma, eps, length, start)
                    zo = public_norm(x, gamma, eps, length, start)
                    assert digest(qo) == record["qoriginal_digest"]
                    assert digest(zo) == record["original_z_digest"]
                    qi = qo if arm in ("F", "T") else (qlin+qcurv).astype(np.float32)
                    assert digest(qi) == record["qinstalled_digest"]
                    zi = zo if arm in ("F", "T", "H") else parts["linear"].astype(np.float32)
                    assert digest(zi) == record["installed_z_digest"]
                    write, nodes = policy(arm, parts, zo, star, matrices, length, start)
                    assert node_seal(nodes) == record["nodes"]
                    assert digest(write) == record["write_digest"]
                    assert digest(x+write) == record["block_digest"]
                    assert record["prefix_digest"] == records["F", layer]["prefix_digest"]
                    source = "F" if arm in ("F", "Z") else "T"
                    assert record["source_digest"] == records[source, layer]["source_digest"]
                    if arm in ("P11", "D11"):
                        checked(checks, "direct_finite_corner", write, nodes[arm[0].lower()]["write"], 0.)
                    if meta["statuses"][arm]["status"] == "complete" and layer == 28:
                        checked(checks, "final_receiving_state", x+write, values[arm+"_final"][1:], 0.)
                        assert digest(values[arm+"_final"][0]) == record["source_digest"]
                    for corner, node in nodes.items():
                        for branch, operand, field in (("gate", "z", "gate"), ("up", "z", "up"), ("down", "product", "write")):
                            error = float(np.max(abs(node[operand].astype(float)@doubles[branch].T-node[field])))
                            raw.append(dict(key=key, arm=arm, layer=layer, corner=corner, node=branch,
                                error=error, original_bound=.0001, passes=error <= .0001))
                    counts["exact_field_points"] += 1
                    counts["public_norm_replays"] += 2
                    counts["public_MLP_node_seals"] += len(nodes)
            del matrices, doubles, weights
            print("267 separate primitive audit", layer, "/28", flush=True)
        public_head = PublicHead(handle, config)
        for case, values, meta, _ in inputs:
            key, length = case["key"], len(case["false_ids"])
            for arm in ARMS:
                if meta["statuses"][arm]["status"] != "complete":
                    continue
                checked(checks, "exact_public_finite_head", public_head(values[arm+"_final"][-1], length), values[arm+"_scores"], 0.)
                cpu = finite_scores(values[arm+"_final"][-1], cpu_head)
                error = float(np.max(abs(cpu-values[arm+"_scores"])))
                head_raw.append(dict(key=key, arm=arm, error=error, original_bound=.005, passes=error <= .005))
                counts["exact_public_head_replays"] += 1
            checked(checks, "zero_final", values["Z_final"], values["F_final"], 0.)
            checked(checks, "zero_scores", values["Z_scores"], values["F_scores"], 0.)
            native = values["T_final"].astype(float)-values["F_final"].astype(float)
            for arm in ARMS:
                if arm in ("F", "Z", "T"):
                    continue
                row = dict(key=key, **case["group"], arm=arm, status=meta["statuses"][arm]["status"], passes=False)
                if row["status"] == "complete":
                    prediction = values[arm+"_final"].astype(float)-values["F_final"].astype(float)
                    ok = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = metrics(prediction[pick], native[pick])
                        row.update({scope+"_"+k: v for k, v in m.items()})
                        ok.append(m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99)
                    row["margin_error"] = abs(margin(values[arm+"_scores"])-margin(values["T_scores"]))
                    row["passes"] = bool(all(ok) and row["margin_error"] <= .1)
                native_rows.append(row)
            signed = {}
            complete = {}
            for family in FAMILIES:
                complete[family] = all(meta["statuses"][p+family]["status"] == "complete" for p in ("P", "D", "A"))
                if complete[family]:
                    signed[family] = (values["A"+family+"_final"].astype(float)
                        -values["P"+family+"_final"].astype(float)
                        -values["D"+family+"_final"].astype(float)+values["L_final"].astype(float))
            for family in FAMILIES:
                row = dict(key=key, **case["group"], family=family, complete=complete[family], passes=False)
                if complete["11"] and complete[family]:
                    target, candidate = signed["11"], signed[family]
                    ok = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = metrics(candidate[pick], target[pick])
                        row.update({scope+"_"+k: v for k, v in m.items()})
                        ok.append(m["norm"] >= .001 and m["relative"] <= .25 and m["cosine"] >= .95)
                    anchor = values["A11_final"].astype(float)
                    target_endpoint = (anchor-target).astype(np.float32)
                    endpoint = (anchor-candidate).astype(np.float32)
                    ma, mt, mc = (margin(public_head(v[-1], length)) for v in (anchor.astype(np.float32), target_endpoint, endpoint))
                    row.update(anchor_margin=ma, target_endpoint_margin=mt, candidate_endpoint_margin=mc,
                        margin_error=abs(mc-mt), margin_bound=.005+.25*abs(ma-mt))
                    row["passes"] = bool(all(ok) and row["margin_error"] <= row["margin_bound"])
                outcomes.append(row)
            if all(complete[s] for s in ("11", "10", "01", "00")):
                signed["generator_ablation"] = signed["11"]-signed["01"]
                signed["reader_ablation"] = signed["11"]-signed["10"]
                signed["adaptation_interaction"] = signed["11"]-signed["10"]-signed["01"]+signed["00"]
                for term in ("generator_ablation", "reader_ablation", "adaptation_interaction"):
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        contrasts.append(dict(key=key, term=term, scope=scope,
                            vector_norm=float(np.linalg.norm(signed[term][pick])), **metrics(signed[term][pick], signed["11"][pick])))
            path = run/("signed-fields-"+key+".npz")
            assert not path.exists()
            np.savez_compressed(path, **signed)
            values.close()
    for name, rows in (("factorial.csv", outcomes), ("native.csv", native_rows),
                       ("contrasts.csv", contrasts), ("raw-projections.csv", raw), ("raw-head.csv", head_raw)):
        table(run/name, rows)
    decisions = {s: dict(total=4, passes=sum(r["passes"] for r in outcomes if r["family"] == s),
                        complete=sum(r["complete"] for r in outcomes if r["family"] == s)) for s in FAMILIES}
    calibration = {a: sum(r["passes"] for r in native_rows if r["arm"] == a) for a in ("H", "A11")}
    emit(run, run/"audit.json", dict(status="passed", experiment=267, retrospective=True,
        source_sha256=sha(__file__), design_sha256=sha(run/"design.json"), status_sha256=sha(run/"status.json"),
        checkpoint=checkpoint, checks=checks, counts=dict(counts), decisions=decisions,
        preserved={s: decisions[s]["passes"] >= 3 for s in FAMILIES}, native_calibration=calibration,
        native_connection_qualified=all(v == 4 for v in calibration.values()),
        random_direction_control_qualified=decisions["r"]["complete"] == 4,
        raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        raw_head_checks=dict(total=len(head_raw), failed=sum(not r["passes"] for r in head_raw), original_bound=.005),
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        vectors_sha256={p.name: sha(p) for p in run.glob("signed-fields-*.npz")},
        numerical_boundary="Exact matched-full-shape publicFloat32 primitive/installation/head scope and checked smooth RMS only; all original raw CPU64 failures retained. No whole-model Jacobian certificate.",
        limitation="Four old contexts; extra L forecast and primitive evaluations; intervention-specific sufficiency, no fresh confirmation, unique learned circuit or complete Question B solution."))
    print(dict(decisions=decisions, calibration=calibration, raw_failed=sum(not r["passes"] for r in raw)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.original.resolve(), args.reference.resolve(), args.run.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/"failure-audit.json", dict(error=repr(exc)))
        raise
