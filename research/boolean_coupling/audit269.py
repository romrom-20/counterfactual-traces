"""Pre-native numerical qualification and post-native forecast evaluation269."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .amplitude268 import (
    ROOT,
    deltas,
    freeze_fields,
    geometry,
    normalization,
    policy,
)
from .audit229 import checked, sha
from .audit261b import expansion, metrics
from .checkpoint238 import locate, read_head
from .fields267 import (
    PublicHead,
    digest,
    emit,
    node_seal,
    public_norm,
    read,
)
from .forecast269 import ARMS, sources_check, stamp
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


def preaudit(run):
    global ACTIVE
    assert not (run/"forecast-audit.json").exists() and not (run/"failure-pre-audit.json").exists()
    assert not (run/"native-started.json").exists()
    design, status = sources_check(run), read(run/"forecast-status.json")
    assert design["experiment"] == 269 and status["status"] == "complete"
    ACTIVE = run
    for name, value in design["source_hashes"].items():
        assert sha(ROOT/name) == sha(run/"sources"/Path(name).name) == value
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config = read(folder/"config.json")
    eps = config["rms_norm_eps"]
    cpu_head = read_head(folder)
    inputs = []
    checks, counts, raw, head_raw, score_rows = [], Counter(), [], [], []
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        assert case["key"] == entry["key"]
        path = run/"forecasts"/(case["key"]+".npz")
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
                geometry(lparts, checks, checked)
                checked(checks, "fixed_random_unit_norm", np.linalg.norm(star["random_unit"], axis=-1), np.ones(len(star["unit"])), 1e-12)
                for arm in ARMS:
                    if (arm, layer) not in records:
                        assert meta["statuses"][arm]["status"] == "guard_failed"
                        continue
                    record = records[arm, layer]
                    x, qx = (values[arm+"_"+n][i] for n in ("x", "qx"))
                    parts = normalization(x0, x, z0, gamma, eps)
                    geometry(parts, checks, checked)
                    assert {n: digest(parts[n]) for n in ("alpha", "amplitude", "unit", "weighted_norm", "overlap", "energy")} == record["geometry_seal"]
                    if arm not in ("F", "Z", "L", "H", "T"):
                        family = arm[1:]
                        delta = deltas(family, parts, star)
                        magnitude = star["amplitude"] if family in ("00", "01") else parts["amplitude"]
                        expected = abs(magnitude[:, 0])
                        checked(checks, "crossed_D_input_norm", np.linalg.norm(delta["d"], axis=-1), expected, 1e-10)
                        relative = abs(np.linalg.norm(delta["d"].astype(np.float32).astype(float), axis=-1)-expected)/np.maximum(expected, 1e-8)
                        checked(checks, "crossed_Float32_D_input_norm", relative, np.zeros_like(relative), 1e-5)
                        counts["crossed_amplitude_controls"] += 1
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
                    source = "F" if arm in ("F", "Z") else "H"
                    assert record["source_digest"] == records[source, layer]["source_digest"]
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
            print("269 pre-native primitive audit", layer, "/28", flush=True)
        public_head = PublicHead(handle, config)
        for case, values, meta, _ in inputs:
            key, length = case["key"], len(case["false_ids"])
            for arm in ARMS:
                complete = meta["statuses"][arm]["status"] == "complete"
                row = dict(key=key, **case["group"], arm=arm, status=meta["statuses"][arm]["status"])
                if complete:
                    actual = public_head(values[arm+"_final"][-1], length)
                    checked(checks, "exact_public_finite_head", actual, values[arm+"_scores"], 0.)
                    cpu = finite_scores(values[arm+"_final"][-1], cpu_head)
                    error = float(np.max(abs(cpu-values[arm+"_scores"])))
                    head_raw.append(dict(key=key, arm=arm, error=error, original_bound=.005, passes=error <= .005))
                    counts["exact_public_head_replays"] += 1
                    row.update(score_false=float(actual[0]), score_true=float(actual[1]),
                        margin=margin(actual), final_state_digest=digest(values[arm+"_final"]))
                score_rows.append(row)
            checked(checks, "zero_final", values["Z_final"], values["F_final"], 0.)
            checked(checks, "zero_scores", values["Z_scores"], values["F_scores"], 0.)
            values.close()
    for name, data in (("forecast-raw-projections.csv", raw), ("forecast-raw-head.csv", head_raw), ("forecast-scores.csv", score_rows)):
        table(run/name, data)
    emit(run, run/"forecast-audit.json", dict(status="passed", experiment=269, phase="pre-native", ordinary_true=0,
        design_sha256=sha(run/"design.json"), status_sha256=sha(run/"forecast-status.json"),
        source_sha256=sha(__file__), checks=checks, counts=dict(counts), completed=stamp(),
        raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        raw_head_checks=dict(total=len(head_raw), failed=sum(not r["passes"] for r in head_raw), original_bound=.005),
        csv_sha256={p.name: sha(p) for p in run.glob("forecast-*.csv")},
        numerical_boundary="Exact matched-full-shape publicFloat32 primitives, interventions and finite head plus smooth RMS AD/FD; original raw CPU64 failures remain failed, not a whole-model Jacobian certificate."))
    print("269 globally pre-native audit passed", dict(checks=len(checks), counts=dict(counts), raw_failed=sum(not r["passes"] for r in raw)), flush=True)


def evaluate(run):
    global ACTIVE
    assert not (run/"audit.json").exists() and not (run/"failure-final-audit.json").exists()
    design = sources_check(run)
    forecast = read(run/"forecast-status.json")
    native = read(run/"native-status.json")
    assert native["status"] == "complete" and native["ordinary_true"] == 32
    assert native["forecast_seal_sha256"] == sha(run/"forecast-seal.json")
    assert native["native_started_sha256"] == sha(run/"native-started.json")
    assert read(run/"native-started.json")["started"] > read(run/"forecast-seal.json")["created"]
    assert read(run/"forecast-audit.json")["status"] == "passed"
    assert read(run/"forecast-seal.json")["audit_sha256"] == sha(run/"forecast-audit.json")
    ACTIVE = run
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config, cpu_head = read(folder/"config.json"), read_head(folder)
    checks, rows, raw, counts = [], [], [], Counter()
    native_entries = {r["key"]: r for r in native["cases"]}
    with safe_open(folder/"model.safetensors", framework="pt", device="cpu") as handle:
        head = PublicHead(handle, config)
        for case, entry in zip(design["cases"], forecast["cases"], strict=True):
            key, length = case["key"], len(case["false_ids"])
            path = run/"forecasts"/(key+".npz")
            target = run/"observations"/(key+".npz")
            ne = native_entries[key]
            assert sha(path) == entry["arrays_sha256"] and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            assert sha(target) == ne["arrays_sha256"] and sha(target.with_suffix(".json")) == ne["metadata_sha256"]
            meta, tm = read(path.with_suffix(".json")), read(target.with_suffix(".json"))
            truth_records = {r["layer"]: r for r in tm["records"]}
            for record in meta["records"]:
                truth = truth_records[record["layer"]]
                assert record["prefix_digest"] == truth["prefix_digest"]
                if record["arm"] not in ("F", "Z"):
                    assert record["source_digest"] == truth["source_digest"]
                counts["source_prefix_points"] += 1
            with np.load(path) as values, np.load(target) as observed:
                state, score = observed["T_final"], observed["T_scores"]
                checked(checks, "native_public_finite_head", head(state[-1], length), score, 0.)
                error = float(np.max(abs(finite_scores(state[-1], cpu_head)-score)))
                raw.append(dict(key=key, error=error, original_bound=.005, passes=error <= .005))
                counts["native_head_replays"] += 1
                displacement = state.astype(float)-values["F_final"].astype(float)
                for arm in ("L", "H", "A11", "A10", "A01", "A00", "Ar"):
                    row = dict(key=key, **case["group"], arm=arm, status=meta["statuses"][arm]["status"], passes=False)
                    if row["status"] == "complete":
                        prediction = values[arm+"_final"].astype(float)-values["F_final"].astype(float)
                        passed = []
                        for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                            m = metrics(prediction[pick], displacement[pick])
                            row.update({scope+"_"+k: v for k, v in m.items()})
                            passed.append(m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99)
                        row.update(predicted_margin=margin(values[arm+"_scores"]), native_margin=margin(score),
                            margin_error=abs(margin(values[arm+"_scores"])-margin(score)))
                        row["passes"] = bool(all(passed) and row["margin_error"] <= .1)
                    rows.append(row)
    for name, data in (("native-outcomes.csv", rows), ("native-raw-head.csv", raw)):
        table(run/name, data)
    decisions = {a: dict(total=32, passes=sum(r["passes"] for r in rows if r["arm"] == a),
                        guard_failed=sum(r["status"] != "complete" for r in rows if r["arm"] == a))
                 for a in ("L", "H", "A11", "A10", "A01", "A00", "Ar")}
    strata = []
    for arm in decisions:
        for schema in ("generic", "descriptive"):
            for reverse in (0, 1):
                selected = [r for r in rows if r["arm"] == arm and r["schema"] == schema and r["reverse"] == reverse]
                assert len(selected) == 8
                strata.append(dict(arm=arm, schema=schema, reverse=reverse, total=8, passes=sum(r["passes"] for r in selected)))
    emit(run, run/"audit.json", dict(status="passed", experiment=269, phase="post-native", retrospective=False,
        source_sha256=sha(__file__), design_sha256=sha(run/"design.json"),
        forecast_status_sha256=sha(run/"forecast-status.json"), forecast_audit_sha256=sha(run/"forecast-audit.json"),
        seal_sha256=sha(run/"forecast-seal.json"), native_status_sha256=sha(run/"native-status.json"),
        checks=checks, counts=dict(counts), decisions=decisions, strata=strata,
        primary_pass=decisions["A10"]["passes"] == 32, reference_calibrated=decisions["H"]["passes"] == 32,
        raw_native_head=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.005),
        csv_sha256={n: sha(run/n) for n in ("native-outcomes.csv", "native-raw-head.csv")},
        limitation="Fresh literals within dependent templates, one small checkpoint and rich finite/L-forecast budget; predictive accuracy does not pass the failed268 interaction mechanism or identify a learned circuit."))
    print("269 frozen fresh native outcome", decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pre", "final"))
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    try:
        (preaudit if args.phase == "pre" else evaluate)(args.run.resolve())
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE/("failure-"+args.phase+"-audit.json"), dict(error=repr(exc), time=stamp()))
        raise
