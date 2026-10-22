"""Qualify forecast-amplitude/query mediation of the backbone-removal effect."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen2 import modeling_qwen2 as qwen

from .amplitude268 import freeze_fields, geometry, normalization
from .audit229 import checked, sha
from .audit261b import expansion, metrics
from .audit268 import margin, table
from .backbone270 import cache_nodes
from .checkpoint238 import locate, read_head
from .fields267 import ROOT, PublicHead, digest, emit, node_seal, public_norm, read
from .local265 import finite_scores
from .mediation271 import ARMS, policy
from .silu258 import sigmoid

ACTIVE = None


def derivatives(g0, g, checks, counts):
    direction = (g - g0) / max(1., float(np.max(abs(g - g0))))
    sigma = sigmoid(g0)
    analytic = sigma * (1 + g0 * (1 - sigma)) * direction
    x, d = (torch.tensor(v, dtype=torch.float64) for v in (g0, direction))
    _, automatic = torch.func.jvp(torch.nn.functional.silu, (x,), (d,))
    checked(checks, "SiLU_primitive_derivative_AD", automatic.numpy(), analytic, 1e-9)
    for step, bound in ((1e-3, 1e-3), (1e-4, 1e-5)):
        plus, minus = g0 + step * direction, g0 - step * direction
        fd = (plus * sigmoid(plus) - minus * sigmoid(minus)) / (2 * step)
        checked(checks, "SiLU_primitive_derivative_FD", fd, analytic, bound)
    counts["SiLU_AD"] += 1
    counts["SiLU_FD"] += 2


def main(original, reference, previous, controller, run):
    global ACTIVE
    assert not (run / "audit.json").exists() and not (run / "failure-audit.json").exists()
    ACTIVE = run
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 271 and status["status"] == "complete"
    for name, value in design["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == value
    assert sha(qwen.__file__) == design["qwen_source_sha256"]
    for folder, name in ((original, "original_hashes"), (reference, "reference_hashes"),
                         (previous, "previous_hashes"), (controller, "controller_hashes")):
        for file, value in design[name].items():
            assert sha(folder / file) == value
    folder, checkpoint = locate(ROOT)
    assert checkpoint == read(previous / "audit.json")["checkpoint"]
    config = read(folder / "config.json")
    eps = config["rms_norm_eps"]
    cpu_head = read_head(folder)
    checks, counts, raw, head_raw, native_rows, outcomes, amplitudes = (
        [], Counter(), [], [], [], [], [])
    inputs, vectors = [], {}
    control_entries = {e["key"]: e for e in read(controller / "status.json")["cases"]}
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        key = case["key"]
        path = run / "cases" / (key + ".npz")
        assert sha(path) == entry["arrays_sha256"]
        assert sha(path.with_suffix(".json")) == entry["metadata_sha256"]
        meta = read(path.with_suffix(".json"))
        assert meta["arrays_sha256"] == entry["arrays_sha256"]
        assert all(c["error"] <= c["bound"] for c in meta["checks"])
        checks.extend(meta["checks"])
        cp = controller / "cases" / (key + ".npz")
        assert sha(cp) == control_entries[key]["arrays_sha256"]
        assert sha(cp.with_suffix(".json")) == control_entries[key]["metadata_sha256"]
        inputs.append((case, np.load(path), np.load(cp), meta,
                       {(r["arm"], r["layer"]): r for r in meta["records"]}))
        vectors[key] = {}
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        for layer in range(1, 29):
            prefix = f"model.layers.{layer - 1}."
            gamma = handle.get_tensor(prefix + "post_attention_layernorm.weight").float().numpy()
            qgamma = handle.get_tensor(prefix + "input_layernorm.weight").float().numpy()
            weights = {n: handle.get_tensor(prefix + "mlp." + n + "_proj.weight").float().numpy()
                       for n in ("gate", "up", "down")}
            matrices = {n: torch.tensor(w, device="mps") for n, w in weights.items()}
            doubles = {n: w.astype(float) for n, w in weights.items()}
            for case, values, controls, meta, records in inputs:
                key, start, length, i = case["key"], case["position"], len(case["false_ids"]), layer - 1
                x0, z0, qx0, qz0 = (values["F_" + n][i] for n in ("x", "z", "qx", "qz"))
                lp = normalization(x0, values["L_x"][i], z0, gamma, eps)
                star, field_nodes = freeze_fields(lp, matrices, length, start, key, layer)
                cache_nodes(star, field_nodes)
                assert node_seal(field_nodes) == records["L", layer]["frozen_field_nodes"]
                assert {k: digest(v) for k, v in star.items()} == records["L", layer]["frozen_field_seal"]
                parts_by_arm = {}
                for arm in ARMS:
                    if (arm, layer) not in records:
                        assert meta["statuses"][arm]["status"] == "guard_failed"
                        continue
                    record = records[arm, layer]
                    x, qx = (values[arm + "_" + n][i] for n in ("x", "qx"))
                    parts = normalization(x0, x, z0, gamma, eps)
                    control_parts = normalization(x0, controls["B_x"][i], z0, gamma, eps)
                    parts_by_arm[arm] = parts
                    geometry(parts, checks, checked)
                    expansion(x0.astype(float), x.astype(float), z0.astype(float),
                              gamma.astype(float), eps, checks, counts)
                    qlin, qcurv, _ = expansion(qx0.astype(float), qx.astype(float), qz0.astype(float),
                                              qgamma.astype(float), eps, checks, counts)
                    qo, zo = (public_norm(v, w, eps, length, start)
                              for v, w in ((qx, qgamma), (x, gamma)))
                    assert digest(qo) == record["qoriginal_digest"]
                    assert digest(zo) == record["original_z_digest"]
                    if arm in ("Q", "C"):
                        qlin, qcurv, _ = expansion(qx0.astype(float), controls["B_qx"][i].astype(float),
                            qz0.astype(float), qgamma.astype(float), eps, checks, counts)
                        counts["clamped_query_maps"] += 1
                    qi = qo if arm in ("F", "T") else (qlin + qcurv).astype(np.float32)
                    zi = zo if arm in ("F", "T", "H") else parts["linear"].astype(np.float32)
                    assert digest(qi) == record["qinstalled_digest"]
                    assert digest(zi) == record["installed_z_digest"]
                    assert {n: digest(parts[n]) for n in ("alpha", "amplitude", "unit",
                            "weighted_norm", "overlap", "energy")} == record["geometry_seal"]
                    write, nodes, terms = policy(arm, parts, control_parts, zo, star, matrices, length, start, key, layer)
                    assert node_seal(nodes) == record["nodes"]
                    assert {k: digest(v) for k, v in terms.items()} == record["mechanism_seal"]
                    assert digest(write) == record["write_digest"]
                    assert digest(x + write) == record["block_digest"]
                    assert record["prefix_digest"] == records["F", layer]["prefix_digest"]
                    source = "F" if arm in ("F", "Z") else "T"
                    assert record["source_digest"] == records[source, layer]["source_digest"]
                    if layer == 28 and meta["statuses"][arm]["status"] == "complete":
                        checked(checks, "final_receiving_state", x + write, values[arm + "_final"][1:], 0.)
                    for corner, node in nodes.items():
                        for branch, operand, field in (("gate", "z", "gate"), ("up", "z", "up"),
                                                       ("down", "product", "write")):
                            error = float(np.max(abs(node[operand].astype(float) @ doubles[branch].T - node[field])))
                            raw.append(dict(key=key, arm=arm, layer=layer, corner=corner,
                                            node=branch, error=error, original_bound=.0001,
                                            passes=error <= .0001))
                    if terms:
                        derivatives(star["backbone_gate"].astype(float),
                                    nodes["base"]["gate"].astype(float), checks, counts)
                        checked(checks, "finite_product_ledger", terms["linear"] + terms["gate"]
                                + terms["cross"] + terms["arithmetic"], nodes["base"]["product"], 1e-9)
                        checked(checks, "smooth_gate_bridge_identity", terms["gate_smooth"]
                                + terms["gate_arithmetic"], terms["gate"], 1e-9)
                        if arm in ("S", "C"):
                            checked(checks, "clamped_controller_p", terms["controller_p"], control_parts["base"], 0.)
                            checked(checks, "clamped_controller_k", terms["controller_k"], control_parts["amplitude"], 0.)
                            counts["clamped_amplitude_points"] += 1
                    counts["policy_points"] += 1
                    counts["public_MLP_node_seals"] += len(nodes)
                if "B" in parts_by_arm:
                    for arm in ("J", "S", "Q", "C"):
                        if arm not in parts_by_arm:
                            continue
                        for row in range(len(x0)):
                            for name in ("alpha", "amplitude"):
                                b, baseline, a = (float(p[name][row, 0]) for p in
                                           (parts_by_arm["B"], lp, parts_by_arm[arm]))
                                amplitudes.append(dict(key=key, **case["group"], layer=layer,
                                    row=row, verdict=row == len(x0) - 1, arm=arm, amplitude=name,
                                    baseline=baseline, full=b, intervened=a, difference=a - b,
                                    full_minus_baseline=b - baseline))
            print("271 primitive audit", layer, "/28", flush=True)
        public_head = PublicHead(handle, config)
        for case, values, controls, meta, _ in inputs:
            key, length = case["key"], len(case["false_ids"])
            for arm in ARMS:
                if meta["statuses"][arm]["status"] != "complete":
                    continue
                checked(checks, "exact_public_finite_head", public_head(values[arm + "_final"][-1], length),
                        values[arm + "_scores"], 0.)
                error = float(np.max(abs(finite_scores(values[arm + "_final"][-1], cpu_head)
                                         - values[arm + "_scores"])))
                head_raw.append(dict(key=key, arm=arm, error=error, original_bound=.005,
                                     passes=error <= .005))
                counts["exact_public_head_replays"] += 1
            native = values["T_final"].astype(float) - values["F_final"].astype(float)
            for arm in ("L", "B", "J", "S", "Q", "C"):
                row = dict(key=key, **case["group"], arm=arm, passes=False,
                           status=meta["statuses"][arm]["status"])
                if row["status"] == "complete":
                    pred = values[arm + "_final"].astype(float) - values["F_final"].astype(float)
                    ok = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = metrics(pred[pick], native[pick])
                        row.update({scope + "_" + k: v for k, v in m.items()})
                        ok.append(m["norm"] >= .001 and m["relative"] <= .05 and m["cosine"] >= .99)
                    row["margin_error"] = abs(margin(values[arm + "_scores"]) - margin(values["T_scores"]))
                    row["passes"] = bool(all(ok) and row["margin_error"] <= .1)
                native_rows.append(row)
            if all(meta["statuses"][a]["status"] == "complete" for a in ("B", "J", "S", "Q", "C")):
                target = values["B_final"].astype(float) - values["J_final"].astype(float)
                final = {a: values[a + "_final"].astype(float) - values["J_final"].astype(float)
                         for a in ("S", "Q", "C")}
                final["interaction"] = final["C"] - final["S"] - final["Q"]
                final["unmediated"] = target - final["C"]
                final["target"] = target
                for arm in ("S", "Q", "C"):
                    row = dict(key=key, **case["group"], arm=arm)
                    ok = []
                    for scope, pick in (("all", slice(1, None)), ("verdict", -1)):
                        m = metrics(final[arm][pick], target[pick])
                        row.update({scope + "_" + k: v for k, v in m.items()})
                        ok.append(m["norm"] >= .001 and m["relative"] <= .25 and m["cosine"] >= .95)
                    anchor = values["B_final"]
                    endpoint = (anchor.astype(float) - final[arm]).astype(np.float32)
                    ma, mt, mc = (margin(public_head(v[-1], length))
                                  for v in (anchor, values["J_final"], endpoint))
                    row.update(margin_error=abs(mc - mt), margin_bound=.005 + .25 * abs(ma - mt))
                    row["passes"] = bool(all(ok) and row["margin_error"] <= row["margin_bound"])
                    outcomes.append(row)
                vectors[key].update({"final_" + k: v for k, v in final.items()})
            np.savez_compressed(run / ("vectors-" + key + ".npz"),
                                **{k: np.stack(v) if isinstance(v, list) else v
                                   for k, v in vectors[key].items()})
            values.close()
            controls.close()
    for name, rows in (("native.csv", native_rows), ("restoration.csv", outcomes),
                       ("amplitudes.csv", amplitudes),
                       ("raw-projections.csv", raw), ("raw-head.csv", head_raw)):
        table(run / name, rows)
    native_counts = {a: sum(r["passes"] for r in native_rows if r["arm"] == a)
                     for a in ("L", "B", "J", "S", "Q", "C")}
    restoration_counts = {a: sum(r["passes"] for r in outcomes if r["arm"] == a) for a in ("S", "Q", "C")}
    calibrated = native_counts["B"] == 4 and read(controller / "audit.json")["native_calibration"]
    emit(run, run / "audit.json", dict(status="passed", experiment=271, retrospective=True,
        source_sha256=sha(__file__), design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"), checkpoint=checkpoint, checks=checks, counts=dict(counts),
        native_counts=native_counts, restoration_counts=restoration_counts, native_calibration=calibrated,
        primary_joint_controller_sufficient=bool(calibrated and restoration_counts["C"] >= 3),
        raw_projection_checks=dict(total=len(raw), failed=sum(not r["passes"] for r in raw), original_bound=.0001),
        raw_head_checks=dict(total=len(head_raw), failed=sum(not r["passes"] for r in head_raw), original_bound=.005),
        csv_sha256={p.name: sha(p) for p in run.glob("*.csv")},
        vectors_sha256={p.name: sha(p) for p in run.glob("vectors-*.npz")},
        numerical_boundary="Exact public Float32 node/install/head replay; smooth RMS and SiLU primitive derivatives only. Retain measured arithmetic, old raw failures and disputed whole-block derivatives.",
        limitation="Four old dependent contexts; rich finite attention/correction/L plus B-controller budget. Retrospective mediation diagnostic, no compact predictor, learned circuit or complete Question B solution."))
    print(dict(native=native_counts, restoration=restoration_counts, calibrated=calibrated,
               raw_failed=sum(not r["passes"] for r in raw)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "reference", "previous", "controller", "run"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        main(*(getattr(args, n).resolve() for n in ("original", "reference", "previous", "controller", "run")))
    except Exception as exc:
        if ACTIVE is not None:
            emit(ACTIVE, ACTIVE / "failure-audit.json", dict(error=repr(exc)))
        raise
