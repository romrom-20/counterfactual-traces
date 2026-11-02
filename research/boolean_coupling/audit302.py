"""Finite first-source arithmetic and signed conditional effect audit; no derivatives."""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .audit229 import sha, softmax
from .audit272 import read, table
from .audit296 import head
from .cache272 import save_json
from .checkpoint238 import locate, weights
from .geometry302 import ARMS, BOUNDS, ROOT, delta, metrics, record, statistics
from .run302 import load_parents


def complex_rotate(x, angle):
    h = x.shape[-1] // 2
    z = x[..., :h] + 1j * x[..., h:]
    y = z * np.exp(1j * angle)
    return np.concatenate((y.real, y.imag), -1)


def finite_mlp(w, mid):
    z = mid / np.sqrt(np.mean(mid * mid) + float(w["eps"])) * w["norm"]
    gate = z @ w["gate_weight"].T + w.get("gate_bias", 0.0)
    up = z @ w["up_weight"].T + w.get("up_bias", 0.0)
    return (gate / (1 + np.exp(-np.clip(gate, -700, 700))) * up) @ w["down_weight"].T + w.get(
        "down_bias", 0.0
    )


def next_value(w, state):
    z = state / np.sqrt(np.mean(state * state) + float(w["next_eps"])) * w["next_norm"]
    return (z @ w["next_v_weight"].T + w["next_v_bias"]).reshape(-1, int(w["width"]))


def geometry_checks(case, p, c, checks):
    idx = case["alias_changed"]
    dk = p["B_F_0_recipient_k"][:, idx].astype(float) - p["A_F_0_recipient_k"][:, idx].astype(float)
    dv = p["B_F_0_recipient_v"][:, idx].astype(float) - p["A_F_0_recipient_v"][:, idx].astype(float)
    record(checks, "key_difference", c["dk"], dk)
    record(checks, "value_difference", c["dv"], dv)
    record(checks, "hidden_seen_closure", c["hidden"] + c["seen"], dk, BOUNDS["geometry"])
    for alias in "AB":
        for name in ("query", "recipient_k", "recipient_v"):
            a = p[alias + "_F_0_" + name]
            b = p[alias + "_T_0_" + name]
            # Only the Boolean self row changes K/V; earlier prefix is invariant.
            if name != "query":
                record(checks, alias + "/bit_prefix_" + name, a[:, :-1], b[:, :-1])
    for bit in ("F", "T"):
        record(
            checks,
            bit + "/same_initial_query",
            p["A_" + bit + "_0_query"],
            p["B_" + bit + "_0_query"],
        )
    qf = p["A_F_0_query"].astype(float)[:, 0]
    groups = int(p["A_F_0_groups"])
    for g in range(dk.shape[0]):
        q = qf[g * groups : (g + 1) * groups]
        for name in ("hidden", "random_hidden"):
            record(
                checks,
                name + f"/{g}/False_invisible",
                q @ c[name][g].T,
                np.zeros((len(q), len(idx))),
                BOUNDS["geometry"],
            )
        # A second construction checks the SVD span and all shared query heads.
        expected = dk[g] - dk[g] @ np.linalg.pinv(q, rcond=BOUNDS["rank_relative"]) @ q
        record(checks, f"{g}/independent_projector", c["hidden"][g], expected, BOUNDS["geometry"])
    for name, target in [("random_hidden", "hidden"), ("random_key", "dk"), ("random_value", "dv")]:
        a, b = np.linalg.norm(c[name], axis=-1), np.linalg.norm(c[target], axis=-1)
        record(
            checks,
            name + "/norm",
            (a - b) / np.maximum(b, 1e-30),
            np.zeros_like(a),
            BOUNDS["random_relative"],
        )
    angle = (case["position"] - np.array(idx))[None, :, None] * c["inv"]
    record(
        checks,
        "phase_orientation",
        c["phase_difference"],
        complex_rotate(dk, angle),
        BOUNDS["geometry"],
    )
    record(
        checks,
        "phase_norm",
        np.linalg.norm(c["phase_difference"], axis=-1),
        np.linalg.norm(dk, axis=-1),
        BOUNDS["geometry"],
    )


def qualify(case, p, c, data, w, hw, checks, raw):
    geometry_checks(case, p, c, checks)
    for arm in ARMS:
        for alias in "AB":
            for bit in (False, True):
                tag = f"{arm}_{alias}_" + ("T" if bit else "F")
                if tag + "_source" not in data:
                    continue
                prefix = alias + ("_T_0_" if bit else "_F_0_")
                q = p[prefix + "query"].astype(float)
                k = p[prefix + "recipient_k"].astype(float).copy()
                v = p[prefix + "recipient_v"].astype(float).copy()
                sign = 1 if alias == "A" else -1
                idx = case["alias_changed"]
                if arm in ("K", "KV"):
                    k[:, idx] += sign * c["dk"]
                if arm in ("V", "KV"):
                    v[:, idx] += sign * c["dv"]
                name = {"H": "hidden", "S": "seen", "RH": "random_hidden", "RK": "random_key"}.get(
                    arm
                )
                if name:
                    k[:, idx] += sign * c[name]
                if arm == "RV":
                    v[:, idx] += sign * c["random_value"]
                if arm == "P":
                    k[:, idx] += sign * (c["dk"] - c["phase_difference"]) / 2
                if arm == "G":
                    q, k = complex_rotate(q, c["inv"]), complex_rotate(k, c["inv"])
                q, k, v = [x.astype(np.float32).astype(float) for x in (q, k, v)]
                record(checks, tag + "/actual_query", data[tag + "_query"], q)
                record(checks, tag + "/actual_key_rows", data[tag + "_key"], k[:, idx])
                record(checks, tag + "/actual_value_rows", data[tag + "_value"], v[:, idx])
                groups, scale = int(p[prefix + "groups"]), float(p[prefix + "scaling"])
                logits = q @ np.repeat(k, groups, 0).swapaxes(-1, -2) * scale
                prob = softmax(logits + p[prefix + "mask"])
                y = prob @ np.repeat(v, groups, 0)
                raw.append(
                    dict(
                        key=case["key"],
                        arm=tag,
                        kind="scaled_QK",
                        error=float(np.max(abs(logits - data[tag + "_raw"]))),
                        bound=BOUNDS["raw"],
                    )
                )
                record(
                    checks, tag + "/CPU_probability", prob, data[tag + "_prob"], BOUNDS["composed"]
                )
                record(checks, tag + "/CPU_PV", y, data[tag + "_head"], BOUNDS["composed"])
                write = data[tag + "_head"].transpose(1, 0, 2).reshape(-1) @ w["o_weight"].T
                err = float(np.max(abs(write - data[tag + "_write"])))
                raw.append(
                    dict(
                        key=case["key"],
                        arm=tag,
                        kind="output_projection",
                        error=err,
                        bound=BOUNDS["raw"],
                    )
                )
                record(
                    checks,
                    tag + "/output_abs",
                    write,
                    data[tag + "_write"],
                    BOUNDS["projection_abs"],
                )
                record(
                    checks,
                    tag + "/output_relative",
                    np.linalg.norm(write - data[tag + "_write"])
                    / max(np.linalg.norm(data[tag + "_write"]), 1e-30),
                    0.0,
                    BOUNDS["projection_relative"],
                )
                embed = w["embedding"][int(bit)].astype(np.float32)
                mid = (embed + data[tag + "_write"].astype(np.float32)).astype(np.float32)
                record(checks, tag + "/residual_add", mid, data[tag + "_mid"])
                mlp = finite_mlp(w, mid.astype(float))
                record(checks, tag + "/finite_MLP", mlp, data[tag + "_mlp"], BOUNDS["mlp"])
                record(
                    checks,
                    tag + "/MLP_relative",
                    np.linalg.norm(mlp - data[tag + "_mlp"])
                    / max(np.linalg.norm(data[tag + "_mlp"]), 1e-30),
                    0.0,
                    BOUNDS["state_relative"],
                )
                state = (mid + data[tag + "_mlp"].astype(np.float32)).astype(np.float32)
                record(checks, tag + "/MLP_residual_add", state, data[tag + "_source"])
                emit = next_value(w, state.astype(float))
                record(checks, tag + "/next_value", emit, data[tag + "_emit"], BOUNDS["mlp"])
                record(
                    checks,
                    tag + "/finite_head",
                    head(hw, data[tag + "_final"][-1]),
                    data[tag + "_scores"],
                    BOUNDS["head"],
                )
                if arm in ("H", "RH") and not bit:
                    n = f"N_{alias}_F"
                    record(
                        checks,
                        tag + "/False_probability_preserved",
                        data[tag + "_prob"],
                        data[n + "_prob"],
                        BOUNDS["composed"],
                    )
                    record(
                        checks,
                        tag + "/False_source_preserved",
                        data[tag + "_source"],
                        data[n + "_source"],
                        BOUNDS["mlp"],
                    )


def population(rows, arm):
    rr = [r for r in rows if r["arm"] == arm]
    layouts = {
        layout: sum(r["passes"] for r in rr if r["layout"] == layout)
        for layout in sorted({r["layout"] for r in rr})
    }
    orders = {str(o): sum(r["passes"] for r in rr if r["mention_order"] == o) for o in (0, 1)}
    count = sum(r["passes"] for r in rr)
    return dict(
        passes=count,
        local_passes=sum(r["local_passes"] for r in rr),
        total=len(rr),
        layouts=layouts,
        mention_orders=orders,
        sufficient=count >= 6
        and all(v >= 1 for v in layouts.values())
        and all(v >= 3 for v in orders.values()),
    )


def main(run, parent, phase):
    design = read(run / "design.json")
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == h
    for n, h in design["parent_files"].items():
        assert sha(parent / n) == h
    folder, checkpoint = locate(ROOT)
    assert checkpoint == design["checkpoint"]
    config = read(folder / "config.json")
    with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as handle:
        w = weights(handle, config, 0, "mlp")
        for field, name in [
            ("o_weight", "model.layers.0.self_attn.o_proj.weight"),
            ("next_norm", "model.layers.1.input_layernorm.weight"),
            ("next_v_weight", "model.layers.1.self_attn.v_proj.weight"),
            ("next_v_bias", "model.layers.1.self_attn.v_proj.bias"),
        ]:
            w[field] = handle.get_tensor(name).to(torch.float32).numpy().astype(float)
        emb = handle.get_slice("model.embed_tokens.weight")
        w["embedding"] = (
            torch.cat([emb[i : i + 1] for i in (895, 830)]).float().numpy().astype(float)
        )
    w.update(
        next_eps=np.array(config["rms_norm_eps"]),
        width=np.array(config["hidden_size"] // config["num_attention_heads"]),
    )
    with np.load(parent / "head.npz") as z:
        hw = {n: z[n] for n in z.files}
    checks, raw, rows, vec, phases, ledgers = [], [], [], [], [], []
    statuses = ["calibration"] if phase == "calibration" else ["calibration", "intervention"]
    indices = {
        s: {e["key"]: e for e in read(run / (s + "-status.json"))["cases"]} for s in statuses
    }
    for case in design["cases"]:
        key, data = case["key"], {}
        for status in statuses:
            folder_name = "interventions" if status == "intervention" else status
            path = run / folder_name / (key + ".npz")
            entry = indices[status][key]
            assert (
                sha(path) == entry["sha256"]
                and sha(path.with_suffix(".json")) == entry["metadata_sha256"]
            )
            with np.load(path) as z:
                data.update({n: z[n] for n in z.files})
            checks.extend(dict(key=key, **r) for r in read(path.with_suffix(".json"))["checks"])
        with np.load(run / "components" / (key + ".npz")) as z:
            comp = {n: z[n] for n in z.files}
        p, _, _ = load_parents(parent, case)
        start = len(checks)
        qualify(case, p, comp, data, w, hw, checks, raw)
        for r in checks[start:]:
            r["key"] = key
        if phase == "complete":
            rr, vv = metrics(data, case)
            rows.extend(rr)
            vec.extend(vv)
            target = delta(data, "N", "B", "source") - delta(data, "N", "A", "source")
            alternate = delta(data, "P", "B", "source") - delta(data, "P", "A", "source")
            ratio = np.linalg.norm(alternate) / max(np.linalg.norm(target), 1e-30)
            phases.append(
                dict(
                    key=key,
                    layout=case["group"]["block"],
                    mention_order=case["group"]["mention_order"],
                    native_norm=float(np.linalg.norm(target)),
                    remaining_norm_ratio=float(ratio),
                    passes=bool(np.linalg.norm(target) >= 0.001 and ratio <= 0.25),
                    **{k: v for k, v in statistics(alternate, target).items() if k != "norm"},
                )
            )
            for name in ("write", "mlp"):
                value = delta(data, "N", "B", name) - delta(data, "N", "A", name)
                ledgers.append(dict(key=key, component=name, **statistics(value, target)))
    for r in raw:
        r["passes"] = bool(r["error"] <= r["bound"])
    suffix = "calibration-" if phase == "calibration" else ""
    table(run / (suffix + "checks.csv"), checks)
    table(run / (suffix + "raw.csv"), raw)
    passed = all(r["passes"] for r in checks)
    result = dict(
        experiment=302,
        status="passed" if passed else "failed",
        required_checks=len(checks),
        failed_checks=sum(not r["passes"] for r in checks),
        raw_checks=len(raw),
        raw_failed=sum(not r["passes"] for r in raw),
        scientific_outcomes_withheld=not passed,
        retrospective=True,
        general_question_B_solved=False,
    )
    if phase == "complete" and passed:
        table(run / "effects.csv", rows)
        table(run / "phase.csv", phases)
        table(run / "ledger.csv", ledgers)
        save_json(run / "vector-records.json", dict(records=vec))
        pops = {arm: population(rows, arm) for arm in ("K", "V", "H", "S", "RH", "RK", "RV")}
        result.update(
            populations=pops,
            H_key=pops["K"]["sufficient"]
            and pops["V"]["passes"] <= 2
            and pops["RK"]["passes"] <= 2,
            H_value=pops["V"]["sufficient"]
            and pops["K"]["passes"] <= 2
            and pops["RV"]["passes"] <= 2,
            H_hidden=pops["H"]["sufficient"]
            and pops["S"]["passes"] <= 2
            and pops["RH"]["passes"] <= 2,
            H_phase=sum(r["passes"] for r in phases) >= 6
            and all(
                sum(r["passes"] for r in phases if r["layout"] == v) >= 1
                for v in {r["layout"] for r in phases}
            )
            and all(
                sum(r["passes"] for r in phases if r["mention_order"] == o) >= 3 for o in (0, 1)
            ),
            material_both_references=sum(
                r["A_material"] and r["B_material"] for r in rows if r["arm"] == "K"
            ),
        )
    result["csv_sha256"] = {p.name: sha(p) for p in run.glob(suffix + "*.csv")}
    save_json(run / ("calibration-audit.json" if phase == "calibration" else "audit.json"), result)
    print(result, flush=True)
    assert passed, [r for r in checks if not r["passes"]][:8]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--phase", choices=["calibration", "complete"], required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.parent.resolve(), args.phase)
