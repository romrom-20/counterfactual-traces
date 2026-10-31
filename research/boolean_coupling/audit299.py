"""Separate NumPy operand, normalization, query, attention and finite-head checks."""

import argparse
import re
from pathlib import Path

import numpy as np

from .audit229 import sha, softmax
from .audit272 import read, table
from .audit296 import head
from .cache272 import save_json
from .generator299 import ARMS, BOUNDS, EFFECT, PRIMARY, ROOT, record
from .subspace293 import statistics


def audit_bundle(case, data, witness, meta, w, phase, checks, raw, false_data, maps):
    first = len(checks)
    checks.extend(dict(key=case["key"], **r) for r in meta["checks"])
    arms = ("F", *ARMS, "ZC", "ZM", "ZB") if phase == "forecast" else ("T", "O")
    for arm in arms:
        record(
            checks,
            arm + "/finite_head",
            head(w, data[arm + "_final"][-1]),
            data[arm + "_scores"],
            BOUNDS["head"],
        )
        if not arm.startswith("Z"):
            record(
                checks, arm + "/trace_endpoint", data[arm + "_trace"][-1], data[arm + "_final"][1:]
            )
    if phase == "forecast":
        for j in range(len(data["F_trace"])):
            record(
                checks,
                f"F/{j}/independent_operand_sum",
                np.add(data[f"F_mid{j + 1}"], data[f"F_write{j + 1}"], dtype=np.float32),
                data["F_trace"][j],
            )
        for arm in ("ZC", "ZM", "ZB"):
            for suffix in ("_final", "_scores"):
                record(checks, arm + "/saved_zero" + suffix, data[arm + suffix], data["F" + suffix])
    else:
        for suffix in ("_final", "_scores", "_trace"):
            record(
                checks, "saved_native_calibration" + suffix, data["T" + suffix], data["O" + suffix]
            )
    count = 0
    for name in witness:
        if not re.fullmatch(r"(F|C|M|B|T|O)_[0-9]+_cm", name):
            continue
        prefix = name[:-2]
        arm, layer = prefix.rstrip("_").split("_")
        layer = int(layer)
        cm, cw, fm, fw = [witness[prefix + n] for n in ("cm", "cw", "fm", "fw")]
        for n, value in (("mid", fm), ("write", fw)):
            record(checks, prefix + "false_" + n, value, false_data[f"F_{n}{layer}"])
        weight, bias, gamma = [
            maps[f"{layer}_{n}"].astype(float) for n in ("query", "bias", "norm")
        ]
        eps = float(maps["eps"])
        k, v, mask, cos, sin = [
            witness[prefix + n].astype(float) for n in ("k", "v", "mask", "cos", "sin")
        ]
        groups, scaling = int(witness[prefix + "groups"]), float(witness[prefix + "scaling"])
        kk, vv = np.repeat(k, groups, axis=0), np.repeat(v, groups, axis=0)
        heads, rows, dim = witness[prefix + "N_q"].shape
        selected = "N" if arm in ("F", "T", "O") else arm
        for corner in ("N", "C", "M", "B"):
            mid = cm if corner in ("N", "C") else fm
            write = cw if corner in ("N", "M") else fw
            x = np.add(mid, write, dtype=np.float32)
            record(
                checks,
                prefix + corner + "/operand_identity",
                x,
                witness[prefix + corner + "_input"],
            )
            x = x.astype(float)
            norm = gamma * x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
            record(
                checks,
                prefix + corner + "/CPU_norm",
                norm,
                witness[prefix + corner + "_normal"],
                BOUNDS["normalized"],
            )
            q = (norm @ weight.T + bias).reshape(rows, heads, dim).transpose(1, 0, 2)
            half = np.concatenate((-q[..., dim // 2 :], q[..., : dim // 2]), axis=-1)
            q = q * cos[None] + half * sin[None]
            actual_q = witness[prefix + corner + "_q"].astype(float)
            error = float(np.max(abs(q - actual_q)))
            raw.append(
                dict(
                    key=case["key"],
                    arm=arm,
                    layer=layer + 1,
                    corner=corner,
                    kind="query_projection_rotary",
                    error=error,
                    bound=BOUNDS["raw_diagnostic"],
                    passes=error <= BOUNDS["raw_diagnostic"],
                )
            )
            record(
                checks, prefix + corner + "/CPU_query_abs", q, actual_q, BOUNDS["projection_abs"]
            )
            record(
                checks,
                prefix + corner + "/CPU_query_relative",
                np.linalg.norm(q - actual_q) / max(np.linalg.norm(actual_q), 1e-30),
                0.0,
                BOUNDS["projection_relative"],
            )
            p = softmax(q @ kk.swapaxes(-1, -2) * scaling + mask)
            stored_p = witness[prefix + corner + "_prob"].astype(float)
            y = witness[prefix + corner + "_output"].astype(float)
            record(checks, prefix + corner + "/CPU_probability", p, stored_p, BOUNDS["composed"])
            record(checks, prefix + corner + "/CPU_output", p @ vv, y, BOUNDS["composed"])
            record(checks, prefix + corner + "/runtime_PV", stored_p @ vv, y, BOUNDS["composed"])
            record(
                checks,
                prefix + corner + "/probability_sum",
                stored_p.sum(-1),
                np.ones(stored_p.shape[:-1]),
                1e-6,
            )
            hidden = np.broadcast_to(mask < -1e20, p.shape)
            record(
                checks, prefix + corner + "/masked_mass", stored_p[hidden], np.zeros(np.sum(hidden))
            )
        record(
            checks,
            prefix + "B_reconstructs_F_query",
            witness[prefix + "B_q"],
            false_data[f"F_q{layer + 1}"],
        )
        for suffix in ("q", "prob", "output"):
            record(
                checks,
                prefix + "selected_" + suffix,
                witness[prefix + "selected_" + suffix],
                witness[prefix + selected + "_" + suffix],
            )
        count += 1
    for r in checks[first:]:
        r.setdefault("key", case["key"])
    return count


def native_sources(case, fm, nm, checks):
    for arm in ARMS:
        record(
            checks,
            arm + "/independent_native_source",
            float(fm["source_hashes"][arm] != nm["source_hashes"]["T"]),
            0.0,
        )
        checks[-1]["key"] = case["key"]


def local(case, witness, maps, checks):
    """Signed local effects at shared native parents; supplied-state retrospection."""
    out = []
    for name in witness:
        if not re.fullmatch(r"T_[0-9]+_cm", name):
            continue
        prefix = name[:-2]
        layer = int(prefix.split("_")[1])
        wo = maps[f"{layer}_output"].astype(float)

        def project(y):
            return y.transpose(1, 0, 2).reshape(y.shape[1], -1) @ wo.T

        for kind in ("query", "attention_write"):
            values = {
                c: witness[prefix + c + ("_q" if kind == "query" else "_output")].astype(float)
                for c in ("N", "C", "M", "B")
            }
            values = {
                c: y.transpose(1, 0, 2).reshape(y.shape[1], -1) if kind == "query" else project(y)
                for c, y in values.items()
            }
            target = values["N"] - values["B"]
            parts = dict(
                C=values["C"] - values["B"],
                M=values["M"] - values["B"],
                interaction=values["N"] - values["C"] - values["M"] + values["B"],
            )
            record(
                checks,
                case["key"] + prefix + kind + "/factorial_closure",
                sum(parts.values()),
                target,
                1e-9,
            )
            for a, value in parts.items():
                for row in range(len(target)):
                    s = statistics(value[row], target[row])
                    out.append(
                        dict(
                            key=case["key"],
                            layout=case["group"]["block"],
                            schema=case["group"]["schema"],
                            layer=layer + 1,
                            position=case["position"] + 1 + row,
                            is_verdict=row == len(target) - 1,
                            kind=kind,
                            arm=a,
                            target_eligible=s["norm"] >= 0.001,
                            **s,
                        )
                    )
    return out


def margin(data, arm):
    scores = data[arm + "_scores"].astype(float)
    return float(scores[1] - scores[0])


def scientific(case, f, t):
    rows, vectors = [], []
    delta = t["T_final"].astype(float) - f["F_final"].astype(float)
    query = t["T_final"].astype(float) - f["B_final"].astype(float)
    for arm in ARMS:
        native, effect = {}, {}
        for scope, sel in [("receiving", slice(1, None)), ("verdict", -1)]:
            native[scope] = statistics(
                f[arm + "_final"].astype(float)[sel] - f["F_final"].astype(float)[sel], delta[sel]
            )
            effect[scope] = statistics(
                f[arm + "_final"].astype(float)[sel] - f["B_final"].astype(float)[sel], query[sel]
            )
            for target, stats in [("native", native[scope]), ("query_effect", effect[scope])]:
                vectors.append(
                    dict(
                        key=case["key"],
                        layout=case["group"]["block"],
                        arm=arm,
                        scope=scope,
                        target=target,
                        **stats,
                    )
                )
        error = abs(margin(f, arm) - margin(t, "T"))
        material = abs(margin(t, "T") - margin(f, "B"))
        passed = (
            all(
                s["norm"] >= PRIMARY["norm"]
                and s["relative"] <= PRIMARY["relative"]
                and s["cosine"] >= PRIMARY["cosine"]
                for s in native.values()
            )
            and error <= PRIMARY["margin"]
        )
        effect_passed = (
            all(
                s["norm"] >= EFFECT["norm"]
                and s["relative"] <= EFFECT["relative"]
                and s["cosine"] >= EFFECT["cosine"]
                for s in effect.values()
            )
            and error <= EFFECT["margin"]
            and material >= EFFECT["material"]
        )
        rows.append(
            dict(
                key=case["key"],
                layout=case["group"]["block"],
                arm=arm,
                receiving_relative=native["receiving"]["relative"],
                verdict_relative=native["verdict"]["relative"],
                margin_error=error,
                passes=bool(passed),
                query_receiving_relative=effect["receiving"]["relative"],
                query_verdict_relative=effect["verdict"]["relative"],
                query_margin_magnitude=material,
                effect_passes=bool(effect_passed),
            )
        )
    for scope, sel in [("receiving", slice(1, None)), ("verdict", -1)]:
        interaction = (
            t["T_final"].astype(float)
            - f["C_final"].astype(float)
            - f["M_final"].astype(float)
            + f["B_final"].astype(float)
        )
        vectors.append(
            dict(
                key=case["key"],
                layout=case["group"]["block"],
                arm="interaction",
                scope=scope,
                target="query_effect",
                **statistics(interaction[sel], query[sel]),
            )
        )
    layers = []
    for j in range(len(f["F_trace"])):
        target = t["T_trace"][j].astype(float) - f["B_trace"][j].astype(float)
        for arm in ["C", "M", "interaction"]:
            pred = (
                f[arm + "_trace"][j].astype(float) - f["B_trace"][j].astype(float)
                if arm != "interaction"
                else t["T_trace"][j].astype(float)
                - f["C_trace"][j].astype(float)
                - f["M_trace"][j].astype(float)
                + f["B_trace"][j].astype(float)
            )
            layers.append(dict(key=case["key"], layer=j + 1, arm=arm, **statistics(pred, target)))
    return rows, vectors, layers


def populations(rows, field="passes"):
    out = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        layouts = {
            s: sum(r[field] for r in selected if r["layout"] == s)
            for s in sorted({r["layout"] for r in selected})
        }
        n = sum(layouts.values())
        out[arm] = dict(
            passes=n,
            total=len(selected),
            layouts=layouts,
            sufficient=n >= PRIMARY["population"]
            and min(layouts.values()) >= PRIMARY["per_layout"],
        )
    return out


def main(run, phase):
    dest = run / ("pre-audit.json" if phase == "forecast" else "audit.json")
    assert not dest.exists()
    design, status = read(run / "design.json"), read(run / "forecast-status.json")
    assert (
        design["experiment"] == 299 and status["ordinary_true"] == 0 and len(status["cases"]) == 32
    )
    assert (
        design["primary"] == PRIMARY and design["effect"] == EFFECT and design["bounds"] == BOUNDS
    )
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    assert (
        sha(run / "head.npz") == design["head_sha256"]
        and sha(run / "query-weights.npz") == design["query_weights_sha256"]
    )
    with np.load(run / "head.npz") as z:
        w = dict(z)
    with np.load(run / "query-weights.npz") as z:
        maps = dict(z)
    checks, raw, rows, vectors, layers, locals_ = [], [], [], [], [], []
    count = 0
    native = None
    if phase == "forecast":
        assert not list((run / "observations").glob("*"))
    else:
        native, seal, proof = (
            read(run / "native-status.json"),
            read(run / "forecast-seal.json"),
            read(run / "measurement-proof.json"),
        )
        assert (
            sha(run / "forecast-seal.json")
            == native["forecast_seal_sha256"]
            == proof["seal_sha256"]
        )
        assert (
            native["proof_sha256"] == sha(run / "measurement-proof.json")
            and proof["accepted_utc"] > seal["created_utc"]
        )
        assert proof["remote_head"] == proof["commit"] and native["ordinary_true"] == 32
    for i, (case, entry) in enumerate(zip(design["cases"], status["cases"], strict=True)):
        assert case["key"] == entry["key"]
        p = run / "forecasts" / (case["key"] + ".npz")
        fm = read(p.with_suffix(".json"))
        assert (
            sha(p) == entry["arrays_sha256"]
            and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
            and fm["ordinary_true"] == 0
        )
        with np.load(p) as z:
            f = dict(z)
        wit = {}
        if entry["witness_sha256"]:
            wp = run / "witnesses-forecast" / p.name
            assert sha(wp) == entry["witness_sha256"]
            with np.load(wp) as z:
                wit = dict(z)
        count += audit_bundle(case, f, wit, fm, w, "forecast", checks, raw, f, maps)
        if phase == "native":
            ne = native["cases"][i]
            p = run / "observations" / p.name
            nm = read(p.with_suffix(".json"))
            assert (
                ne["key"] == case["key"]
                and sha(p) == ne["arrays_sha256"]
                and sha(p.with_suffix(".json")) == ne["metadata_sha256"]
            )
            assert (
                nm["native_started_utc"] > proof["accepted_utc"]
                and nm["forecast_arrays_sha256"] == entry["arrays_sha256"]
            )
            with np.load(p) as z:
                t = dict(z)
            nw = {}
            if ne["witness_sha256"]:
                wp = run / "witnesses-native" / p.name
                assert sha(wp) == ne["witness_sha256"]
                with np.load(wp) as z:
                    nw = dict(z)
            count += audit_bundle(case, t, nw, nm, w, "native", checks, raw, f, maps)
            native_sources(case, fm, nm, checks)
            rr, vv, ll = scientific(case, f, t)
            rows.extend(rr)
            vectors.extend(vv)
            layers.extend(ll)
            locals_.extend(local(case, nw, maps, checks))
    assert count == (96 if phase == "forecast" else 144), count
    failed = [r for r in checks if not r["passes"]]
    stem = "pre-" if phase == "forecast" else ""
    table(run / (stem + "checks.csv"), checks)
    table(run / (stem + "raw-queries.csv"), raw)
    report = dict(
        experiment=299,
        status="failed" if failed else "passed",
        design_sha256=sha(run / "design.json"),
        forecast_status_sha256=sha(run / "forecast-status.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_query_checks=len(raw),
        raw_query_failed=sum(not r["passes"] for r in raw),
        witness_kernels=count,
        scientific_outcomes_withheld=bool(failed),
        general_question_B_solved=False,
    )
    if phase == "native" and not failed:
        table(run / "native.csv", rows)
        save_json(run / "vector-records.json", dict(statistics=vectors))
        save_json(run / "layer-records.json", dict(statistics=layers))
        save_json(run / "local-records.json", dict(statistics=locals_))
        pop = populations(rows)
        report.update(
            populations=pop,
            query_effect_populations=populations(rows, "effect_passes"),
            H1_carry=bool(
                pop["C"]["sufficient"] and pop["M"]["passes"] <= 8 and pop["B"]["passes"] <= 8
            ),
            H2_last_MLP=bool(
                pop["M"]["sufficient"] and pop["C"]["passes"] <= 8 and pop["B"]["passes"] <= 8
            ),
            native_status_sha256=sha(run / "native-status.json"),
        )
    report["csv_sha256"] = {
        p.name: sha(p) for p in run.glob("*.csv") if phase == "native" or p.name.startswith("pre-")
    }
    save_json(dest, report)
    print(report, flush=True)
    assert not failed, "Required numerical checks failed; scientific outcomes withheld"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=["forecast", "native"], required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.phase)
