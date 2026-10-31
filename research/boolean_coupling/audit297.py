"""Independent hierarchical probability maps, finite kernels and signed effects."""

import argparse
import re
from pathlib import Path

import numpy as np

from .audit229 import sha, softmax
from .audit272 import read, table
from .audit296 import head
from .cache272 import save_json
from .routes297 import ARMS, BOUNDS, EFFECT, PRIMARY, ROOT, record
from .subspace293 import statistics


def compose(arm, current, false, bc, bf, source, equal):
    """Separate NumPy implementation from full and conditional probabilities."""
    if arm in ("N", "B"):
        return (current if arm == "N" else false).copy()
    assert arm in ("S", "C")
    alpha = (current if arm == "S" else false)[..., source]
    beta = bf if arm == "S" else bc
    out = beta * (1 - alpha[..., None])
    out[..., source] = alpha
    return np.where(equal, current, out)


def audit_bundle(case, data, witness, meta, w, phase, checks, raw, false_data):
    first = len(checks)
    checks.extend(dict(key=case["key"], **r) for r in meta["checks"])
    arms = ("F", *ARMS, "ZS", "ZC", "ZB") if phase == "forecast" else ("T", "O")
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
        for arm in ["ZS", "ZC", "ZB"]:
            for suffix in ["_final", "_scores"]:
                record(checks, arm + "/saved_zero" + suffix, data[arm + suffix], data["F" + suffix])
    else:
        for suffix in ["_final", "_scores", "_trace"]:
            record(
                checks, "saved_native_calibration" + suffix, data["T" + suffix], data["O" + suffix]
            )
    count = 0
    for name in witness:
        if not re.fullmatch(r"(F|S|C|B|T|O)_[0-9]+_current_q", name):
            continue
        prefix = name[: -len("current_q")]
        arm, layer = prefix.rstrip("_").split("_")
        layer = int(layer)
        q, f, k, v, mask, cr, fr = [
            witness[prefix + n].astype(float)
            for n in ["current_q", "false_q", "k", "v", "mask", "current_raw", "false_raw"]
        ]
        record(checks, prefix + "saved_false_query", f, false_data[f"F_q{layer + 1}"])
        groups, scaling = int(witness[prefix + "groups"]), float(witness[prefix + "scaling"])
        kk, vv = np.repeat(k, groups, axis=0), np.repeat(v, groups, axis=0)
        cc, ff = q @ kk.swapaxes(-1, -2) * scaling, f @ kk.swapaxes(-1, -2) * scaling
        for kind, calc, actual in [("current", cc, cr), ("false", ff, fr)]:
            err = float(np.max(abs(calc - actual)))
            raw.append(
                dict(
                    key=case["key"],
                    arm=arm,
                    layer=layer + 1,
                    kind=kind,
                    error=err,
                    bound=BOUNDS["raw_diagnostic"],
                    passes=bool(np.isfinite(err) and err <= BOUNDS["raw_diagnostic"]),
                )
            )
        selected = "N" if arm in ["F", "T", "O"] else arm
        pos = case["position"]
        equal = np.all(q == f, axis=-1, keepdims=True)
        pc, pf = softmax(cc + mask), softmax(ff + mask)
        lc, lf = (cc + mask).copy(), (ff + mask).copy()
        lc[..., pos], lf[..., pos] = -np.inf, -np.inf
        bc, bf = softmax(lc), softmax(lf)
        stored = {
            n: witness[prefix + n].astype(float)
            for n in ["current_prob", "false_prob", "current_context", "false_context"]
        }
        for n, expected in [
            ("current_prob", pc),
            ("false_prob", pf),
            ("current_context", bc),
            ("false_context", bf),
        ]:
            record(checks, prefix + n + "/CPU", stored[n], expected, BOUNDS["composed"])
        for corner in ["B", "S", "C", "N"]:
            p, y = (
                witness[prefix + corner + "_prob"].astype(float),
                witness[prefix + corner + "_output"].astype(float),
            )
            cp = compose(corner, pc, pf, bc, bf, pos, equal)
            runtime = compose(
                corner,
                stored["current_prob"],
                stored["false_prob"],
                stored["current_context"],
                stored["false_context"],
                pos,
                equal,
            )
            record(
                checks, prefix + corner + "/independent_probability_map", runtime, p, BOUNDS["map"]
            )
            for label, a, b in [
                ("CPU_probability", cp, p),
                ("CPU_output", cp @ vv, y),
                ("runtime_PV", p @ vv, y),
            ]:
                record(checks, prefix + corner + "/" + label, a, b, BOUNDS["composed"])
            record(
                checks,
                prefix + corner + "/probability_sum",
                p.sum(-1),
                np.ones(p.shape[:-1]),
                BOUNDS["probability_sum"],
            )
            hidden = np.broadcast_to(mask < -1e20, p.shape)
            record(checks, prefix + corner + "/masked_mass", p[hidden], np.zeros(np.sum(hidden)))
        record(
            checks,
            prefix + "selected_prob",
            witness[prefix + "selected_prob"],
            witness[prefix + selected + "_prob"],
        )
        record(
            checks,
            prefix + "selected_output",
            witness[prefix + "selected_output"],
            witness[prefix + selected + "_output"],
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
            - f["S_final"].astype(float)
            - f["C_final"].astype(float)
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
        for arm in ["S", "C", "interaction"]:
            pred = (
                f[arm + "_trace"][j].astype(float) - f["B_trace"][j].astype(float)
                if arm != "interaction"
                else t["T_trace"][j].astype(float)
                - f["S_trace"][j].astype(float)
                - f["C_trace"][j].astype(float)
                + f["B_trace"][j].astype(float)
            )
            layers.append(dict(key=case["key"], layer=j + 1, arm=arm, **statistics(pred, target)))
    return rows, vectors, layers


def local(case, witness, maps, checks):
    """Native T parents only; CPU64 linear output map, no learned-feature claim."""
    out = []
    for name in witness:
        if not re.fullmatch(r"T_[0-9]+_current_q", name):
            continue
        prefix = name[: -len("current_q")]
        layer = int(prefix.split("_")[1])
        wo = maps[str(layer)].astype(float)

        def project(y):
            return y.transpose(1, 0, 2).reshape(y.shape[1], -1) @ wo.T

        y = {a: witness[prefix + a + "_output"].astype(float) for a in ["B", "S", "C", "N"]}
        target = project(y["N"] - y["B"])
        parts = {
            "S": project(y["S"] - y["B"]),
            "C": project(y["C"] - y["B"]),
            "interaction": project(y["N"] - y["S"] - y["C"] + y["B"]),
        }
        record(
            checks,
            case["key"] + prefix + "factorial_vector_closure",
            sum(parts.values()),
            target,
            1e-10,
        )
        groups = int(witness[prefix + "groups"])
        v = np.repeat(witness[prefix + "v"].astype(float), groups, axis=0)
        dp = witness[prefix + "N_prob"].astype(float) - witness[prefix + "B_prob"].astype(float)
        s = case["position"]
        ac = witness[prefix + "current_prob"].astype(float)[..., s : s + 1]
        af = witness[prefix + "false_prob"].astype(float)[..., s : s + 1]
        context_change = (
            witness[prefix + "current_context"].astype(float)
            - witness[prefix + "false_context"].astype(float)
        ) @ v
        bilinear = project(-(ac - af) * context_change)
        record(
            checks,
            case["key"] + prefix + "bilinear_interaction_bridge",
            parts["interaction"],
            bilinear,
            BOUNDS["composed"],
        )
        pieces = {}
        for label, sl in [
            ("prefix_values", slice(None, s)),
            ("source_value", slice(s, s + 1)),
            ("receiver_values", slice(s + 1, None)),
        ]:
            pieces[label] = project(dp[..., sl] @ v[:, sl])
        record(
            checks,
            case["key"] + prefix + "projected_PV_bridge",
            sum(pieces.values()),
            target,
            BOUNDS["composed"],
        )
        for arm, pred in {**parts, **pieces}.items():
            for row in range(len(target)):
                stats = statistics(pred[row], target[row])
                out.append(
                    dict(
                        key=case["key"],
                        layout=case["group"]["block"],
                        layer=layer + 1,
                        position=s + row + 1,
                        is_verdict=row == len(target) - 1,
                        arm=arm,
                        target_eligible=stats["norm"] >= 0.001,
                        **stats,
                    )
                )
    return out


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
        design["experiment"] == 297 and status["ordinary_true"] == 0 and len(status["cases"]) == 32
    )
    assert (
        design["primary"] == PRIMARY and design["effect"] == EFFECT and design["bounds"] == BOUNDS
    )
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    assert (
        sha(run / "head.npz") == design["head_sha256"]
        and sha(run / "output-maps.npz") == design["output_maps_sha256"]
    )
    with np.load(run / "head.npz") as z:
        w = dict(z)
    with np.load(run / "output-maps.npz") as z:
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
        count += audit_bundle(case, f, wit, fm, w, "forecast", checks, raw, f)
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
            count += audit_bundle(case, t, nw, nm, w, "native", checks, raw, f)
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
    table(run / (stem + "raw-logits.csv"), raw)
    report = dict(
        experiment=297,
        status="failed" if failed else "passed",
        design_sha256=sha(run / "design.json"),
        forecast_status_sha256=sha(run / "forecast-status.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_logit_checks=len(raw),
        raw_logit_failed=sum(not r["passes"] for r in raw),
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
            H1_source=bool(
                pop["S"]["sufficient"] and pop["C"]["passes"] <= 8 and pop["B"]["passes"] <= 8
            ),
            H2_other=bool(
                pop["C"]["sufficient"] and pop["S"]["passes"] <= 8 and pop["B"]["passes"] <= 8
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
