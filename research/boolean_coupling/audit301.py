"""Independent source-prefix routing, finite attention and sealed alias contrasts."""

import argparse
import re
from pathlib import Path

import numpy as np

from .alias301 import BOUNDS, FORECAST_ARMS, NATIVE, NATIVE_ARMS, PRIMARY, ROOT, definitions, record
from .audit229 import sha, softmax
from .audit272 import read, table
from .audit296 import head
from .cache272 import save_json
from .subspace293 import statistics


def bundle(case, data, witness, meta, head_weights, maps, phase, checks, raw, false):
    first = len(checks)
    checks.extend(dict(key=case["key"], **r) for r in meta["checks"])
    arms = FORECAST_ARMS if phase == "forecast" else NATIVE_ARMS
    for arm in (*arms, *(("P_A", "P_B") if phase == "forecast" else ())):
        record(
            checks,
            arm + "/finite_head",
            head(head_weights, data[arm + "_final"][-1]),
            data[arm + "_scores"],
            BOUNDS["head"],
        )
        if arm + "_trace" in data:
            record(checks, arm + "/trace_endpoint", data[arm + "_trace"][-1], data[arm + "_final"])
    layers = len(false["A_F_trace"])
    if phase == "forecast":
        for alias, hybrid in (("A", "BA"), ("B", "AB")):
            for suffix in ("final", "scores"):
                record(
                    checks,
                    alias + "/saved_zero_" + suffix,
                    data[alias + "Z_F_" + suffix],
                    data[alias + "_F_" + suffix],
                )
            expected = (
                data[alias + "_F_final"].astype(float)
                + data[hybrid + "_T_final"].astype(float)
                - data[hybrid + "_F_final"].astype(float)
            ).astype(np.float32)
            record(
                checks,
                "P_" + alias + "/independent_composition",
                data["P_" + alias + "_final"],
                expected,
            )
            record(
                checks,
                hybrid + "/independent_false_source",
                float(meta["source_hashes"][hybrid + "_F"] != meta["source_hashes"][alias + "_F"]),
                0.0,
            )
            for j in range(layers):
                record(
                    checks,
                    hybrid + f"/{j}/false_emitter",
                    data[f"{hybrid}_F_emit{j}"],
                    data[f"{alias}_F_emit{j}"],
                )
        for j in range(layers):
            for kind in ("k", "v"):
                a, b, r = [data[f"bank_{x}_{kind}{j}"].astype(float) for x in ("A", "B", "R")]
                target = np.linalg.norm(b - a, axis=-1)
                actual = np.linalg.norm(r - a, axis=-1)
                record(
                    checks,
                    f"R/{j}/{kind}_matched_norm",
                    (actual - target) / np.maximum(target, BOUNDS["random_norm_floor"]),
                    np.zeros_like(target),
                    BOUNDS["random_norm_relative"],
                )
    else:
        for alias in ("A", "B"):
            for suffix in ("final", "scores", "trace"):
                record(
                    checks,
                    alias + "/independent_native_replay_" + suffix,
                    data[alias + "O_T_" + suffix],
                    data[alias + "_T_" + suffix],
                )
    count = 0
    for name in witness:
        if not re.fullmatch(
            r"(A_F|B_F|AB_F|AB_T|BA_F|BA_T|RA_F|RA_T|A_T|B_T|AO_T|BO_T)_[0-9]+_query", name
        ):
            continue
        prefix = name[: -len("query")]
        arm, layer = prefix.rstrip("_").rsplit("_", 1)
        j = int(layer)
        receiver, donor, _ = definitions(arm)
        q, kk, vv, pk, pv, mask, actual_raw, p, y = [
            witness[prefix + n].astype(float)
            for n in (
                "query",
                "used_k",
                "used_v",
                "recipient_k",
                "recipient_v",
                "mask",
                "raw",
                "prob",
                "head_output",
            )
        ]
        expected_k, expected_v = pk.copy(), pv.copy()
        if donor is not None:
            for kind, value in (("k", expected_k), ("v", expected_v)):
                value[:, case["alias_start"] : case["position"]] = false[f"bank_{donor}_{kind}{j}"]
        record(checks, prefix + "independent_key_route", kk, expected_k)
        record(checks, prefix + "independent_value_route", vv, expected_v)
        for kind, value in (("k", pk), ("v", pv)):
            record(
                checks,
                prefix + "recipient_bank_" + kind,
                value[:, case["alias_start"] : case["position"]],
                false[f"bank_{receiver}_{kind}{j}"],
            )
        groups, scale = int(witness[prefix + "groups"]), float(witness[prefix + "scaling"])
        k, v = np.repeat(kk, groups, axis=0), np.repeat(vv, groups, axis=0)
        calc = q @ k.swapaxes(-1, -2) * scale
        e = float(np.max(abs(calc - actual_raw)))
        raw.append(
            dict(
                key=case["key"],
                arm=arm,
                layer=j + 1,
                kind="scaled_source_logits",
                error=e,
                bound=BOUNDS["raw"],
                passes=e <= BOUNDS["raw"],
            )
        )
        cp = softmax(calc + mask)
        record(checks, prefix + "CPU_probability", cp, p, BOUNDS["composed"])
        record(checks, prefix + "CPU_head_output", cp @ v, y, BOUNDS["composed"])
        record(checks, prefix + "runtime_PV", p @ v, y, BOUNDS["composed"])
        record(checks, prefix + "probability_sum", p.sum(-1), np.ones(p.shape[:-1]), 1e-6)
        write = y.transpose(1, 0, 2).reshape(1, -1) @ maps[str(j)].astype(float).T
        expected = witness[prefix + "source_write"].astype(float)
        err = float(np.max(abs(write[0] - expected)))
        raw.append(
            dict(
                key=case["key"],
                arm=arm,
                layer=j + 1,
                kind="output_projection",
                error=err,
                bound=BOUNDS["raw"],
                passes=err <= BOUNDS["raw"],
            )
        )
        record(
            checks, prefix + "CPU_source_write_abs", write[0], expected, BOUNDS["projection_abs"]
        )
        record(
            checks,
            prefix + "CPU_source_write_relative",
            np.linalg.norm(write[0] - expected) / max(np.linalg.norm(expected), 1e-30),
            0.0,
            BOUNDS["projection_relative"],
        )
        count += 1
    for r in checks[first:]:
        r.setdefault("key", case["key"])
    return count


def native_sources(case, f, n, fm, nm, checks):
    for alias, hybrid in (("A", "BA"), ("B", "AB")):
        record(
            checks,
            hybrid + "/independent_true_source",
            float(fm["source_hashes"][hybrid + "_T"] != nm["source_hashes"][alias + "_T"]),
            0.0,
        )
        for j in range(len(f["A_F_trace"])):
            record(
                checks,
                hybrid + f"/{j}/true_emitter",
                f[f"{hybrid}_T_emit{j}"],
                n[f"{alias}_T_emit{j}"],
            )
    for r in checks:
        r.setdefault("key", case["key"])


def margin(data, arm):
    s = data[arm + "_scores"].astype(float)
    return float(s[1] - s[0])


def scientific(case, f, n):
    responses = {
        "AA": n["A_T_final"].astype(float) - f["A_F_final"].astype(float),
        "BB": n["B_T_final"].astype(float) - f["B_F_final"].astype(float),
        "AB": f["AB_T_final"].astype(float) - f["AB_F_final"].astype(float),
        "BA": f["BA_T_final"].astype(float) - f["BA_F_final"].astype(float),
        "RA": f["RA_T_final"].astype(float) - f["RA_F_final"].astype(float),
    }
    coupling = {
        "AA": margin(n, "A_T") - margin(f, "A_F"),
        "BB": margin(n, "B_T") - margin(f, "B_F"),
        **{h: margin(f, h + "_T") - margin(f, h + "_F") for h in ("AB", "BA", "RA")},
    }
    target = responses["BB"] - responses["AA"]
    effect = coupling["BB"] - coupling["AA"]
    pieces = dict(
        source_A=responses["AB"] - responses["AA"],
        source_B=responses["BB"] - responses["BA"],
        remaining_A=responses["BA"] - responses["AA"],
        remaining_B=responses["BB"] - responses["AB"],
        random_A=responses["RA"] - responses["AA"],
        interaction=responses["BB"] - responses["BA"] - responses["AB"] + responses["AA"],
    )
    margins = dict(
        source_A=coupling["AB"] - coupling["AA"],
        source_B=coupling["BB"] - coupling["BA"],
        remaining_A=coupling["BA"] - coupling["AA"],
        remaining_B=coupling["BB"] - coupling["AB"],
        random_A=coupling["RA"] - coupling["AA"],
        interaction=coupling["BB"] - coupling["BA"] - coupling["AB"] + coupling["AA"],
    )
    vectors = []
    metrics = {}
    for arm, pred in pieces.items():
        metrics[arm] = {}
        for scope, sel in (("receiving", slice(1, None)), ("verdict", -1)):
            s = statistics(pred[sel], target[sel])
            metrics[arm][scope] = s
            vectors.append(dict(key=case["key"], arm=arm, scope=scope, **s))
    rows = []
    bound = min(PRIMARY["margin_abs"], PRIMARY["margin_fraction"] * abs(effect))
    for account, arms in (
        ("source", ("source_A", "source_B")),
        ("remaining", ("remaining_A", "remaining_B")),
        ("random", ("random_A",)),
    ):
        errs = [abs(margins[a] - effect) for a in arms]
        passed = (
            abs(effect) >= PRIMARY["material"]
            and max(errs) <= bound
            and all(
                s["norm"] >= PRIMARY["norm"]
                and s["relative"] <= PRIMARY["relative"]
                and s["cosine"] >= PRIMARY["cosine"]
                for a in arms
                for s in metrics[a].values()
            )
        )
        rows.append(
            dict(
                key=case["key"],
                layout=case["group"]["block"],
                mention_order=case["group"]["mention_order"],
                account=account,
                material=abs(effect) >= PRIMARY["material"],
                native_alias_margin_contrast=effect,
                margin_bound=bound,
                margin_error=max(errs),
                receiving_relative=max(metrics[a]["receiving"]["relative"] for a in arms),
                verdict_relative=max(metrics[a]["verdict"]["relative"] for a in arms),
                passes=bool(passed),
            )
        )
    native = []
    for alias in ("A", "B"):
        stats = {
            scope: statistics(
                f["P_" + alias + "_final"].astype(float)[sel]
                - f[alias + "_F_final"].astype(float)[sel],
                n[alias + "_T_final"].astype(float)[sel] - f[alias + "_F_final"].astype(float)[sel],
            )
            for scope, sel in (("receiving", slice(1, None)), ("verdict", -1))
        }
        for scope, s in stats.items():
            vectors.append(dict(key=case["key"], arm="P_" + alias, scope=scope, **s))
        err = abs(margin(f, "P_" + alias) - margin(n, alias + "_T"))
        passed = err <= NATIVE["margin"] and all(
            s["norm"] >= NATIVE["norm"]
            and s["relative"] <= NATIVE["relative"]
            and s["cosine"] >= NATIVE["cosine"]
            for s in stats.values()
        )
        native.append(
            dict(
                key=case["key"],
                layout=case["group"]["block"],
                mention_order=case["group"]["mention_order"],
                alias=alias,
                receiving_relative=stats["receiving"]["relative"],
                verdict_relative=stats["verdict"]["relative"],
                margin_error=err,
                passes=bool(passed),
            )
        )
    coupling_rows = [
        dict(
            key=case["key"],
            layout=case["group"]["block"],
            mention_order=case["group"]["mention_order"],
            corner=c,
            value=v,
        )
        for c, v in coupling.items()
    ]
    return rows, vectors, native, coupling_rows


def populations(rows):
    result = {}
    for account in ("source", "remaining", "random"):
        rr = [r for r in rows if r["account"] == account]
        layouts = {
            s: sum(r["passes"] for r in rr if r["layout"] == s)
            for s in sorted({r["layout"] for r in rr})
        }
        order = {str(s): sum(r["passes"] for r in rr if r["mention_order"] == s) for s in (0, 1)}
        total = sum(layouts.values())
        result[account] = dict(
            passes=total,
            total=len(rr),
            layouts=layouts,
            mention_orders=order,
            sufficient=total >= 24 and min(layouts.values()) >= 6 and min(order.values()) >= 12,
        )
    return result


def main(run, phase):
    dest = run / ("pre-audit.json" if phase == "forecast" else "audit.json")
    assert not dest.exists()
    d, status = read(run / "design.json"), read(run / "forecast-status.json")
    assert (
        d["experiment"] == 301
        and d["primary"] == PRIMARY
        and d["native"] == NATIVE
        and d["bounds"] == BOUNDS
        and status["ordinary_true"] == 0
    )
    for name, h in d["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == h, name
    assert (
        sha(run / "head.npz") == d["head_sha256"]
        and sha(run / "output-maps.npz") == d["output_maps_sha256"]
    )
    with np.load(run / "head.npz") as z:
        w = dict(z)
    with np.load(run / "output-maps.npz") as z:
        maps = dict(z)
    checks, raw, rows, vectors, native_rows, coupling_rows = [], [], [], [], [], []
    count = 0
    native = None
    if phase == "forecast":
        assert not list((run / "observations").glob("*"))
    else:
        native = read(run / "native-status.json")
        seal = read(run / "forecast-seal.json")
        proof = read(run / "measurement-proof.json")
        assert (
            native["ordinary_true"] == 64
            and native["forecast_seal_sha256"]
            == sha(run / "forecast-seal.json")
            == proof["seal_sha256"]
        )
        assert (
            native["proof_sha256"] == sha(run / "measurement-proof.json")
            and proof["accepted_utc"] > seal["created_utc"]
            and proof["remote_head"] == proof["commit"]
        )
    for i, (case, e) in enumerate(zip(d["cases"], status["cases"], strict=True)):
        assert case["key"] == e["key"]
        p = run / "forecasts" / (case["key"] + ".npz")
        fm = read(p.with_suffix(".json"))
        assert (
            sha(p) == e["arrays_sha256"]
            and sha(p.with_suffix(".json")) == e["metadata_sha256"]
            and fm["ordinary_true"] == 0
        )
        with np.load(p) as z:
            f = dict(z)
        wit = {}
        if e["witness_sha256"]:
            wp = run / "witnesses-forecast" / p.name
            assert sha(wp) == e["witness_sha256"]
            with np.load(wp) as z:
                wit = dict(z)
        count += bundle(case, f, wit, fm, w, maps, "forecast", checks, raw, f)
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
                nm["ordinary_true"] == 2
                and nm["native_started_utc"] > proof["accepted_utc"]
                and nm["forecast_arrays_sha256"] == e["arrays_sha256"]
            )
            with np.load(p) as z:
                n = dict(z)
            nw = {}
            if ne["witness_sha256"]:
                wp = run / "witnesses-native" / p.name
                assert sha(wp) == ne["witness_sha256"]
                with np.load(wp) as z:
                    nw = dict(z)
            count += bundle(case, n, nw, nm, w, maps, "native", checks, raw, f)
            native_sources(case, f, n, fm, nm, checks)
            rr, vv, nn, cc = scientific(case, f, n)
            rows.extend(rr)
            vectors.extend(vv)
            native_rows.extend(nn)
            coupling_rows.extend(cc)
    assert count == (192 if phase == "forecast" else 288), count
    failed = [r for r in checks if not r["passes"]]
    stem = "pre-" if phase == "forecast" else ""
    table(run / (stem + "checks.csv"), checks)
    table(run / (stem + "raw-coordinates.csv"), raw)
    report = dict(
        experiment=301,
        status="failed" if failed else "passed",
        design_sha256=sha(run / "design.json"),
        forecast_status_sha256=sha(run / "forecast-status.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_checks=len(raw),
        raw_failed=sum(not r["passes"] for r in raw),
        witness_kernels=count,
        scientific_outcomes_withheld=bool(failed),
        general_question_B_solved=False,
    )
    if phase == "native" and not failed:
        table(run / "effects.csv", rows)
        table(run / "native-predictions.csv", native_rows)
        table(run / "couplings.csv", coupling_rows)
        save_json(run / "vector-records.json", dict(statistics=vectors))
        pop = populations(rows)
        report.update(
            populations=pop,
            H1_source=pop["source"]["sufficient"]
            and pop["remaining"]["passes"] <= 8
            and pop["random"]["passes"] <= 8,
            H2_remaining=pop["remaining"]["sufficient"]
            and pop["source"]["passes"] <= 8
            and pop["random"]["passes"] <= 8,
            material_alias_cases=sum(r["material"] for r in rows if r["account"] == "source"),
            native_individual_passes=sum(r["passes"] for r in native_rows),
            native_pair_passes=sum(
                all(r["passes"] for r in native_rows if r["key"] == c["key"]) for c in d["cases"]
            ),
            native_status_sha256=sha(run / "native-status.json"),
        )
    report["csv_sha256"] = {
        p.name: sha(p) for p in run.glob("*.csv") if phase == "native" or p.name.startswith("pre-")
    }
    save_json(dest, report)
    print(report)
    assert not failed, "Required checks failed; scientific outcomes withheld"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=["forecast", "native"], required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.phase)
