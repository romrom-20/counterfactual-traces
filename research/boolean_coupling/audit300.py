"""Independent fixed-parent attention kernels, signed writes and finite effects."""

import argparse
from pathlib import Path

import numpy as np

from .audit229 import sha, softmax
from .audit272 import read, table
from .audit296 import head
from .cache272 import save_json
from .interaction300 import ARMS, BOUNDS, CRITERIA, ROOT, record
from .subspace293 import statistics


def bundle(case, data, parent, reference, meta, weights, checks, raw):
    first = len(checks)
    checks.extend(meta["checks"])
    j = int(meta["block"]) - 1
    prefix = f"T_{j}_"
    q = {c: parent[prefix + c + "_q"] for c in ("N", "C", "M", "B")}
    qa = np.subtract(np.add(q["C"], q["M"], dtype=np.float32), q["B"], dtype=np.float32)
    record(checks, "A/independent_query_sum", qa, data["A_query"])
    q["A"] = qa
    groups = int(parent[prefix + "groups"])
    kk = np.repeat(parent[prefix + "k"].astype(float), groups, axis=0)
    vv = np.repeat(parent[prefix + "v"].astype(float), groups, axis=0)
    mask = parent[prefix + "mask"].astype(float)
    scale = float(parent[prefix + "scaling"])
    cpu = {}
    for corner in ("N", "C", "M", "B", "A"):
        p = softmax(q[corner].astype(float) @ kk.swapaxes(-1, -2) * scale + mask)
        y = p @ vv
        sp = data["A_prob"] if corner == "A" else parent[prefix + corner + "_prob"]
        sy = data["A_head"] if corner == "A" else parent[prefix + corner + "_output"]
        record(checks, corner + "/CPU_probability", p, sp, BOUNDS["composed"])
        record(checks, corner + "/CPU_head_output", y, sy, BOUNDS["composed"])
        rows = y.transpose(1, 0, 2).reshape(y.shape[1], -1)
        cpu[corner] = rows @ weights["output"].astype(float).T
        actual = data["write_" + corner].astype(float)
        err = float(np.max(abs(cpu[corner] - actual)))
        raw.append(
            dict(
                key=case["key"],
                corner=corner,
                error=err,
                bound=BOUNDS["raw_diagnostic"],
                passes=err <= BOUNDS["raw_diagnostic"],
            )
        )
        record(checks, corner + "/CPU_write_abs", cpu[corner], actual, BOUNDS["projection_abs"])
        record(
            checks,
            corner + "/CPU_write_relative",
            np.linalg.norm(cpu[corner] - actual) / max(np.linalg.norm(actual), 1e-30),
            0.0,
            BOUNDS["projection_relative"],
        )
    cpu["L"] = cpu["C"] + cpu["M"] - cpu["B"]
    cpu["R"] = cpu["L"] + cpu["N"] - cpu["A"]
    target = data["write_N"].astype(float) - data["write_L"].astype(float)
    denom = max(np.linalg.norm(target), CRITERIA["norm"])
    for arm in ("N", "A", "R"):
        calc = cpu[arm] - cpu["L"]
        actual = data["write_" + arm].astype(float) - data["write_L"].astype(float)
        record(
            checks,
            arm + "/CPU_scaled_write_contrast",
            np.linalg.norm(calc - actual) / denom,
            0.0,
            BOUNDS["contrast_relative"],
        )
    for label, actual, expected in (
        (
            "L/sum",
            data["write_L"],
            np.subtract(
                np.add(data["write_C"], data["write_M"], dtype=np.float32),
                data["write_B"],
                dtype=np.float32,
            ),
        ),
        (
            "R/restoration",
            data["write_R"],
            np.add(
                data["write_L"],
                np.subtract(data["write_N"], data["write_A"], dtype=np.float32),
                dtype=np.float32,
            ),
        ),
        (
            "J/installation",
            data["write_J"],
            np.add(data["write_L"], data["random_delta"], dtype=np.float32),
        ),
    ):
        record(checks, label, actual, expected)
    rnd = data["write_J"].astype(float) - data["write_L"].astype(float)
    nr, nt = np.linalg.norm(rnd, axis=-1), np.linalg.norm(target, axis=-1)
    record(
        checks,
        "J/installed_norm_match",
        (nr - nt) / np.maximum(nt, CRITERIA["norm"]),
        np.zeros_like(nr),
        BOUNDS["random_norm_relative"],
    )
    for suffix in ("final", "trace", "scores"):
        record(
            checks,
            "T0/independent_native_" + suffix,
            data["T0_" + suffix],
            reference["T_" + suffix],
        )
        record(checks, "O/independent_replay_" + suffix, data["O_" + suffix], data["T0_" + suffix])
    for arm in ("T0", "L", *ARMS):
        record(
            checks,
            arm + "/finite_head",
            head(weights, data[arm + "_final"][-1]),
            data[arm + "_scores"],
            BOUNDS["head"],
        )
    for r in checks[first:]:
        r.setdefault("key", case["key"])


def scientific(case, data):
    rows, vectors, local = [], [], []
    target = data["T0_final"].astype(float) - data["L_final"].astype(float)

    def margin(a):
        s = data[a + "_scores"].astype(float)
        return float(s[1] - s[0])

    effect = margin("T0") - margin("L")
    bound = min(CRITERIA["margin_abs"], CRITERIA["margin_fraction"] * abs(effect))
    for arm in ARMS:
        stats = {}
        for scope, sel in (("receiving", slice(1, None)), ("verdict", -1)):
            st = statistics(
                data[arm + "_final"].astype(float)[sel] - data["L_final"].astype(float)[sel],
                target[sel],
            )
            stats[scope] = st
            vectors.append(dict(key=case["key"], arm=arm, scope=scope, **st))
        err = abs(margin(arm) - margin("T0"))
        passed = (
            abs(effect) >= CRITERIA["material"]
            and err <= bound
            and all(
                s["norm"] >= CRITERIA["norm"]
                and s["relative"] <= CRITERIA["relative"]
                and s["cosine"] >= CRITERIA["cosine"]
                for s in stats.values()
            )
        )
        rows.append(
            dict(
                key=case["key"],
                layout=case["group"]["block"],
                schema=case["group"]["schema"],
                arm=arm,
                material=abs(effect) >= CRITERIA["material"],
                native_margin_effect=effect,
                arm_margin_effect=margin(arm) - margin("L"),
                margin_error=err,
                margin_bound=bound,
                receiving_relative=stats["receiving"]["relative"],
                verdict_relative=stats["verdict"]["relative"],
                passes=bool(passed),
            )
        )
    for scope, sel in (("receiving", slice(1, None)), ("verdict", -1)):
        interaction = (
            data["T0_final"].astype(float)
            - data["A_final"].astype(float)
            - data["R_final"].astype(float)
            + data["L_final"].astype(float)
        )
        vectors.append(
            dict(
                key=case["key"],
                arm="suffix_interaction",
                scope=scope,
                **statistics(interaction[sel], target[sel]),
            )
        )
    wt = data["write_N"].astype(float) - data["write_L"].astype(float)
    for arm in ("A", "R", "J"):
        pred = data["write_" + arm].astype(float) - data["write_L"].astype(float)
        for row in range(len(wt)):
            local.append(
                dict(
                    key=case["key"],
                    arm=arm,
                    position=case["position"] + 1 + row,
                    is_verdict=row == len(wt) - 1,
                    **statistics(pred[row], wt[row]),
                )
            )
    return rows, vectors, local


def populations(rows):
    out = {}
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        schema = {
            s: sum(r["passes"] for r in rr if r["schema"] == s) for s in {r["schema"] for r in rr}
        }
        layouts = {
            s: sum(r["passes"] for r in rr if r["layout"] == s) for s in {r["layout"] for r in rr}
        }
        n = sum(schema.values())
        out[arm] = dict(
            passes=n,
            total=len(rr),
            schemas=schema,
            layouts=layouts,
            sufficient=n >= CRITERIA["population"]
            and min(schema.values()) >= CRITERIA["per_schema"]
            and min(layouts.values()) >= CRITERIA["per_layout"],
        )
    return out


def main(run, parent_run):
    d, status = read(run / "design.json"), read(run / "status.json")
    assert d["criteria"] == CRITERIA and d["bounds"] == BOUNDS
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h, n
    with np.load(run / "weights.npz") as z:
        weights = dict(z)
    assert sha(run / "weights.npz") == d["weights_sha256"]
    checks, raw, rows, vectors, local = [], [], [], [], []
    for entry, c in zip(status["cases"], d["parents"]["cases"], strict=True):
        case = c["case"]
        key = case["key"]
        p = run / (key + ".npz")
        assert (
            sha(p) == entry["arrays_sha256"]
            and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        )
        with np.load(p) as z:
            data = dict(z)
        pp = parent_run / "witnesses-native" / (key + ".npz")
        rp = parent_run / "observations" / (key + ".npz")
        assert sha(pp) == c["witness_sha256"] and sha(rp) == c["native_arrays_sha256"]
        with np.load(pp) as z:
            parent = dict(z)
        with np.load(rp) as z:
            reference = dict(z)
        bundle(case, data, parent, reference, read(p.with_suffix(".json")), weights, checks, raw)
        rr, vv, ll = scientific(case, data)
        rows.extend(rr)
        vectors.extend(vv)
        local.extend(ll)
    failed = [r for r in checks if not r["passes"]]
    table(run / "checks.csv", checks)
    table(run / "raw-coordinates.csv", raw)
    report = dict(
        experiment=300,
        retrospective=True,
        native_parents_supplied=True,
        status="failed" if failed else "passed",
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_checks=len(raw),
        raw_failed=sum(not r["passes"] for r in raw),
        scientific_outcomes_withheld=bool(failed),
        general_question_B_solved=False,
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"),
    )
    if not failed:
        table(run / "effects.csv", rows)
        save_json(run / "vector-records.json", dict(statistics=vectors))
        save_json(run / "local-records.json", dict(statistics=local))
        pop = populations(rows)
        report.update(
            populations=pop,
            H1_attention=pop["A"]["sufficient"]
            and pop["R"]["passes"] <= 2
            and pop["J"]["passes"] <= 2,
            H2_query_generator=pop["R"]["sufficient"]
            and pop["A"]["passes"] <= 2
            and pop["J"]["passes"] <= 2,
        )
    report["csv_sha256"] = {p.name: sha(p) for p in run.glob("*.csv")}
    save_json(run / "audit.json", report)
    print(report)
    assert not failed, "Required checks failed; scientific outcomes withheld"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.parent.resolve())
