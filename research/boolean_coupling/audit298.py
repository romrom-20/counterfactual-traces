"""Independent message sums, finite terminal reconstruction and joint decisions."""

import argparse
from pathlib import Path

import numpy as np

from .audit272 import table
from .cache272 import save_json
from .messages298 import BOUNDS, GROUPS, PRIMARY, ROOT, read_parent, record
from .publish285 import read, sha
from .subspace293 import statistics


def cpu_final(w, middle):
    x = np.asarray(middle, float)
    z = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + w["eps"]) * w["norm"]
    g, u = z @ w["gate"].T, z @ w["up"].T
    sig = np.empty_like(g)
    positive = g >= 0
    sig[positive] = 1 / (1 + np.exp(-g[positive]))
    negative_exp = np.exp(g[~positive])
    sig[~positive] = negative_exp / (1 + negative_exp)
    return x + ((g * sig) * u) @ w["down"].T


def cpu_head(w, final):
    x = np.asarray(final[-1], float)
    z = x / np.sqrt(np.mean(x * x) + w["final_eps"]) * w["final_norm"]
    return z @ w["head"].T


def independent_deltas(parent, selection, wo):
    p = parent["p_native"].astype(float) - parent["p_false_query"].astype(float)
    v = np.repeat(parent["values"].astype(float), int(parent["groups"]), axis=0)
    result = {}
    for name, positions in selection["positions"].items():
        heads = np.zeros((*p.shape[:-1], v.shape[-1]))
        for h in range(p.shape[0]):
            for row in range(p.shape[1]):
                for k in positions:
                    heads[h, row] += p[h, row, k] * v[h, k]
        flat = np.concatenate([heads[h] for h in range(len(heads))], axis=-1)
        result[name] = flat @ wo.T
    return result


def qualify(selection, parent, data, producer_checks, w, checks, raw):
    first = len(checks)
    checks.extend(dict(key=selection["key"], **r) for r in producer_checks)
    expected = independent_deltas(parent, selection, w["wo"])
    for name, value in expected.items():
        record(
            checks, name + "/independent_message", data["delta_" + name], value, BOUNDS["group_map"]
        )
    record(
        checks, "independent_native_final", data["native_reference_final"], parent["expected_final"]
    )
    record(
        checks,
        "independent_native_scores",
        data["native_reference_scores"],
        parent["expected_scores"],
    )
    for group in GROUPS:
        for operation, baseline, sign in [
            ("add", "frozen_query_reference", 1),
            ("remove", "native_reference", -1),
        ]:
            expected_write = np.add(
                data[baseline + "_write"],
                np.float32(sign) * data["delta_" + group].astype(np.float32),
                dtype=np.float32,
            )
            record(
                checks,
                group + "_" + operation + "/independent_installation",
                data[group + "_" + operation + "_write"],
                expected_write,
            )
    names = ["native_reference", "frozen_query_reference"] + [
        g + "_" + op for g in GROUPS for op in ["add", "remove"]
    ]
    for name in names:
        expected_final = cpu_final(w, data[name + "_middle"])
        actual = data[name + "_final"].astype(float)
        record(checks, name + "/CPU_final_max", actual, expected_final, BOUNDS["final_max"])
        relative = np.linalg.norm(actual - expected_final) / max(
            np.linalg.norm(expected_final), 1e-30
        )
        record(checks, name + "/CPU_final_relative", relative, 0.0, BOUNDS["final_relative"])
        record(
            checks,
            name + "/finite_head",
            data[name + "_scores"],
            cpu_head(w, actual),
            BOUNDS["head"],
        )
        err = float(np.max(abs(actual - expected_final)))
        raw.append(
            dict(
                key=selection["key"],
                arm=name,
                error=err,
                bound=BOUNDS["raw_diagnostic"],
                passes=bool(np.isfinite(err) and err <= BOUNDS["raw_diagnostic"]),
            )
        )
        record(
            checks,
            name + "/independent_middle",
            data[name + "_middle"],
            np.add(parent["input"], data[name + "_write"], dtype=np.float32),
        )
    for label in ["random1", "random2"]:
        target = np.linalg.norm(expected["chosen"], axis=-1)
        actual = np.linalg.norm(data["delta_" + label].astype(np.float32).astype(float), axis=-1)
        record(
            checks,
            label + "/independent_norm_match",
            (actual - target) / np.maximum(target, 1e-30),
            np.zeros_like(target),
            BOUNDS["random_norm_relative"],
        )
    for r in checks[first:]:
        r.setdefault("key", selection["key"])


def margin(data, name):
    a = data[name + "_scores"].astype(float)
    return float(a[1] - a[0])


def scientific(selection, data):
    native, baseline = (
        data["native_reference_final"].astype(float),
        data["frozen_query_reference_final"].astype(float),
    )
    target = native - baseline
    dm = margin(data, "native_reference") - margin(data, "frozen_query_reference")
    bound = min(PRIMARY["margin_absolute"], PRIMARY["margin_relative"] * abs(dm))
    effects, vectors, joint = [], [], []
    for group in GROUPS:
        passed = []
        for operation in ["add", "remove"]:
            name = group + "_" + operation
            pred = (
                data[name + "_final"].astype(float) - baseline
                if operation == "add"
                else native - data[name + "_final"].astype(float)
            )
            candidate_effect = (
                margin(data, name) - margin(data, "frozen_query_reference")
                if operation == "add"
                else margin(data, "native_reference") - margin(data, name)
            )
            error = abs(candidate_effect - dm)
            ok = abs(dm) >= PRIMARY["material"] and error <= bound
            stats = {}
            for scope, index in [("receiving", slice(None)), ("verdict", -1)]:
                s = statistics(pred[index], target[index])
                stats[scope] = s
                ok = (
                    ok
                    and s["norm"] >= PRIMARY["norm"]
                    and s["relative"] <= PRIMARY["relative"]
                    and s["cosine"] >= PRIMARY["cosine"]
                )
                vectors.append(
                    dict(key=selection["key"], group=group, operation=operation, scope=scope, **s)
                )
            effects.append(
                dict(
                    key=selection["key"],
                    schema=selection["group"]["schema"],
                    layout=selection["group"]["block"],
                    group=group,
                    operation=operation,
                    native_margin_effect=dm,
                    candidate_margin_effect=candidate_effect,
                    margin_error=error,
                    margin_bound=bound,
                    receiving_relative=stats["receiving"]["relative"],
                    verdict_relative=stats["verdict"]["relative"],
                    passes=bool(ok),
                )
            )
            passed.append(ok)
        joint.append(
            dict(
                key=selection["key"],
                schema=selection["group"]["schema"],
                layout=selection["group"]["block"],
                group=group,
                eligible_margin=abs(dm) >= PRIMARY["material"],
                passes=bool(all(passed)),
            )
        )
    return effects, vectors, joint


def populations(rows):
    result = {}
    for group in GROUPS:
        rr = [r for r in rows if r["group"] == group]
        schemas = {
            s: sum(r["passes"] for r in rr if r["schema"] == s)
            for s in sorted({r["schema"] for r in rr})
        }
        layouts = {
            s: sum(r["passes"] for r in rr if r["layout"] == s)
            for s in sorted({r["layout"] for r in rr})
        }
        n = sum(r["passes"] for r in rr)
        result[group] = dict(
            passes=n,
            total=len(rr),
            schemas=schemas,
            layouts=layouts,
            sufficient=n >= 6 and min(schemas.values()) >= 3 and min(layouts.values()) >= 1,
        )
    return result


def main(run, parent_folder, weights):
    assert not (run / "audit.json").exists()
    design, status = read(run / "design.json"), read(run / "status.json")
    assert (
        design["experiment"] == 298 and design["retrospective"] and status["status"] == "complete"
    )
    assert design["primary"] == PRIMARY and design["bounds"] == BOUNDS
    for n, h in design["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h, n
    checks, raw, effects, vectors, joint = [], [], [], [], []
    for selection, entry in zip(design["cases"], status["cases"], strict=True):
        assert selection["key"] == entry["key"]
        p = run / "cases" / (entry["key"] + ".npz")
        meta = read(p.with_suffix(".json"))
        assert (
            sha(p) == entry["arrays_sha256"]
            and sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        )
        with np.load(p) as archive:
            data = dict(archive)
        parent = read_parent(parent_folder, selection)
        qualify(selection, parent, data, meta["checks"], weights, checks, raw)
        rr, vv, jj = scientific(selection, data)
        effects.extend(rr)
        vectors.extend(vv)
        joint.extend(jj)
    failed = [r for r in checks if not r["passes"]]
    table(run / "checks.csv", checks)
    table(run / "raw-coordinate.csv", raw)
    result = dict(
        experiment=298,
        retrospective=True,
        status="failed" if failed else "passed",
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_coordinate_checks=len(raw),
        raw_coordinate_failed=sum(not r["passes"] for r in raw),
        scientific_outcomes_withheld=bool(failed),
        general_question_B_solved=False,
    )
    if not failed:
        table(run / "effects.csv", effects)
        table(run / "joint.csv", joint)
        save_json(run / "vectors.json", dict(statistics=vectors))
        pop = populations(joint)
        result.update(
            populations=pop,
            H1=bool(
                pop["chosen"]["sufficient"]
                and all(
                    pop[c]["passes"] <= 2
                    for c in ["swapped", "complement", "neighbor", "random1", "random2"]
                )
            ),
            eligible_margin_cases=sum(
                r["eligible_margin"] for r in joint if r["group"] == "chosen"
            ),
        )
    result["csv_sha256"] = {p.name: sha(p) for p in run.glob("*.csv")}
    save_json(run / "audit.json", result)
    print(result, flush=True)
    assert not failed, "Required numerical checks failed; science withheld"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    from .messages298 import reference_weights

    main(
        args.run.resolve(),
        args.parent.resolve(),
        reference_weights(read(args.run.resolve() / "design.json")["checkpoint"]),
    )
