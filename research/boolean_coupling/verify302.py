"""Verify 302 records and decisions without weights, activations or model imports."""

import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/first-source-geometry-302"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def yes(x):
    assert x in (True, False, "True", "False")
    return x in (True, "True")


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), (a, b)


def vectors(r):
    pp, tt, pt, ee = [
        float(r[n])
        for n in (
            "prediction_squared_norm",
            "target_squared_norm",
            "prediction_target_dot",
            "error_squared_norm",
        )
    ]
    close(ee, pp + tt - 2 * pt)
    n = math.sqrt(max(tt, 0))
    rel = math.sqrt(max(ee, 0)) / max(n, 1e-30)
    co = pt / max(math.sqrt(max(pp * tt, 0)), 1e-30)
    close(r.get("norm", n), n)
    close(r["relative"], rel)
    close(r["cosine"], co)
    close(r["signed_target_fraction"], pt / max(tt, 1e-30))
    return n, rel, co


def main():
    manifest = read(OUT / "release-manifest.json")
    for n, h in manifest["files"].items():
        assert not Path(n).is_absolute() and ".." not in Path(n).parts
        assert sha(ROOT / n) == h, n
    d = read(OUT / "design.json")
    a = read(OUT / "audit.json")
    cal = read(OUT / "calibration-audit.json")
    assert d["experiment"] == 302 and d["retrospective"] and not d["False_only_forecast"]
    assert a["status"] == cal["status"] == "passed" and not a["general_question_B_solved"]
    for n, h in d["source_hashes"].items():
        content = subprocess.check_output(["git", "show", d["source_commit"] + ":" + n], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == h, n
    assert len(d["source_hashes"]) == 168 and len(d["cases"]) == 8
    cohort = {c["key"]: c for c in d["cases"]}
    assert len(cohort) == 8 and all(
        c["group"]["reverse"] == c["group"]["target"] == 0 for c in cohort.values()
    )
    for prefix, summary in [("", a), ("calibration-", cal)]:
        ck = rows(OUT / (prefix + "checks.csv"))
        raw = rows(OUT / (prefix + "raw.csv"))
        assert len(ck) == summary["required_checks"] and all(
            yes(r["passes"]) and float(r["error"]) <= float(r["bound"]) for r in ck
        )
        assert len(raw) == summary["raw_checks"]
        assert sum(not yes(r["passes"]) for r in raw) == summary["raw_failed"]
        for r in raw:
            assert yes(r["passes"]) == (float(r["error"]) <= float(r["bound"]))
    for name, h in a["csv_sha256"].items():
        assert sha(OUT / name) == h
    inventory = read(OUT / "omitted-arrays.json")
    assert not inventory["activation_arrays_published"] and len(inventory["arrays"]) == 24
    scores = {key: {} for key in cohort}
    for c in read(OUT / "scores.json")["cases"]:
        scores[c["key"]].update(c["scores"])
    assert all(len(s) == 48 for s in scores.values())
    vector = read(OUT / "vector-records.json")["records"]
    assert len(vector) == 448
    ix = {(r["key"], r["arm"], r["alias"], r["scope"]): r for r in vector}
    assert len(ix) == 448
    for r in vector:
        vectors(r)
    effect = rows(OUT / "effects.csv")
    assert len(effect) == 56
    arms = ["K", "V", "H", "S", "RH", "RK", "RV"]
    decisions = {}
    for r in effect:
        key, arm = r["key"], r["arm"]
        ok, local = True, True
        for alias in "AB":
            sign = 1 if alias == "A" else -1

            def coupling(branch):
                vals = scores[key]
                t, f = vals[f"{branch}_{alias}_T_scores"], vals[f"{branch}_{alias}_F_scores"]
                return (t[1] - t[0]) - (f[1] - f[0])

            target = sign * (coupling("KV") - coupling("N"))
            pred = sign * (coupling(arm) - coupling("N"))
            error = abs(pred - target)
            bound = min(0.1, 0.25 * abs(target))
            material = abs(target) >= 0.02
            for name, v in [
                ("target_margin", target),
                ("predicted_margin", pred),
                ("margin_error", error),
                ("margin_bound", bound),
            ]:
                close(r[alias + "_" + name], v)
            assert yes(r[alias + "_material"]) == material
            ok &= material and error <= bound
            for scope in ["source", "emit", "receiving", "verdict"]:
                n, re, co = vectors(ix[key, arm, alias, scope])
                v = n >= 0.001 and re <= 0.25 and co >= 0.95
                ok &= v
                if scope in ["source", "emit"]:
                    local &= v
                close(r[alias + "_" + scope + "_relative"], re)
        assert yes(r["passes"]) == ok and yes(r["local_passes"]) == local
        decisions[key, arm] = ok
    populations = {}
    for arm in arms:
        rr = [r for r in effect if r["arm"] == arm]
        layouts = {
            v: sum(yes(r["passes"]) for r in rr if r["layout"] == v)
            for v in sorted({r["layout"] for r in rr})
        }
        orders = {
            str(o): sum(yes(r["passes"]) for r in rr if int(r["mention_order"]) == o)
            for o in (0, 1)
        }
        count = sum(yes(r["passes"]) for r in rr)
        populations[arm] = dict(
            passes=count,
            local_passes=sum(yes(r["local_passes"]) for r in rr),
            total=8,
            layouts=layouts,
            mention_orders=orders,
            sufficient=count >= 6
            and all(v >= 1 for v in layouts.values())
            and all(v >= 3 for v in orders.values()),
        )
    assert populations == a["populations"]
    for name, candidate, competitor, random in [
        ("H_key", "K", "V", "RK"),
        ("H_value", "V", "K", "RV"),
        ("H_hidden", "H", "S", "RH"),
    ]:
        assert a[name] == (
            populations[candidate]["sufficient"]
            and populations[competitor]["passes"] <= 2
            and populations[random]["passes"] <= 2
        )
    phase = rows(OUT / "phase.csv")
    assert len(phase) == 8
    for r in phase:
        n, _, _ = vectors(r)
        ratio = math.sqrt(float(r["prediction_squared_norm"])) / max(n, 1e-30)
        close(r["remaining_norm_ratio"], ratio)
        assert yes(r["passes"]) == (n >= 0.001 and ratio <= 0.25)
    hp = sum(yes(r["passes"]) for r in phase) >= 6
    hp &= all(
        sum(yes(r["passes"]) for r in phase if r["layout"] == v) >= 1
        for v in {r["layout"] for r in phase}
    )
    hp &= all(
        sum(yes(r["passes"]) for r in phase if int(r["mention_order"]) == o) >= 3 for o in (0, 1)
    )
    assert a["H_phase"] == hp
    assert a["material_both_references"] == sum(
        yes(r["A_material"]) and yes(r["B_material"]) for r in effect if r["arm"] == "K"
    )
    summary = read(OUT / "descriptive-summary.json")
    assert summary["posthoc"] and summary["all_eight_cases"]
    for arm, items in summary["medians_of_worse_reference"].items():
        rr = [r for r in effect if r["arm"] == arm]
        for name, value in items.items():
            close(
                value,
                statistics.median(max(float(r["A_" + name]), float(r["B_" + name])) for r in rr),
            )
    joint = read(OUT / "posthoc-joint/provenance.json")
    assert (
        joint["posthoc"] and not joint["threshold_or_primary_changed"] and joint["model_calls"] == 0
    )
    assert sha(OUT / "posthoc-joint/joint.csv") == joint["output_sha256"]
    assert joint["script_sha256"] == sha(ROOT / "research/boolean_coupling/joint302.py")
    assert joint["signed_identity_max_error"] <= 1e-9
    for r in rows(OUT / "posthoc-joint/joint.csv"):
        vectors(r)
    for r in rows(OUT / "ledger.csv"):
        vectors(r)
    cost = read(OUT / "cost.json")
    assert cost["full_checkpoint_forwards"] == 384 and cost["extra_full_QK"] == 736
    assert (
        cost["extra_full_PV"] == cost["extra_full_softmax"] == 352
        and not cost["wallclock_speedup_claimed"]
    )
    assert (
        cost["raw_array_bytes"] == sum(r["bytes"] for r in inventory["arrays"])
        and cost["raw_array_bytes"] < 128 * 1024**2
    )
    for name in ["calibration", "intervention"]:
        st = read(OUT / (name + "-status.json"))
        assert st["passed"] and len(st["cases"]) == 8
        assert st["counts"]["full_forwards"] == (32 if name == "calibration" else 352)
    cp = read(OUT / "cpu-qualification.json")
    assert cp["passed"] and cp["source_hashes"] == d["source_hashes"]
    prov = read(ROOT / "docs/figures/first-source-geometry-302/provenance.json")
    assert prov["script_sha256"] == sha(ROOT / "research/boolean_coupling/describe302.py")
    for n, h in prov["inputs"].items():
        assert sha(ROOT / n) == h
    for n, h in prov["outputs"].items():
        assert sha(ROOT / "docs/figures/first-source-geometry-302" / n) == h
    print(
        json.dumps(
            dict(
                passed=True,
                required_checks=a["required_checks"],
                raw_failed=a["raw_failed"],
                hypotheses={k: a[k] for k in ["H_key", "H_value", "H_hidden", "H_phase"]},
                all_eight_cases_retained=True,
                omitted_arrays_replicated=False,
                general_question_B_solved=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
