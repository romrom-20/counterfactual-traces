"""Read-only baseline namespace correction; no new model/intervention computation."""

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .cache272 import save_json
from .run286 import cpu_head
from .subspace293 import ARMS, PRIMARY, ROOT, statistics

SOURCES = (
    "research/boolean_coupling/namespace293c.py",
    "research/boolean_coupling/preflight293c.py",
    "docs/experiments/293c-baseline-namespace-qualification.md",
)


def margin(v):
    return float(v[1]) - float(v[0])


def baseline(pack):
    return dict(final=pack["F_final"], scores=pack["F_scores"], first_source=pack["F_states"][0, 0])


def install_baseline(pack, b, expected):
    for field, value in b.items():
        assert np.array_equal(value, expected[field]), ("False baseline mismatch", field)
        assert np.asarray(value).dtype == np.float32
        pack["B0_" + field] = np.asarray(value).copy()
    return pack


def score(bank, cases):
    native = []
    pairs = []
    vectors = []

    def vector(p, t, rel, cos):
        v = statistics(p, t)
        return dict(**v, passes=v["norm"] >= 0.001 and v["relative"] <= rel and v["cosine"] >= cos)

    for c in cases:
        a = bank[c["key"]]
        for arm in ARMS:
            met = {}
            for scope, row in [("receiving", slice(1, None)), ("verdict", -1)]:
                v = vector(
                    a[arm + "_final"][row].astype(float) - a["B0_final"][row],
                    a["T_final"][row].astype(float) - a["B0_final"][row],
                    0.05,
                    0.99,
                )
                met[scope] = v
                vectors.append(
                    dict(
                        key=c["key"],
                        arm=arm,
                        scope=scope,
                        contrast="native",
                        **{k: x for k, x in v.items() if k != "passes"},
                    )
                )
            err = abs(margin(a[arm + "_scores"]) - margin(a["T_scores"]))
            native.append(
                dict(
                    key=c["key"],
                    **c["group"],
                    arm=arm,
                    receiving_relative=met["receiving"]["relative"],
                    receiving_cosine=met["receiving"]["cosine"],
                    verdict_relative=met["verdict"]["relative"],
                    verdict_cosine=met["verdict"]["cosine"],
                    margin_error=err,
                    passes=met["receiving"]["passes"] and met["verdict"]["passes"] and err <= 0.1,
                )
            )
    for block in sorted({c["group"]["block"] for c in cases}):
        cs = sorted(
            [c for c in cases if c["group"]["block"] == block], key=lambda c: c["group"]["schema"]
        )
        assert len(cs) == 2
        a, b = (bank[c["key"]] for c in cs)

        def contrast(arm, field, row):
            return (a[arm + "_" + field].astype(float) - a["B0_" + field])[row] - (
                b[arm + "_" + field].astype(float) - b["B0_" + field]
            )[row]

        for arm in ARMS:
            met = {}
            for scope, field, row in [
                ("source", "first_source", slice(None)),
                ("verdict", "final", -1),
            ]:
                v = vector(contrast(arm, field, row), contrast("T", field, row), 0.1, 0.995)
                met[scope] = v
                vectors.append(
                    dict(
                        block=block,
                        arm=arm,
                        scope=scope,
                        contrast="pairs",
                        **{k: x for k, x in v.items() if k != "passes"},
                    )
                )

            def effect(arm):
                return (
                    margin(a[arm + "_scores"])
                    - margin(a["B0_scores"])
                    - margin(b[arm + "_scores"])
                    + margin(b["B0_scores"])
                )

            pred, target = effect(arm), effect("T")
            pairs.append(
                dict(
                    block=block,
                    arm=arm,
                    source_relative=met["source"]["relative"],
                    source_cosine=met["source"]["cosine"],
                    verdict_relative=met["verdict"]["relative"],
                    verdict_cosine=met["verdict"]["cosine"],
                    prediction=pred,
                    target=target,
                    error=abs(pred - target),
                    passes=met["source"]["passes"]
                    and met["verdict"]["passes"]
                    and abs(pred - target) <= 0.1,
                )
            )
    return native, pairs, vectors


def main(parent, out, reference=None, synthetic=False, baselines=None, targets=None):
    assert not out.exists()
    d = read(parent / "design.json")
    st = read(parent / "status.json")
    old = read(parent / "audit.json")
    assert old["status"] == "passed" and old["failed_checks"] == 0 and d["primary"] == PRIMARY
    out.mkdir()
    bank = {}
    proofs = []
    checks = []
    scores = []
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: sha(ROOT / n) for n in SOURCES}
    if not synthetic:
        for n, h in hashes.items():
            assert (
                hashlib.sha256(
                    subprocess.check_output(["git", "show", commit + ":" + n], cwd=ROOT)
                ).hexdigest()
                == h
            )
        assert sha(reference / "design.json") == d["reference_hashes"]["design.json"]
        fs, ns = read(reference / "forecast-status.json"), read(reference / "native-status.json")
    with np.load(parent / "head.npz") as z:
        head = {n: z[n] for n in z.files}
    for c, e in zip(d["cases"], st["cases"], strict=True):
        key = c["key"]
        p = parent / "cases" / (key + ".npz")
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        with np.load(p) as z:
            a = {n: z[n] for n in z.files}
        if synthetic:
            b = baselines[key]
            t = targets[key]
            refproof = {}
        else:
            fe = next(x for x in fs["cases"] if x["key"] == key)
            te = next(x for x in ns["cases"] if x["key"] == key)
            fp = reference / "forecasts" / (key + ".npz")
            tp = reference / "native" / (key + ".npz")
            assert (
                sha(fp) == fe["arrays_sha256"]
                and sha(fp.with_suffix(".json")) == fe["metadata_sha256"]
                and sha(tp) == te["arrays_sha256"]
                and sha(tp.with_suffix(".json")) == te["metadata_sha256"]
            )
            with np.load(fp) as z:
                f = {n: z[n] for n in z.files}
                b = baseline(f)
            with np.load(tp) as z:
                t = dict(final=z["final"], scores=z["scores"], first_source=z["states"][0, 0])
            refproof = dict(forecast=fe, native=te)
        install_baseline(a, b, b)
        for n, v in t.items():
            assert np.array_equal(v, a["T_" + n])
        m = read(p.with_suffix(".json"))
        # Recorded zero-edit finite scores bind B0 separately from condition F.
        for arm in ARMS:
            rr = [r for r in m["arms"][arm + "0"]["engine"]["checks"] if r["name"] == "zero_scores"]
            assert len(rr) == 1 and rr[0]["passes"] and rr[0]["error"] == 0.0
        err = float(np.max(abs(cpu_head(head, a["B0_final"]) - a["B0_scores"])))
        checks.append(
            dict(key=key, name="CPU64_B0_head", error=err, bound=0.005, passes=err <= 0.005)
        )
        proofs.append(
            dict(
                key=key,
                parent=e,
                reference=refproof,
                baseline={
                    n: dict(
                        shape=list(v.shape),
                        dtype=str(v.dtype),
                        sha256=hashlib.sha256(v.tobytes()).hexdigest(),
                    )
                    for n, v in b.items()
                },
                fixed_condition_equals_baseline=bool(np.array_equal(a["F_final"], b["final"])),
            )
        )
        scores.append(dict(key=key, scores={n: a[n].tolist() for n in a if n.endswith("_scores")}))
        bank[key] = a
    assert not any(r["fixed_condition_equals_baseline"] for r in proofs), (
        "Fixture must distinguish baseline and condition"
    )
    native, pairs, vectors = score(bank, d["cases"])
    for n, rr in [("checks", checks), ("native", native), ("pairs", pairs)]:
        table(out / (n + ".csv"), rr)
    outcomes = {}
    for arm in ARMS:
        rr = [r for r in native if r["arm"] == arm]
        pp = [r for r in pairs if r["arm"] == arm]
        lc = {
            b: sum(r["passes"] for r in rr if r["block"] == b)
            for b in sorted({r["block"] for r in rr})
        }
        n = sum(lc.values())
        pn = sum(r["passes"] for r in pp)
        outcomes[arm] = dict(
            native=n,
            total=8,
            layouts=lc,
            pairs=pn,
            pair_total=4,
            sufficient=n >= 6 and min(lc.values()) >= 1 and pn >= 3,
        )
    save_json(
        out / "design.json",
        dict(
            experiment="293c",
            retrospective=True,
            synthetic=synthetic,
            model_calls=0,
            source_commit=commit,
            source_hashes=hashes,
            parent_hashes={
                n: sha(parent / n)
                for n in ("design.json", "status.json", "audit.json", "checks.csv")
            },
            primary=PRIMARY,
            boundary="Original293b native/pair tables invalid. Candidate arrays unchanged; B0 restored from immutable292False. No new predictions or model calls.",
        ),
    )
    save_json(out / "baseline-proof.json", dict(cases=proofs))
    save_json(out / "scores.json", dict(cases=scores))
    save_json(out / "vector-records.json", dict(model_calls=0, statistics=vectors))
    failed = [r for r in checks if not r["passes"]]
    report = dict(
        experiment="293c",
        status="failed" if failed else "passed",
        original_scientific_decisions_valid=False,
        original_primitive_checks_retained=old["required_checks"],
        new_required_checks=len(checks),
        failed_checks=len(failed),
        outcomes=outcomes,
        primary_pass=outcomes["E"]["sufficient"],
        adaptation_specific=outcomes["E"]["sufficient"]
        and outcomes["F"]["native"] <= 2
        and outcomes["R"]["native"] <= 2,
        general_question_B_solved=False,
        csv_sha256={n + ".csv": sha(out / (n + ".csv")) for n in ("checks", "native", "pairs")},
    )
    save_json(out / "audit.json", report)
    print(report, flush=True)
    assert not failed
    return bank


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    main(a.parent.resolve(), a.out.resolve(), a.reference.resolve())
