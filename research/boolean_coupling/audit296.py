"""Separate query geometry, composed attention, finite head and native decisions."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha, softmax
from .audit272 import read, table
from .cache272 import save_json
from .query296 import ARMS, BOUNDS, PRIMARY, ROOT, record
from .subspace293 import statistics


def mapped(arm, q, f, key, layer):
    if arm == "N" or np.array_equal(q, f):
        return q.copy()
    if arm == "B":
        return f.copy()
    q, f = torch.tensor(q, dtype=torch.float64), torch.tensor(f, dtype=torch.float64)
    nq, nf = (
        torch.linalg.vector_norm(q, dim=-1, keepdim=True),
        torch.linalg.vector_norm(f, dim=-1, keepdim=True),
    )
    if arm == "A":
        out = q.new_tensor(1.0) * f * (nq / nf)
    elif arm == "D":
        out = q * (nf / nq)
    else:
        assert arm == "R"
        import hashlib

        seed = int.from_bytes(hashlib.sha256(f"{key}/{layer}/296".encode()).digest()[:8], "big")
        rnd = torch.tensor(np.random.default_rng(seed).normal(size=q.shape))
        unit = f / nf
        par = (q * unit).sum(-1, keepdim=True)
        tan = q - par * unit
        rnd = rnd - (rnd * unit).sum(-1, keepdim=True) * unit
        out = par * unit + torch.linalg.vector_norm(
            tan, dim=-1, keepdim=True
        ) * rnd / torch.linalg.vector_norm(rnd, dim=-1, keepdim=True)
    return out.numpy().astype(np.float32)


def head(w, state):
    x = np.asarray(state, float)
    r = np.sqrt(np.mean(x * x) + float(w["eps"]))
    return (x / r * w["norm"]) @ w["head"].T


def audit_bundle(case, data, witness, meta, w, phase, checks, raw):
    first_check = len(checks)
    for r in meta["checks"]:
        checks.append(dict(key=case["key"], **r))
    arms = ("F", *ARMS, "ZA", "ZD", "ZR") if phase == "forecast" else ("T", "O")
    for a in arms:
        record(
            checks,
            a + "/finite_head",
            head(w, data[a + "_final"][-1]),
            data[a + "_scores"],
            BOUNDS["head"],
        )
    if phase == "forecast":
        for a in ("ZA", "ZD", "ZR"):
            record(checks, a + "/saved_zero_final", data[a + "_final"], data["F_final"])
            record(checks, a + "/saved_zero_scores", data[a + "_scores"], data["F_scores"])
    else:
        record(checks, "saved_native_calibration", data["T_final"], data["O_final"])
        record(checks, "saved_native_scores", data["T_scores"], data["O_scores"])
    count = 0
    for name in witness:
        if not re.fullmatch(r"(F|A|D|B|R|T|O)_[0-9]+_q", name):
            continue
        prefix = name[:-1]
        a, layer = prefix.rstrip("_").split("_")
        layer = int(layer)
        q, c, f, k, v, mask, p, y = (
            witness[prefix + n].astype(float)
            for n in ("q", "current_q", "false_q", "k", "v", "mask", "prob", "output")
        )
        expect = mapped(
            "N" if a in ("F", "T", "O") else a,
            c.astype(np.float32),
            f.astype(np.float32),
            case["key"],
            layer,
        )
        record(checks, prefix + "independent_map", q, expect, BOUNDS["map"])
        groups = int(witness[prefix + "groups"])
        scaling = float(witness[prefix + "scaling"])
        calc = (q @ np.repeat(k, groups, axis=0).swapaxes(-1, -2)) * scaling
        actual = witness[prefix + "raw"].astype(float)
        err = float(np.max(abs(calc - actual)))
        raw.append(
            dict(
                key=case["key"],
                arm=a,
                layer=layer + 1,
                error=err,
                bound=1e-4,
                passes=bool(np.isfinite(err) and err <= 1e-4),
            )
        )
        cp = softmax(calc + mask)
        record(checks, prefix + "CPU_probability", cp, p, BOUNDS["composed"])
        record(
            checks, prefix + "CPU_output", cp @ np.repeat(v, groups, axis=0), y, BOUNDS["composed"]
        )
        record(checks, prefix + "runtime_softmax", softmax(actual + mask), p, BOUNDS["composed"])
        record(
            checks, prefix + "runtime_PV", p @ np.repeat(v, groups, axis=0), y, BOUNDS["composed"]
        )
        nq, nc, nf = (np.linalg.norm(z, axis=-1) for z in (q, c, f))
        if a == "A":
            record(checks, prefix + "magnitude", nq, nc, BOUNDS["map"])
            record(
                checks, prefix + "direction", q / nq[..., None], f / nf[..., None], BOUNDS["map"]
            )
        elif a == "D":
            record(checks, prefix + "magnitude", nq, nf, BOUNDS["map"])
            record(
                checks, prefix + "direction", q / nq[..., None], c / nc[..., None], BOUNDS["map"]
            )
        elif a == "R":
            record(checks, prefix + "magnitude", nq, nc, BOUNDS["map"])
            record(
                checks,
                prefix + "angle",
                np.sum(q * f, axis=-1) / (nq * nf),
                np.sum(c * f, axis=-1) / (nc * nf),
                BOUNDS["map"],
            )
        count += 1
    for r in checks[first_check:]:
        r.setdefault("key", case["key"])
    return count


def native_sources(case, forecast_meta, native_meta, checks):
    for arm in ARMS:
        record(
            checks,
            arm + "/independent_native_source",
            float(native_meta["source_hashes"]["T"] != forecast_meta["source_hashes"][arm]),
            0.0,
        )
        checks[-1]["key"] = case["key"]


def scientific(case, f, t):
    rows = []
    vectors = []
    for a in ARMS:
        v = {}
        for scope, selection in [("receiving", slice(1, None)), ("verdict", -1)]:
            s = statistics(
                f[a + "_final"][selection].astype(float) - f["F_final"][selection],
                t["T_final"][selection].astype(float) - f["F_final"][selection],
            )
            v[scope] = s
            vectors.append(
                dict(key=case["key"], layout=case["group"]["block"], arm=a, scope=scope, **s)
            )
        margin_error = abs(
            float(
                np.diff(f[a + "_scores"].astype(float))[0] - np.diff(t["T_scores"].astype(float))[0]
            )
        )
        passed = (
            all(
                s["norm"] >= PRIMARY["norm"]
                and s["relative"] <= PRIMARY["relative"]
                and s["cosine"] >= PRIMARY["cosine"]
                for s in v.values()
            )
            and margin_error <= PRIMARY["margin"]
        )
        rows.append(
            dict(
                key=case["key"],
                layout=case["group"]["block"],
                arm=a,
                receiving_relative=v["receiving"]["relative"],
                verdict_relative=v["verdict"]["relative"],
                margin_error=margin_error,
                passes=passed,
            )
        )
    return rows, vectors


def populations(rows):
    out = {}
    for a in ARMS:
        selected = [r for r in rows if r["arm"] == a]
        layouts = {
            s: sum(r["passes"] for r in selected if r["layout"] == s)
            for s in sorted({r["layout"] for r in selected})
        }
        n = sum(layouts.values())
        out[a] = dict(
            passes=n,
            total=len(selected),
            layouts=layouts,
            sufficient=n >= 24 and min(layouts.values()) >= 6,
        )
    return out


def main(run, phase):
    dest = run / ("pre-audit.json" if phase == "forecast" else "audit.json")
    assert not dest.exists()
    d = read(run / "design.json")
    st = read(run / "forecast-status.json")
    assert d["experiment"] == 296 and st["ordinary_true"] == 0 and len(st["cases"]) == 32
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    with np.load(run / "head.npz") as z:
        w = dict(z)
    assert sha(run / "head.npz") == d["head_sha256"]
    cases = d["cases"]
    checks = []
    raw = []
    wcount = 0
    rows = []
    vectors = []
    native = None
    if phase == "forecast":
        assert not list((run / "observations").glob("*"))
    else:
        native = read(run / "native-status.json")
        seal = read(run / "forecast-seal.json")
        proof = read(run / "measurement-proof.json")
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
    for i, (case, e) in enumerate(zip(cases, st["cases"], strict=True)):
        assert case["key"] == e["key"]
        p = run / "forecasts" / (e["key"] + ".npz")
        m = read(p.with_suffix(".json"))
        assert sha(p) == e["arrays_sha256"] and sha(p.with_suffix(".json")) == e["metadata_sha256"]
        assert m["ordinary_true"] == 0
        with np.load(p) as z:
            f = dict(z)
        wit = {}
        if e["witness_sha256"]:
            wp = run / "witnesses-forecast" / p.name
            assert sha(wp) == e["witness_sha256"]
            with np.load(wp) as z:
                wit = dict(z)
        wcount += audit_bundle(case, f, wit, m, w, "forecast", checks, raw)
        if phase == "native":
            ne = native["cases"][i]
            q = run / "observations" / p.name
            nm = read(q.with_suffix(".json"))
            assert (
                ne["key"] == case["key"]
                and sha(q) == ne["arrays_sha256"]
                and sha(q.with_suffix(".json")) == ne["metadata_sha256"]
                and nm["native_started_utc"] > proof["accepted_utc"]
                and nm["forecast_arrays_sha256"] == e["arrays_sha256"]
            )
            with np.load(q) as z:
                t = dict(z)
            nw = {}
            if ne["witness_sha256"]:
                wp = run / "witnesses-native" / p.name
                assert sha(wp) == ne["witness_sha256"]
                with np.load(wp) as z:
                    nw = dict(z)
            wcount += audit_bundle(case, t, nw, nm, w, "native", checks, raw)
            native_sources(case, m, nm, checks)
            rr, vv = scientific(case, f, t)
            rows.extend(rr)
            vectors.extend(vv)
    assert wcount == (120 if phase == "forecast" else 168)
    failed = [r for r in checks if not r["passes"]]
    stem = "pre-" if phase == "forecast" else ""
    table(run / (stem + "checks.csv"), checks)
    table(run / (stem + "raw-logits.csv"), raw)
    report = dict(
        experiment=296,
        status="failed" if failed else "passed",
        design_sha256=sha(run / "design.json"),
        forecast_status_sha256=sha(run / "forecast-status.json"),
        required_checks=len(checks),
        failed_checks=len(failed),
        raw_logit_checks=len(raw),
        raw_logit_failed=sum(not r["passes"] for r in raw),
        witness_kernels=wcount,
        scientific_outcomes_withheld=bool(failed),
        general_question_B_solved=False,
    )
    if phase == "native" and not failed:
        table(run / "native.csv", rows)
        save_json(run / "vector-records.json", dict(statistics=vectors))
        pop = populations(rows)
        report.update(
            populations=pop,
            H1_magnitude=pop["A"]["sufficient"]
            and pop["D"]["passes"] <= 8
            and pop["R"]["passes"] <= 8,
            H2_direction=pop["D"]["sufficient"]
            and pop["A"]["passes"] <= 8
            and pop["R"]["passes"] <= 8,
            native_status_sha256=sha(run / "native-status.json"),
        )
    report["csv_sha256"] = {
        p.name: sha(p)
        for p in run.glob("*.csv")
        if (phase == "native" or p.name.startswith("pre-"))
    }
    save_json(dest, report)
    print(report, flush=True)
    assert not failed, "Required numerical checks failed; science withheld"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase", choices=["forecast", "native"], required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.phase)
