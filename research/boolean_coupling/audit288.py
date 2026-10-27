"""Saved-array numerical gate, then signed accounting for frozen experiment 288."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .audit287 import metrics
from .cache272 import save_json
from .composition280 import digest
from .qualify287b import vector_stats
from .readout286 import ROOT, margin
from .routing288 import ARMS, BOUNDS, PRIMARY, SOURCES
from .run286 import cpu_head, entry_data
from .run288 import references


def expected_counts(layers):
    return dict(full_F=1, first_block=1, anchor_suffixes=2, anchor_blocks=2 * layers,
                modified_forwards=112, modified_blocks=168 * layers, modified_heads=112,
                tape_backwards=8, triangular_blocks=12 * layers,
                triangular_block_VJPs=12 * layers, triangular_head_VJPs=8,
                native_suffixes=2, native_blocks=2 * layers)


def numerical(run, reference, previous, qualification):
    d, status = read(run / "design.json"), read(run / "status.json")
    assert d["experiment"] == 288 and d["retrospective"] and status["status"] == "complete"
    assert d["numerical_bounds"] == BOUNDS and d["primary"] == PRIMARY
    assert d["FD_steps"] == [0.05, 0.025] and d["arms"] == list(ARMS)
    assert d["resource_guards_MiB"] == [640, 384, 128]
    assert d["factors"] == ["source_blocks", "receiving_blocks", "final_head"]
    assert d["anchors"] == ["False", "source-only-MF"]
    assert d["reference_hashes"] == references(reference, previous, qualification)
    assert set(d["source_hashes"]) == set(SOURCES)
    for name, value in d["source_hashes"].items():
        assert sha(ROOT / name) == sha(run / "sources" / Path(name).name) == value
    assert read(qualification / "audit.json")["status"] == "qualified_boundary_only"
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ps = {e["key"]: e for e in read(previous / "status.json")["cases"]}
    assert d["cases"] == read(previous / "design.json")["cases"]
    assert len(d["cases"]) == (4 if d["synthetic"] else 32)
    assert [e["key"] for e in status["cases"]] == [c["key"] for c in d["cases"]]
    assert sum(e["bytes"] for e in status["cases"]) == status["raw_bytes"] <= 128 * 1024**2
    assert sha(previous / "head.npz") == read(previous / "head.json")["arrays_sha256"]
    with np.load(previous / "head.npz") as h:
        head = {n: h[n] for n in h.files}
    bank, required, derivative_rows, trace_rows, raw, counts = {}, [], [], [], [], []
    for entry, case in zip(status["cases"], d["cases"], strict=True):
        key = entry["key"]
        a = entry_data(run, "cases", entry)
        f, p = entry_data(reference, "forecasts", fs[key]), entry_data(previous, "cases", ps[key])
        meta = read(run / "cases" / (key + ".json"))
        assert meta["forecast_entry"] == fs[key] and meta["previous_entry"] == ps[key]
        assert meta["prefix_unchanged"] and meta["prefix_seals"] == read(previous / "cases" / (key + ".json"))["prefix_seals"]
        policy = meta["first_source_policy"]
        assert policy["activation_calls"] == 1 and policy["False_primal_exact"]
        assert policy["input_gate_sha256"] == digest(f["F_gate"])
        assert policy["output_phi_sha256"] == digest(f["F_phi"])
        assert all(np.isfinite(x).all() for x in a.values())
        layers = len(a["000_source_trace"])
        assert meta["counts"] == expected_counts(layers)
        counts.append(meta["counts"])
        computed = {}

        def check(name, actual, expected, bound=0.0):
            err = float(np.max(abs(np.asarray(actual, float) - np.asarray(expected, float))))
            computed[name] = dict(name=name, error=err, bound=bound,
                                  passes=bool(np.isfinite(err) and err <= bound))

        check("ordinary_F_final", a["ordinary_F_final"], p["ordinary_F_final"], 1e-4)
        check("ordinary_F_scores", a["ordinary_F_scores"], p["ordinary_F_scores"], 1e-4)
        check("initial_F_source", a["initial_F"][0], f["F_source"])
        check("initial_E_source", a["initial_E"][0], f["M_source"])
        check("initial_receivers_unchanged", a["initial_E"][1:], a["initial_F"][1:])
        check("delta_identity", a["delta"], f["C_source"].astype(float) - f["M_source"])
        for n in ("gate", "phi", "write"):
            check("False_" + n, a["F_" + n], f["F_" + n])
        for anchor, corner in ((0, "FF"), (1, "MF")):
            check(f"anchor{anchor}_primal", a[f"anchor{anchor}_final"], p[corner + "_final"], 1e-4)
            check(f"anchor{anchor}_scores", a[f"anchor{anchor}_scores"], p[corner + "_scores"], 1e-4)
            check(f"anchor{anchor}_old_finite_final", a[f"anchor{anchor}_native_bump_final"], p[corner + "_plus_delta_final"], 0.002)
            check(f"anchor{anchor}_old_finite_scores", a[f"anchor{anchor}_native_bump_scores"], p[corner + "_plus_delta_scores"], 0.001)
        for arm in ARMS:
            tr, vjp = a[arm + "_source_trace"], a[arm + "_triangular_source_trace"]
            assert tr.shape == vjp.shape == (layers, len(a["delta"]))
            assert np.array_equal(tr[0], a[arm + "_gradient"])
            assert np.array_equal(vjp[0], a[arm + "_triangular_gradient"])
            check(arm + "_zero_offset", a[arm + "_zero_offset"], 0.0)
            check(arm + "_zero_head", a[arm + "_zero_raw_scores"], a[f"anchor{int(arm[-1])}_scores"])
            check(arm + "_boundary_adjoint", tr[0], vjp[0], 1e-4)
            noncausal = a[arm + "_triangular_noncausal"]
            check(arm + "_noncausal", np.max(noncausal) if noncausal.size else 0.0, 0.0)
            check(arm + "_CPU64_head", cpu_head(head, a[arm + "_finite_head_state"]), a[arm + "_finite_raw_scores"], 0.005)
            lift = a["anchor1_scores"].astype(float) + a[arm + "_finite_raw_scores"].astype(float) - a[f"anchor{int(arm[-1])}_scores"]
            assert np.array_equal(lift, a[arm + "_finite_lifted_scores"])
            # The head state is at H's anchor, the conceptual primal is always E.
            assert np.max(abs(a[arm + "_finite_head_state"].astype(float)
                              - (a[f"anchor{int(arm[-1])}_final"].astype(float)
                                 + a[arm + "_finite_offset"]))) <= 0.0001
            if arm in ("000", "111"):
                anchor, corner = (0, "FF") if arm == "000" else (1, "MF")
                check(arm + "_old_cached_gradient", tr[0], p[corner + "_gradient"], 1e-4)
                check(arm + "_old_full_gradient", tr[0], p[corner + "_full_source_trace"][0], 1e-4)
                check(arm + "_matched_native_final", a[f"anchor{anchor}_final"].astype(float) + a[arm + "_finite_offset"], a[f"anchor{anchor}_native_bump_final"], 0.002)
                check(arm + "_matched_native_scores", a[arm + "_finite_raw_scores"], a[f"anchor{anchor}_native_bump_scores"], 0.001)
            errs = abs(tr.astype(float) - vjp.astype(float)).max(axis=1)
            trace_rows.extend(dict(key=key, arm=arm, input_before_block=j + 2,
                                   error=float(err), bound=1e-4, passes=bool(err <= 1e-4),
                                   required_boundary=j == 0) for j, err in enumerate(errs))
        assert set(computed) == {r["name"] for r in meta["checks"]}
        for row in meta["checks"]:
            got = computed[row["name"]]
            assert got["bound"] == row["bound"] and abs(got["error"] - row["error"]) <= 1e-10
            assert got["passes"] == row["passes"]
            required.append(dict(key=key, **got))
            if row["name"].endswith(("old_finite_final", "matched_native_final")):
                raw.append(dict(key=key, name=row["name"], error=got["error"], original_bound=1e-4,
                                passes=got["error"] <= 1e-4, required_bound=got["bound"]))
        coords = np.random.default_rng(int(key[:16], 16)).choice(len(a["delta"]), 2, replace=False)
        directions = {"delta": a["delta"]}
        for c in coords:
            v = np.zeros_like(a["delta"])
            v[c] = 1
            directions["coordinate_" + str(c)] = v
        index = {(r["arm"], r["direction"], r["step"]): r for r in meta["derivatives"]}
        assert len(index) == len(meta["derivatives"]) == 72
        assert set(index) == set(itertools.product(ARMS, directions, (0.05, 0.025, "halving")))
        for (arm, label, step), row in index.items():
            ad = float(a[arm + "_gradient"].astype(float) @ directions[label])
            assert ad == row["AD"]
            if step == "halving":
                err = abs(index[arm, label, 0.05]["FD"] - index[arm, label, 0.025]["FD"])
                bound = 0.002 + 0.03 * abs(ad)
                assert row["FD"] == index[arm, label, 0.025]["FD"]
            else:
                fd = (margin(row["plus_scores"]) - margin(row["minus_scores"])) / (2 * step)
                assert row["FD"] == fd
                err, bound = abs(ad - fd), 0.001 + 0.02 * abs(ad)
            assert row["error"] == err and row["bound"] == bound
            derivative_rows.append(dict(key=key, **row, passes=bool(np.isfinite(err) and err <= bound)))
        bank[key] = a
    assert status["counts"] == {n: sum(r[n] for r in counts) for n in counts[0]}
    return bank, required, derivative_rows, trace_rows, raw


def scientific(run, reference, bank):
    cases = read(run / "design.json")["cases"]
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ns = {e["key"]: e for e in read(reference / "native-status.json")["cases"]}
    hypotheses, ledger, effects, native = [], [], [], []
    for case in cases:
        key, group = case["key"], case["group"]
        a = bank[key]
        g = {arm: a[arm + "_gradient"].astype(float) for arm in ARMS}
        e = {arm: margin(a[arm + "_finite_lifted_scores"]) - margin(a["anchor1_scores"]) for arm in ARMS}
        target, finite_target = g["111"] - g["000"], e["111"] - e["000"]
        for name, arm in (("SOURCE", "100"), ("RECEIVING_AND_HEAD", "011"),
                          ("RECEIVING_BLOCKS", "010"), ("HEAD", "001"), ("ADD", "ADD")):
            p = g[arm] - g["000"] if arm != "ADD" else g["100"] + g["011"] - 2 * g["000"]
            pe = e[arm] - e["000"] if arm != "ADD" else e["100"] + e["011"] - 2 * e["000"]
            met, err = metrics(p, target), abs(pe - finite_target)
            vp = met["norm"] >= 0.001 and met["relative"] <= 0.25 and met["cosine"] >= 0.95
            fp = err <= 0.005 + 0.25 * abs(finite_target)
            hypotheses.append(dict(key=key, **group, hypothesis=name, **met, **vector_stats(p, target),
                                   finite_target=finite_target, finite_prediction=pe, finite_error=err,
                                   vector_pass=vp, finite_pass=fp, passes=vp and fp,
                                   material_eligible=abs(finite_target) >= 0.02,
                                   material_pass=vp and fp and abs(finite_target) >= 0.02))
        pieces, finite_pieces = [], []
        for arm in ARMS[1:]:
            subset = [j for j, bit in enumerate(arm) if bit == "1"]
            v, fe = np.zeros_like(target), 0.0
            for bits in itertools.product((0, 1), repeat=len(subset)):
                corner = ["0"] * 3
                for j, b in zip(subset, bits, strict=True):
                    corner[j] = str(b)
                corner = "".join(corner)
                sign = (-1) ** (len(subset) - sum(bits))
                v += sign * g[corner]
                fe += sign * e[corner]
            pieces.append(v)
            finite_pieces.append(fe)
            ledger.append(dict(key=key, **group, term=arm, order=len(subset),
                               norm=float(np.linalg.norm(v)),
                               signed_target_projection=float(v @ target) / max(float(np.linalg.norm(target)), 1e-30),
                               projection_on_delta=float(v @ a["delta"]), finite_contribution=fe))
        assert np.max(abs(np.sum(pieces, axis=0) - target)) <= 1e-10
        assert abs(sum(finite_pieces) - finite_target) <= 1e-10
        f, t = entry_data(reference, "forecasts", fs[key]), entry_data(reference, "native", ns[key])
        y = margin(t["T_scores"]) - margin(f["M_scores"])
        for arm in ARMS:
            scalar = float(g[arm] @ a["delta"])
            effects.append(dict(key=key, **group, arm=arm, effect=e[arm],
                                dot_delta=scalar, local_linear_error=abs(e[arm] - scalar),
                                gradient_norm=float(np.linalg.norm(g[arm]))))
            native.append(dict(key=key, **group, arm=arm, target_effect=y, prediction=scalar,
                               error=abs(y - scalar), passes=abs(y - scalar) <= 0.1,
                               target_large=abs(y) > 0.1, predicted_large=abs(scalar) > 0.1,
                               baseline="Original forecast M margin; not the MF primal"))
    return dict(hypotheses=hypotheses, ledger=ledger, effects=effects, native_scalar=native)


def main(run, reference, previous, qualification):
    assert not (run / "audit.json").exists()
    bank, checks, fd, traces, raw = numerical(run, reference, previous, qualification)
    tables = dict(checks=checks, derivatives=fd, intermediate_diagnostics=traces,
                  native_coordinate_diagnostics=raw)
    for name, rows in tables.items():
        table(run / (name + ".csv"), rows)
    failed = [r for r in checks + fd if not r["passes"]]
    audit = dict(status="failed" if failed else "passed", experiment=288, retrospective=True,
                 design_sha256=sha(run / "design.json"), status_sha256=sha(run / "status.json"),
                 required_checks=len(checks) + len(fd), failed_checks=len(failed),
                 intermediate_diagnostic_failures=sum(not r["passes"] for r in traces),
                 original_coordinate_diagnostic_failures=sum(not r["passes"] for r in raw),
                 old_287_failed_gate_preserved=True, general_question_B_solved=False,
                 audit_source_sha256=sha(ROOT / "research/boolean_coupling/audit288.py"))
    if not failed:
        tables.update(scientific(run, reference, bank))
        for name in ("hypotheses", "ledger", "effects", "native_scalar"):
            table(run / (name + ".csv"), tables[name])
        audit["hypotheses"] = {}
        for name in ("SOURCE", "RECEIVING_AND_HEAD", "RECEIVING_BLOCKS", "HEAD", "ADD"):
            rows = [r for r in tables["hypotheses"] if r["hypothesis"] == name]
            layouts = {b: sum(r["passes"] for r in rows if r["block"] == b)
                       for b in sorted({r["block"] for r in rows})}
            count = sum(r["passes"] for r in rows)
            audit["hypotheses"][name] = dict(passes=count, total=len(rows), by_layout=layouts,
                                             population_pass=count >= 24 and all(v >= 6 for v in layouts.values()),
                                             material_passes=sum(r["material_pass"] for r in rows),
                                             material_eligible=sum(r["material_eligible"] for r in rows))
    audit["csv_sha256"] = {n + ".csv": sha(run / (n + ".csv")) for n in tables}
    save_json(run / "audit.json", audit)
    print(audit, flush=True)
    assert not failed, "Required numerical gate failed; scientific tables withheld"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "reference-285", "previous-287", "qualification-287b"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.reference_285.resolve(), a.previous_287.resolve(), a.qualification_287b.resolve())
