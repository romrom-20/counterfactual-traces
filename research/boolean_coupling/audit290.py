"""Gate290 using saved receiving normalization parents and finite candidate scores."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .audit287 import metrics
from .cache272 import save_json
from .composition280 import digest
from .interface289 import BOUNDS
from .qualify287b import vector_stats
from .readout286 import ROOT, margin
from .receiver290 import ARMS, PRIMARY, SOURCES, norm_records
from .run286 import cpu_head, entry_data
from .run290 import refs


def counts(layers):
    return dict(
        full_F=1,
        first_block=1,
        anchor_suffixes=2,
        anchor_blocks=2 * layers,
        modified_forwards=1,
        modified_blocks=layers,
        modified_heads=1,
        tape_backwards=0,
        triangular_blocks=0,
        triangular_block_VJPs=0,
        triangular_head_VJPs=0,
        native_suffixes=1,
        native_blocks=layers,
        interface_forwards=56,
        interface_blocks=112 * layers,
        interface_heads=56,
        interface_backwards=4,
        blockwise_blocks=8 * layers,
        blockwise_VJPs=4 * layers,
        blockwise_head_VJPs=4,
        encoding_norm_calls=60 * layers,
        reference_forwards=1,
        reference_backwards=1,
        primitive_norm_VJPs=0,
        primitive_attention_VJPs=0,
        anchor_qk_recomputations=2 * layers,
        receiving_norm_calls=60 * layers,
        receiving_MLP_reanchors=30 * layers,
        receiver_norm_probe_VJPs=4 * layers,
    )


def numerical(run, reference, previous, head_path):
    d, st = read(run / "design.json"), read(run / "status.json")
    assert d["experiment"] == 290 and d["retrospective"] and st["status"] == "complete"
    assert d["arms"] == list(ARMS) and d["FD_steps"] == [0.05, 0.025] and d["FD_unit_delta"]
    assert d["primary"] == PRIMARY and d["numerical_bounds"] == BOUNDS
    assert d["factors"] == [
        "receiving_normalization",
        "gated_MLP",
    ]
    assert d["resource_guards_MiB"] == [384, 256, 96]
    assert d["reference_hashes"] == refs(reference, previous, head_path)
    assert set(d["source_hashes"]) == set(SOURCES)
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "sources" / Path(n).name) == h
    assert read(previous / "audit.json")["status"] == "passed"
    assert d["cases"] == read(previous / "design.json")["cases"]
    assert len(d["cases"]) == (4 if d["synthetic"] else 32)
    assert [e["key"] for e in st["cases"]] == [c["key"] for c in d["cases"]]
    assert sum(e["bytes"] for e in st["cases"]) == st["raw_bytes"] <= 96 * 1024**2
    assert sha(head_path / "head.npz") == read(head_path / "head.json")["arrays_sha256"]
    with np.load(head_path / "head.npz") as z:
        head = {n: z[n] for n in z.files}
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ps = {e["key"]: e for e in read(previous / "status.json")["cases"]}
    bank, checks, fd, primitives, raw, trace, costs = {}, [], [], [], [], [], []
    for entry, case in zip(st["cases"], d["cases"], strict=True):
        key = entry["key"]
        a = entry_data(run, "cases", entry)
        f, p = entry_data(reference, "forecasts", fs[key]), entry_data(previous, "cases", ps[key])
        m = read(run / "cases" / (key + ".json"))
        assert m["forecast_entry"] == fs[key] and m["previous_entry"] == ps[key]
        assert (
            m["prefix_unchanged"]
            and m["prefix_seals"] == read(previous / "cases" / (key + ".json"))["prefix_seals"]
        )
        policy = m["first_source_policy"]
        assert policy["activation_calls"] == 1 and policy["False_primal_exact"]
        assert policy["input_gate_sha256"] == digest(f["F_gate"])
        assert policy["output_phi_sha256"] == digest(f["F_phi"])
        assert np.array_equal(a["F_gate"], f["F_gate"]) and np.array_equal(a["F_phi"], f["F_phi"])
        assert all(np.isfinite(v).all() for v in a.values())
        layers = len(a["input_weights"])
        assert m["counts"] == counts(layers)
        costs.append(m["counts"])
        assert np.array_equal(a["unit_delta"], a["delta"] / np.linalg.norm(a["delta"]))
        computed = {}

        def check(name, actual, expected, bound=0.0):
            err = float(np.max(abs(np.asarray(actual, float) - np.asarray(expected, float))))
            computed[name] = dict(
                name=name, error=err, bound=bound, passes=bool(np.isfinite(err) and err <= bound)
            )

        check("ordinary_F_final", a["ordinary_F_final"], p["ordinary_F_final"], 1e-4)
        check("ordinary_F_scores", a["ordinary_F_scores"], p["ordinary_F_scores"], 1e-4)
        check("initial_F", a["initial_F"], p["initial_F"], 1e-4)
        check("initial_E", a["initial_E"], p["initial_E"], 1e-4)
        check("delta_identity", a["delta"], f["C_source"].astype(float) - f["M_source"])
        for anchor in (0, 1):
            check(
                f"anchor{anchor}_final",
                a[f"anchor{anchor}_final"],
                p[f"anchor{anchor}_final"],
                1e-4,
            )
            check(
                f"anchor{anchor}_scores",
                a[f"anchor{anchor}_scores"],
                p[f"anchor{anchor}_scores"],
                1e-4,
            )
        check("reference_gradient", a["reference_gradient"], p["reference_gradient"], 1e-4)
        check("reference_primal_offset", a["reference_zero_offset"], 0.0)
        check("reference_primal_scores", a["reference_scores"], a["anchor1_scores"])
        check(
            "reference_old_finite",
            a["reference_finite_final"],
            p["reference_finite_final"],
            0.002,
        )
        check(
            "reference_old_scores",
            a["reference_finite_scores"],
            p["reference_finite_scores"],
            0.001,
        )
        check(
            "reference_CPU64_head",
            cpu_head(head, a["reference_finite_final"]),
            a["reference_finite_scores"],
            0.005,
        )
        assert np.array_equal(
            a["reference_finite_offset"],
            a["reference_finite_final"].astype(float) - a["anchor1_final"].astype(float),
        )
        for arm in ARMS:
            check(arm + "_zero_offset", a[arm + "_zero_offset"], 0.0)
            check(arm + "_zero_head", a[arm + "_zero_raw_scores"], a["anchor1_scores"])
            check(
                arm + "_boundary_gradient",
                a[arm + "_gradient"],
                a[arm + "_blockwise_gradient"],
                1e-4,
            )
            check(
                arm + "_CPU64_head",
                cpu_head(head, a[arm + "_finite_head_state"]),
                a[arm + "_finite_raw_scores"],
                0.005,
            )
            assert np.array_equal(
                a[arm + "_finite_lifted_scores"], a[arm + "_finite_raw_scores"].astype(float)
            )
            if arm == "00":
                check("baseline_old_gradient", a[arm + "_gradient"], p["111_gradient"], 1e-4)
                check(
                    "baseline_old_final", a[arm + "_finite_offset"], p["111_finite_offset"], 0.002
                )
                check(
                    "baseline_old_scores",
                    a[arm + "_finite_raw_scores"],
                    p["111_finite_raw_scores"],
                    0.001,
                )
        check("allE_gradient", a["11_gradient"], a["reference_gradient"], 1e-4)
        check("allE_finite_offset", a["11_finite_offset"], a["reference_finite_offset"], 0.002)
        check("allE_finite_scores", a["11_finite_raw_scores"], a["reference_finite_scores"], 0.001)
        assert set(m["source_paths"]) == set(ARMS)
        assert all(len(x) == layers for x in m["source_paths"].values())
        check(
            "source_paths_identical",
            float(any(x != m["source_paths"]["00"] for x in m["source_paths"].values())),
            0.0,
        )
        assert len(computed) == len(m["checks"]) == 38
        for row in m["checks"]:
            got = computed[row["name"]]
            assert got == row, (row, got)
            checks.append(dict(key=key, **got))
            if row["name"] in ("reference_old_finite", "baseline_old_final", "allE_finite_offset"):
                raw.append(
                    dict(
                        key=key,
                        name=row["name"],
                        error=row["error"],
                        bound=1e-4,
                        passes=bool(row["error"] <= 1e-4),
                    )
                )
        actual_primitives = norm_records(a, float(a["norm_eps"]))
        assert actual_primitives == m["primitive_checks"] and len(actual_primitives) == 8 * layers
        primitives.extend(dict(key=key, **x) for x in actual_primitives)
        assert len(m["primitive_zero_checks"]) == 0
        for row in m["primitive_zero_checks"]:
            assert row["bound"] == 0.0 and row["passes"] == (row["error"] == 0.0)
            primitives.append(dict(key=key, **row))
        coords = np.random.default_rng(int(key[:16], 16)).choice(len(a["delta"]), 2, replace=False)
        directions = {"unit_delta": a["unit_delta"]}
        for c in coords:
            x = np.zeros_like(a["delta"])
            x[c] = 1.0
            directions["coordinate_" + str(c)] = x
        ix = {(r["arm"], r["direction"], r["step"]): r for r in m["derivatives"]}
        assert len(ix) == len(m["derivatives"]) == 36
        assert set(ix) == set(itertools.product(ARMS, directions, (0.05, 0.025, "halving")))
        for (arm, label, step), r in ix.items():
            ad = float(a[arm + "_gradient"].astype(float) @ directions[label])
            assert r["AD"] == ad
            if step == "halving":
                err = abs(ix[arm, label, 0.05]["FD"] - ix[arm, label, 0.025]["FD"])
                bound = 0.002 + 0.03 * abs(ad)
                assert r["FD"] == ix[arm, label, 0.025]["FD"]
            else:
                estimate = (margin(r["plus_scores"]) - margin(r["minus_scores"])) / (2 * step)
                assert r["FD"] == estimate
                err = abs(ad - estimate)
                bound = 0.001 + 0.02 * abs(ad)
            assert r["error"] == err and r["bound"] == bound
            fd.append(dict(key=key, **r, passes=bool(np.isfinite(err) and err <= bound)))
        assert len(m["trace_diagnostics"]) == 4 * layers
        for row in m["trace_diagnostics"]:
            delta_trace = a[row["arm"] + "_trace_difference"]
            assert delta_trace.shape == (layers, len(a["delta"]))
            assert np.array_equal(
                delta_trace[0],
                a[row["arm"] + "_gradient"].astype(float)
                - a[row["arm"] + "_blockwise_gradient"].astype(float),
            )
            assert row["error"] == float(np.max(abs(delta_trace[row["layer"] - 2])))
            assert row["bound"] == 1e-4 and row["passes"] == (row["error"] <= 1e-4)
            trace.append(dict(key=key, **row))
        bank[key] = a
    assert st["counts"] == {n: sum(c[n] for c in costs) for n in costs[0]}
    return bank, dict(
        checks=checks,
        derivatives=fd,
        primitives=primitives,
        original_coordinates=raw,
        intermediate_diagnostics=trace,
    )


def science(run, reference, bank):
    hypotheses, vectors, native, ledger = [], [], [], []
    fs = {e["key"]: e for e in read(reference / "forecast-status.json")["cases"]}
    ns = {e["key"]: e for e in read(reference / "native-status.json")["cases"]}
    for case in read(run / "design.json")["cases"]:
        key, group = case["key"], case["group"]
        a = bank[key]
        g = {k: a[k + "_gradient"].astype(float) for k in ARMS}
        e = {k: margin(a[k + "_finite_raw_scores"]) - margin(a["anchor1_scores"]) for k in ARMS}
        v = a["reference_gradient"].astype(float) - g["00"]
        target_effect = margin(a["reference_finite_scores"]) - margin(a["anchor1_scores"]) - e["00"]
        target_vector = (a["reference_finite_offset"] - a["00_finite_offset"])[1:]
        f, t = entry_data(reference, "forecasts", fs[key]), entry_data(reference, "native", ns[key])
        y = margin(t["T_scores"]) - margin(f["M_scores"])
        for arm in ARMS:
            p = g[arm] - g["00"]
            met = metrics(p, v)
            pred = e[arm] - e["00"]
            err = abs(pred - target_effect)
            vp = met["norm"] >= 0.001 and met["relative"] <= 0.25 and met["cosine"] >= 0.95
            ep = err <= 0.005 + 0.25 * abs(target_effect)
            hypotheses.append(
                dict(
                    key=key,
                    **group,
                    arm=arm,
                    **met,
                    **vector_stats(p, v),
                    finite_target=target_effect,
                    finite_prediction=pred,
                    finite_error=err,
                    vector_pass=vp,
                    finite_pass=ep,
                    passes=vp and ep,
                    material_eligible=abs(target_effect) >= 0.02,
                    material_pass=vp and ep and abs(target_effect) >= 0.02,
                )
            )
            vector = (a[arm + "_finite_offset"] - a["00_finite_offset"])[1:]
            vm = metrics(vector, target_vector)
            vectors.append(
                dict(
                    key=key,
                    **group,
                    arm=arm,
                    **vm,
                    **vector_stats(vector, target_vector),
                    signed_target_projection=float(np.sum(vector * target_vector))
                    / max(float(np.linalg.norm(target_vector)), 1e-30),
                    passes=vm["norm"] >= 0.001 and vm["relative"] <= 0.25 and vm["cosine"] >= 0.95,
                )
            )
            scalar = float(g[arm] @ a["delta"])
            native.append(
                dict(
                    key=key,
                    **group,
                    arm=arm,
                    prediction=scalar,
                    native_target=y,
                    error=abs(scalar - y),
                    passes=abs(scalar - y) <= 0.1,
                    target_large=abs(y) > 0.1,
                    predicted_large=abs(scalar) > 0.1,
                    finite_effect=e[arm],
                )
            )
        for term in ARMS[1:]:
            subset = [j for j, b in enumerate(term) if b == "1"]
            coeff = np.zeros_like(v)
            fe = 0.0
            for bits in itertools.product((0, 1), repeat=len(subset)):
                c = ["0"] * 2
                for j, b in zip(subset, bits, strict=True):
                    c[j] = str(b)
                c = "".join(c)
                sign = (-1) ** (len(subset) - sum(bits))
                coeff += sign * g[c]
                fe += sign * e[c]
            ledger.append(
                dict(
                    key=key,
                    **group,
                    term=term,
                    norm=float(np.linalg.norm(coeff)),
                    projection_on_delta=float(coeff @ a["delta"]),
                    finite_contribution=fe,
                )
            )
    return dict(
        hypotheses=hypotheses, receiving_vectors=vectors, native_scalar=native, ledger=ledger
    )


def main(run, reference, previous, head):
    assert not (run / "audit.json").exists()
    bank, tables = numerical(run, reference, previous, head)
    for n, r in tables.items():
        table(run / (n + ".csv"), r)
    required = tables["checks"] + tables["derivatives"] + tables["primitives"]
    failed = [r for r in required if not r["passes"]]
    out = dict(
        experiment=290,
        status="failed" if failed else "passed",
        retrospective=True,
        design_sha256=sha(run / "design.json"),
        status_sha256=sha(run / "status.json"),
        audit_source_sha256=sha(ROOT / "research/boolean_coupling/audit290.py"),
        required_checks=len(required),
        failed_checks=len(failed),
        original_coordinate_failures=sum(not r["passes"] for r in tables["original_coordinates"]),
        intermediate_diagnostic_failures=sum(
            not r["passes"] for r in tables["intermediate_diagnostics"]
        ),
        learned_circuit_identified=False,
        general_question_B_solved=False,
    )
    if not failed:
        tables.update(science(run, reference, bank))
        for n in ("hypotheses", "receiving_vectors", "native_scalar", "ledger"):
            table(run / (n + ".csv"), tables[n])
        out["arms"] = {}
        for arm in ARMS:
            rr = [r for r in tables["hypotheses"] if r["arm"] == arm]
            layouts = {
                b: sum(r["passes"] for r in rr if r["block"] == b)
                for b in sorted({r["block"] for r in rr})
            }
            count = sum(r["passes"] for r in rr)
            out["arms"][arm] = dict(
                passes=count,
                total=len(rr),
                layouts=layouts,
                population_pass=count >= 24 and all(n >= 6 for n in layouts.values()),
                material_passes=sum(r["material_pass"] for r in rr),
                material_eligible=sum(r["material_eligible"] for r in rr),
            )
        out["primary_pass"] = {
            name: out["arms"][arm]["population_pass"] for name, arm in PRIMARY["hypotheses"].items()
        }
    out["csv_sha256"] = {n + ".csv": sha(run / (n + ".csv")) for n in tables}
    save_json(run / "audit.json", out)
    print(out, flush=True)
    assert not failed, "Required numerical qualification failed; scientific tables withheld"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "reference-285", "previous-289", "head-287"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    main(a.run.resolve(), a.reference_285.resolve(), a.previous_289.resolve(), a.head_287.resolve())
