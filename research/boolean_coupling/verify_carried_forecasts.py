"""Stdlib verification of immutable284 forecasts and qualification before native."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/carried-state-forecast-284"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (DEST / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def boolean(row, name):
    assert row[name] in ("True", "False")
    return row[name] == "True"


def main():
    manifest, design, status, audit, seal = (
        read(DEST / (n + ".json"))
        for n in (
            "forecast-manifest",
            "design",
            "forecast-status",
            "forecast-audit",
            "forecast-seal",
        )
    )
    assert manifest["stage"] == "forecasts-sealed-before-native"
    assert (
        manifest["scientific_freeze"]
        == design["source_commit"]
        == "4ddb047b3015cc0b4d9ccf0a4668c172fe9a2eb3"
    )
    for name, h in manifest["files"].items():
        path = Path(name)
        assert (
            not path.is_absolute()
            and ".." not in path.parts
            and ".context" not in path.parts
            and "drafts" not in path.parts
            and path.suffix != ".npz"
        )
        assert sha(ROOT / path) == h, name
        if path.is_relative_to(DEST.relative_to(ROOT)):
            assert "native" not in path.parts and not path.name.startswith(("native-", "native_"))
    assert design["experiment"] == audit["experiment"] == seal["experiment"] == 284
    assert (
        design["globally_pre_native"]
        and seal["globally_pre_native"]
        and not design["synthetic"]
        and not design["published_literal_repetition"]
    )
    assert audit["status"] == "passed" and status["status"] == "complete"
    assert (
        status["ordinary_true"]
        == status["full_current_source2_primal"]
        == audit["ordinary_true"]
        == audit["full_current_source2_primal"]
        == seal["ordinary_true"]
        == seal["full_current_source2_primal"]
        == 0
    )
    assert audit["source_sha256"] == sha(ROOT / "research/boolean_coupling/audit284.py")
    assert audit["design_sha256"] == seal["design_sha256"] == sha(DEST / "design.json")
    assert (
        audit["forecast_status_sha256"]
        == seal["forecast_status_sha256"]
        == sha(DEST / "forecast-status.json")
    )
    assert (
        seal["forecast_audit_sha256"] == sha(DEST / "forecast-audit.json")
        and seal["cases"] == status["cases"]
    )
    assert (
        seal["requires_public_push_before_native"]
        and seal["source_hashes"] == design["source_hashes"]
    )
    for name, h in design["source_hashes"].items():
        assert sha(ROOT / name) == h
    assert design["unchanged_method_sha256"] == sha(
        ROOT / "research/boolean_coupling/carried283.py"
    )
    assert design["storage"] == dict(
        initial_free_MiB=640, per_case_free_MiB=512, new_array_cap_MiB=96, lossless_aliases=True
    )
    cases = read(ROOT / "research/boolean_coupling/cases284.json")
    assert cases == design["cases"]
    history = read(ROOT / "research/boolean_coupling/historical-input-hashes-284.json")
    assert (
        history["hashes"] == design["historical_input_hashes"]
        and len(set(history["hashes"])) == 2048
    )
    for row in history["provenance"]:
        assert sha(ROOT / "research/boolean_coupling" / row["path"]) == row["sha256"]
    seen, groups = set(), {}
    for c in cases:
        f, t = c["false_ids"], c["true_ids"]
        assert len(f) == len(t)
        assert [i for i, (a, b) in enumerate(zip(f, t, strict=True)) if a != b] == [c["position"]]
        assert f[c["position"]] == 895 and t[c["position"]] == 830
        hs = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in (f, t)
        ]
        assert hs == c["input_hashes"] and not set(hs) & (set(history["hashes"]) | seen)
        seen.update(hs)
        g = c["group"]
        index = tuple(g[n] for n in ("assignment", "block", "reverse", "target"))
        assert g["schema"] not in groups.setdefault(index, set())
        groups[index].add(g["schema"])
    assert (
        len(cases) == 32
        and len(seen) == 64
        and len(groups) == 16
        and all(x == {"generic", "descriptive"} for x in groups.values())
    )
    assert all(
        sum(c["group"]["schema"] == s and c["group"]["reverse"] == v for c in cases) == 8
        for s in ("generic", "descriptive")
        for v in (0, 1)
    )
    inventory = read(DEST / "forecast-array-inventory.json")["arrays"]
    indexed = {r["path"]: r for r in inventory}
    assert (
        len(inventory) == len(status["cases"]) == 32
        and sum(r["bytes"] for r in inventory) == status["raw_bytes"] <= 96 * 1024**2
    )
    arms = ("F", "B", "K", "P", "H", "J", "G0", "G1", "G2", "FZ")
    assert tuple(design["forecast_arms"]) == arms and tuple(design["native_arms"]) == (
        "C1",
        "U",
        "N",
        "T",
    )
    for entry in status["cases"]:
        p = DEST / "forecasts" / (entry["key"] + ".json")
        assert sha(p) == entry["metadata_sha256"]
        meta = read(p)
        assert (
            meta["arrays_sha256"]
            == entry["arrays_sha256"]
            == indexed["forecasts/" + entry["key"] + ".npz"]["sha256"]
        )
        assert meta["ordinary_true"] == meta["full_current_source2_primal"] == 0
        assert (
            meta["own_B_inputs"]
            and meta["zero_exact"]
            and meta["all_prefixes_exact"]
            and meta["staged_not_whole_model_derivative"]
        )
        assert set(meta["arms"]) == set(arms)
        logical = meta["logical_arrays"]
        assert not any(n.startswith(("T_", "C1_", "U_", "N_", "full_")) for n in logical)
        for row in logical.values():
            assert row["dtype"] in ("float32", "float64") and len(row["sha256"]) == 64
        for name, alias in meta["array_aliases"].items():
            assert (
                set(alias) == {"local"}
                and name in logical
                and alias["local"] not in meta["array_aliases"]
            )
            assert logical[name] == logical[alias["local"]]
        for arm, data in meta["arms"].items():
            assert (
                data["prefix_hashes"] == meta["arms"]["F"]["prefix_hashes"]
                and len(data["prefix_hashes"]) == 28
                and data["other_rows_unchanged"]
            )
            assert data["product_delta_installed"] == (
                arm in ("K", "P", "H", "J", "G0", "G1", "G2")
            )
            if arm != "F":
                assert (
                    data["actual_gate_hashes"] == meta["arms"]["F"]["actual_gate_hashes"]
                    and data["primal_hashes"] == meta["arms"]["F"]["primal_hashes"]
                )
                assert data["current_MLP2_finite_coordinates"] == 0
    for name, h in audit["csv_sha256"].items():
        assert sha(DEST / name) == h
    checks, raw, coeff, deriv, heads, scores = (
        rows(n)
        for n in (
            "forecast_checks",
            "forecast_raw",
            "forecast_coefficients",
            "forecast_derivatives",
            "forecast_heads",
            "forecast_scores",
        )
    )
    assert len(checks) == audit["numerical_checks"] and all(
        math.isfinite(float(r["error"])) and float(r["error"]) <= float(r["bound"]) for r in checks
    )
    assert (
        len(coeff) == audit["False_anchors"] == 64
        and audit["False_coefficient_vectors"] == 192
        and len(deriv) == audit["staged_derivative_anchors"] == 32
    )
    for r in (*coeff, *deriv):
        assert float(r["bound"]) == 0.0001 and all(
            math.isfinite(float(v)) and float(v) <= 0.0001
            for k, v in r.items()
            if k not in ("key", "site", "bound")
        )
    assert (
        len(raw) == audit["raw_checks"]["total"] == 224
        and sum(not boolean(r, "passes") for r in raw) == audit["raw_checks"]["failed"]
    )
    for r in raw:
        assert float(r["original_bound"]) == 0.0001 and boolean(r, "passes") == (
            float(r["error"]) <= 0.0001
        )
        assert boolean(r, "required_bridge") == (r["node"] in ("norm", "gate", "phi"))
        assert not boolean(r, "required_bridge") or boolean(r, "passes")
    assert (
        len(heads) == audit["finite_head_maps"] == 320
        and audit["public_corner_maps"] == 32
        and audit["public_down_maps"] == 512
    )
    assert all(float(r["public_error"]) == 0 and float(r["CPU64_error"]) <= 0.005 for r in heads)
    assert len(scores) == 320 and all(
        float(r["margin"]) == float(r["score_true"]) - float(r["score_false"]) for r in scores
    )
    counts = status["counts"]
    assert (
        counts["full_forwards"] == 320
        and counts["finite_C_corner_maps"] == 32
        and counts["local_down_maps"] == 64
        and counts["staged_norm_jvps"] == 32
        and counts["staged_gate_up_matrix_products"] == 64
    )
    assert counts["other_source_sites_finite"] == 26 and counts["all_receiving_operations_finite"]
    notes = read(DEST / "forecast-execution-notes.json")
    assert (
        notes["ordinary_true_before_publication"]
        == notes["full_current_source2_before_publication"]
        == 0
    )
    assert (
        notes["full_unequal_CPU_pipeline_passed_before_freeze"]
        and notes["premature_and_wrong_seal_native_tests_passed"]
    )
    assert not any(
        notes[n]
        for n in (
            "learned_circuit",
            "efficiency_claim",
            "general_Question_B_solved",
            "private_article_included",
        )
    )
    for row in read(DEST / "forecast-log-provenance.json")["logs"]:
        assert sha(DEST / (row["stage"] + ".log")) == row["published_sha256"]
    print(
        "284 immutable forecast seal verified:32 new literals/16 dependent pairs,320 forecasts,zero pre-seal nativeTrue; raw original-bound failures retained",
        audit["raw_checks"]["failed"],
        ". Native outcomes are checked separately.",
    )


if __name__ == "__main__":
    main()
