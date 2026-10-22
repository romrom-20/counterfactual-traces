"""Standard-library integrity and CSV-summary checks for260 and reader figures."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def near(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def main():
    folder = ROOT/"results/silu-energy-concentration-260"
    for path, digest in read(folder/"manifest.json")["files"].items():
        assert sha(ROOT/path) == digest, path
    audit = read(folder/"audit.json")
    assert audit["status"] == "passed" and audit["retrospective"] and audit["model_calls"] == 0
    assert audit["experiment"] == 260 and len(audit["checks"]) == 192
    assert audit["source_sha256"] == sha(ROOT/"research/boolean_coupling/energy260.py")
    assert audit["protocol_sha256"] == sha(ROOT/"docs/experiments/260-silu-energy-concentration.md")
    assert audit["input_audit_sha256"] == sha(ROOT/"results/receiver-mlp-curvature-258/audit.json")
    assert audit["csv_sha256"] == sha(folder/"energy-concentration.csv")
    assert all(math.isfinite(c["error"]) and c["error"] <= c["bound"] for c in audit["checks"])
    with (folder/"energy-concentration.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48 and len({(r["key"], r["arm"], r["layer"], r["scope"]) for r in rows}) == 48
    assert {r["arm"] for r in rows} == {"C", "S"} and {r["layer"] for r in rows} == {"1", "14", "28"}
    assert len({r["key"] for r in rows}) == 4
    for row in rows:
        numbers = {k:float(v) for k, v in row.items() if k not in ("key", "arm", "layer", "scope", "eligible")}
        assert all(math.isfinite(v) for v in numbers.values())
        total, retained, omitted = (numbers[k] for k in ("individual_energy", "selected_individual_energy", "unselected_individual_energy"))
        gn, vn, un, cross = (numbers[k] for k in ("gap_squared_norm", "selected_squared_norm", "unselected_squared_norm", "cross_energy"))
        assert (row["eligible"] == "True") == (gn >= 1e-16 and total >= 1e-20)
        near(total, retained+omitted)
        near(gn, vn+un+cross)
        near(numbers["selected_fraction"], retained/max(total, 1e-30))
        near(numbers["complete_coherence"], gn/max(total, 1e-30))
        near(numbers["cross_relative"], cross/max(gn, 1e-30))
        near(numbers["remaining_relative"], math.sqrt(un/max(gn, 1e-30)))
    for scope in ("all", "verdict"):
        selected = [r for r in rows if r["scope"] == scope]
        eligible = [r for r in selected if r["eligible"] == "True"]
        summary = audit["summary"][scope]
        assert summary["total"] == len(selected) == 24 and summary["eligible"] == len(eligible)
        assert summary["above_90"] == sum(float(r["selected_fraction"]) >= .9 for r in eligible)
        assert summary["negative_cross"] == sum(float(r["cross_energy"]) < 0 for r in eligible)
        near(summary["minimum_selected_fraction"], min(float(r["selected_fraction"]) for r in eligible))
        for source, target in (("selected_fraction", "median_selected_fraction"), ("complete_coherence", "median_coherence"),
                               ("cross_relative", "median_cross_relative"), ("remaining_relative", "median_remaining_relative")):
            near(median(float(r[source]) for r in eligible), summary[target])
    assert audit["primary_concentration_account"] == (audit["summary"]["all"]["above_90"] >= 18)
    evidence258 = ROOT/"results/receiver-mlp-curvature-258"
    forecast = {e["key"]:e for e in read(evidence258/"forecast-status.json")["cases"]}
    for item in audit["inputs"]:
        original = forecast[item["key"]]
        assert item["witness_sha256"] == original["witness_sha256"]
        assert item["forecast_sha256"] == original["arrays_sha256"]
    figures = ROOT/"docs/figures/reader"
    provenance = read(figures/"provenance.json")
    assert provenance["posthoc"] and provenance["model_calls"] == 0 and provenance["all_cases"]
    assert provenance["source_sha256"] == sha(ROOT/"research/boolean_coupling/reader_figures.py")
    for path, digest in provenance["input_sha256"].items():
        assert sha(ROOT/path) == digest, path
    for name, digest in provenance["figure_sha256"].items():
        assert sha(figures/name) == digest, name
    print("Verified260: all48 retrospective records,192 checks, energy identities and fixed question; reader figures bound to all-case public CSVs. Integrity is not new model replication.")


if __name__ == "__main__":
    main()
