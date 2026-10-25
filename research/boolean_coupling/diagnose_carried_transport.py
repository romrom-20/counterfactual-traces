"""Posthoc saved-array donor transport of the P-K source-write increment.

No checkpoint is opened and no model computation or causal installation is run.
The diagnostic uses the qualified array loaders; its donor comparisons are not
new frozen primary tests or independent replications of each CSV row.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read
from .carried283 import ARMS, Observations, restore
from .carried284 import bindings, load_forecast, load_native

ROOT = Path(__file__).resolve().parents[2]
GROUP = ("assignment", "block", "reverse", "schema", "target")
CONDITIONS = ("same_schema_other_literal", "cross_schema_same_literal")
THRESHOLDS = dict(target_norm_min=0.001, relative_error_max=0.25, cosine_min=0.95)


def record(files, label, path, expected=None):
    actual = sha(path)
    assert expected is None or actual == expected, (label, actual, expected)
    assert label not in files or files[label] == actual
    files[label] = actual
    return actual


def bound_audit(run, name, files, label):
    path = run / (name + ".json")
    audit = read(path)
    assert audit["status"] == "passed", path
    record(files, label + "/" + path.name, path)
    for field, filename in (
        ("design_sha256", "design.json"),
        ("status_sha256", "status.json"),
        ("forecast_status_sha256", "forecast-status.json"),
        ("native_status_sha256", "native-status.json"),
        ("native_started_sha256", "native-started.json"),
        ("forecast_seal_sha256", "forecast-seal.json"),
    ):
        if field in audit:
            record(files, label + "/" + filename, run / filename, audit[field])
    return audit


def case_index(design):
    cases = {case["key"]: case for case in design["cases"]}
    assert len(cases) == len(design["cases"]) == 32
    pairs = {}
    for key, case in cases.items():
        group = case["group"]
        assert set(group) == set(GROUP)
        index = tuple(group[n] for n in ("assignment", "block", "reverse", "target"))
        assert group["schema"] in ("generic", "descriptive")
        assert group["schema"] not in pairs.setdefault(index, {})
        pairs[index][group["schema"]] = key
    assert len(pairs) == 16 and all(len(pair) == 2 for pair in pairs.values())
    assert len({case["group"]["block"] for case in cases.values()}) == 2
    return cases


def status_entries(run, filename, cases, files, label):
    status = read(run / filename)
    assert status["status"] == "complete"
    entries = {entry["key"]: entry for entry in status["cases"]}
    assert len(entries) == len(status["cases"]) == 32 and set(entries) == set(cases)
    record(files, label + "/" + filename, run / filename)
    return entries


def array_entry(run, folder, entry, files, label):
    path = run / folder / (entry["key"] + ".npz")
    for p, field in ((path, "arrays_sha256"), (path.with_suffix(".json"), "metadata_sha256")):
        record(files, label + "/" + str(p.relative_to(run)), p, entry[field])
    return path, read(path.with_suffix(".json"))


def vectors(data):
    # Convert BEFORE subtraction: the saved native/candidate writes are Float32.
    p, k, c = (np.asarray(data[arm + "_write"], dtype=np.float64) for arm in ("P", "K", "C1"))
    assert p.shape == k.shape == c.shape and p.ndim == 1
    assert all(np.isfinite(value).all() for value in (p, k, c))
    return dict(forecast=p - k, target=c - k)


def old_bank(run, reference, upstream, files):
    design, status = read(run / "design.json"), read(run / "status.json")
    assert design["experiment"] == 283 and design["retrospective"]
    assert tuple(design["arms"]) == ARMS and status["status"] == "complete"
    audit = bound_audit(run, "audit", files, "283")
    assert audit["experiment"] == 283 and audit["numerical_checks"] == 5408
    record(
        files,
        "scientific/audit283.py",
        ROOT / "research/boolean_coupling/audit283.py",
        audit["source_sha256"],
    )
    for name, expected in design["source_hashes"].items():
        record(files, "scientific/" + name, ROOT / name, expected)
        record(files, "283/sources/" + Path(name).name, run / "sources" / Path(name).name, expected)
    for directory, field, label in (
        (reference, "reference_hashes", "282-reference"),
        (upstream, "upstream_hashes", "281-upstream"),
    ):
        for name, expected in design[field].items():
            record(files, label + "/" + name, directory / name, expected)
        bound_audit(directory, "audit", files, label)
    cases = case_index(design)
    entries = status_entries(run, "status.json", cases, files, "283")
    prior_entries = status_entries(reference, "status.json", cases, files, "282-reference")
    upstream_entries = status_entries(upstream, "status.json", cases, files, "281-upstream")
    assert read(reference / "design.json")["cases"] == design["cases"]
    assert read(upstream / "design.json")["cases"] == design["cases"]
    bank = {}
    for key in cases:
        path, meta = array_entry(run, "cases", entries[key], files, "283")
        array_entry(reference, "cases", prior_entries[key], files, "282-reference")
        array_entry(upstream, "cases", upstream_entries[key], files, "281-upstream")
        assert meta["reference_entry"] == prior_entries[key]
        assert meta["archive_replays_exact"] and meta["own_B_inputs"]
        prior = Observations(reference, upstream, prior_entries[key])
        bank[key] = vectors(restore(path, meta, prior))
    return design, cases, bank, dict(original=audit["raw_checks"])


def fresh_bank(run, files):
    design = bindings(run)
    assert design["experiment"] == 284 and design["globally_pre_native"]
    forecast_audit = bound_audit(run, "forecast-audit", files, "284")
    native_audit = bound_audit(run, "native-audit", files, "284")
    continuation = (
        bound_audit(run, "native-continuation-audit", files, "284")
        if (run / "native-continuation-audit.json").exists()
        else None
    )
    assert forecast_audit["numerical_checks"] == 4672 and native_audit["numerical_checks"] == 1024
    for audit, script in (
        (forecast_audit, "audit284.py"),
        (native_audit, "audit284.py"),
        *(([(continuation, "audit284b.py")]) if continuation else []),
    ):
        record(
            files,
            "scientific/" + script,
            ROOT / "research/boolean_coupling" / script,
            audit["source_sha256"],
        )
    for name, expected in design["source_hashes"].items():
        record(files, "scientific/" + name, ROOT / name, expected)
        record(files, "284/sources/" + Path(name).name, run / "sources" / Path(name).name, expected)
    if continuation:
        qualification = read(run / "native-continuation-design.json")
        for name, expected in qualification["source_hashes"].items():
            record(files, "scientific/" + name, ROOT / name, expected)
            record(
                files,
                "284/native-continuation-sources/" + Path(name).name,
                run / "native-continuation-sources" / Path(name).name,
                expected,
            )
        for field, filename in (
            ("qualification_sha256", "native-continuation-design.json"),
            ("continuation_status_sha256", "native-continuation-status.json"),
            ("original_native_audit_sha256", "native-audit.json"),
        ):
            record(files, "284/" + filename, run / filename, continuation[field])
        assert continuation["all17_replay_arrays_and_metadata_exact"]
        assert continuation["scientific_predictions_and_bounds_unchanged"]
    seal = read(run / "forecast-seal.json")
    for field, filename in (
        ("design_sha256", "design.json"),
        ("forecast_audit_sha256", "forecast-audit.json"),
        ("forecast_status_sha256", "forecast-status.json"),
    ):
        record(files, "284/" + filename, run / filename, seal[field])
    assert seal["ordinary_true"] == seal["full_current_source2_primal"] == 0
    forecast_status = read(run / "forecast-status.json")
    assert forecast_status["ordinary_true"] == forecast_status["full_current_source2_primal"] == 0
    cases = case_index(design)
    forecasts = status_entries(run, "forecast-status.json", cases, files, "284")
    natives = status_entries(run, "native-status.json", cases, files, "284")
    assert seal["cases"] == forecast_status["cases"]
    bank = {}
    for key in cases:
        array_entry(run, "forecasts", forecasts[key], files, "284")
        array_entry(run, "native", natives[key], files, "284")
        forecast, fm = load_forecast(run, forecasts[key])
        native, nm = load_native(run, natives[key], forecast)
        assert fm["ordinary_true"] == fm["full_current_source2_primal"] == 0
        assert nm["forecast_arrays_sha256"] == forecasts[key]["arrays_sha256"]
        assert nm["forecast_metadata_sha256"] == forecasts[key]["metadata_sha256"]
        assert nm["full_control_exact"]
        bank[key] = vectors({**forecast, **native})
    return design, cases, bank, dict(forecast=forecast_audit["raw_checks"])


def donor(cases, key, condition):
    group = cases[key]["group"]
    fields = (
        ("schema", "assignment", "reverse", "target")
        if condition == CONDITIONS[0]
        else ("block", "assignment", "reverse", "target")
    )
    selected = [
        other
        for other, case in cases.items()
        if other != key and all(case["group"][name] == group[name] for name in fields)
    ]
    assert len(selected) == 1, (key, condition, selected)
    other = selected[0]
    varying = "block" if condition == CONDITIONS[0] else "schema"
    assert cases[other]["group"][varying] != group[varying]
    return other


def score(value, target):
    norm, size = float(np.linalg.norm(target)), float(np.linalg.norm(value))
    relative = float(np.linalg.norm(value - target)) / max(norm, 1e-30)
    cosine = float(np.vdot(value, target)) / max(norm * size, 1e-30)
    assert np.isfinite((norm, size, relative, cosine)).all()
    return dict(
        norm=norm,
        prediction_norm=size,
        relative=relative,
        cosine=cosine,
        passes=bool(norm >= 0.001 and relative <= 0.25 and cosine >= 0.95),
    )


def summarized(rows):
    assert len(rows) == 32 and len({row["recipient_key"] for row in rows}) == 32
    return dict(
        cases=32,
        dependent_schema_pairs=16,
        passes=sum(row["passes"] for row in rows),
        target_norm_eligible=sum(row["norm"] >= 0.001 for row in rows),
        median_relative=float(np.median([row["relative"] for row in rows])),
        median_cosine=float(np.median([row["cosine"] for row in rows])),
        worst_relative=max(row["relative"] for row in rows),
        minimum_cosine=min(row["cosine"] for row in rows),
    )


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(experiment, run, reference=None, upstream=None, out=None):
    dest = out or (
        ROOT
        / "results"
        / ("carried-state-response-283" if experiment == 283 else "carried-state-forecast-284")
        / "transport-posthoc"
    )
    assert not dest.exists(), dest
    files = {}
    if experiment == 283:
        assert reference is not None and upstream is not None
        design, cases, bank, raw = old_bank(run, reference, upstream, files)
    else:
        design, cases, bank, raw = fresh_bank(run, files)
    rows, own = [], []
    for key in cases:
        group = cases[key]["group"]
        own_norm = float(np.linalg.norm(bank[key]["forecast"]))
        base = dict(experiment=experiment, recipient_key=key, **group)
        own.append(
            dict(
                **base,
                donor_key=key,
                condition="own_rule_reference",
                magnitude="raw",
                own_forecast_norm=own_norm,
                **score(bank[key]["forecast"], bank[key]["target"]),
            )
        )
        for condition in CONDITIONS:
            other = donor(cases, key, condition)
            assert donor(cases, other, condition) == key
            value = bank[other]["forecast"]
            donor_norm = float(np.linalg.norm(value))
            assert donor_norm > 0 and own_norm > 0
            for magnitude in ("raw", "own_forecast_norm"):
                factor = 1.0 if magnitude == "raw" else own_norm / donor_norm
                rows.append(
                    dict(
                        **base,
                        donor_key=other,
                        donor_schema=cases[other]["group"]["schema"],
                        donor_block=cases[other]["group"]["block"],
                        condition=condition,
                        magnitude=magnitude,
                        own_forecast_norm=own_norm,
                        donor_forecast_norm=donor_norm,
                        rescale_factor=factor,
                        **score(value * factor, bank[key]["target"]),
                    )
                )
    assert len(rows) == 128 and len(own) == 32
    summary = dict(
        experiment=experiment,
        posthoc=True,
        model_forwards=0,
        context_cases=32,
        dependent_schema_pairs=16,
        donor_comparisons=128,
        own_reference_comparisons=32,
        thresholds=THRESHOLDS,
        prediction="float64(P_write) - float64(K_write)",
        actual_target="float64(C1_write) - float64(K_write)",
        norm_match="Donor forecast rescaled to recipient FORECAST norm; no target norm used.",
        literal_blocks=sorted({case["group"]["block"] for case in cases.values()}),
        conditions={
            condition: {
                magnitude: summarized(
                    [
                        row
                        for row in rows
                        if row["condition"] == condition and row["magnitude"] == magnitude
                    ]
                )
                for magnitude in ("raw", "own_forecast_norm")
            }
            for condition in CONDITIONS
        },
        own_rule_reference=summarized(own),
        retained_raw_diagnostics=raw,
        frozen_primary=False,
        causal_installation=False,
        learned_circuit=False,
        general_Question_B_solved=False,
        boundary="Retrospective vector comparisons on existing arrays. Each condition reuses the same 32 recipients in 16 dependent schema pairs; the 128 donor rows are not independent examples. Fresh 284 predictions were sealed before native measurements, but this transport diagnostic was designed afterward. No donor vector was installed into the model; no downstream sufficiency, learned feature, efficiency, or priority claim.",
    )
    dest.mkdir(parents=True)
    table(dest / "donor-transfer.csv", rows)
    table(dest / "own-reference.csv", own)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = dict(
        experiment=experiment,
        posthoc=True,
        model_forwards=0,
        predictions_modified=False,
        frozen_primary=False,
        causal_installation=False,
        script_sha256=sha(Path(__file__)),
        scientific_freeze=design["source_commit"],
        run_name=run.name,
        reference_name=reference.name if reference else None,
        upstream_name=upstream.name if upstream else None,
        input_files_sha256=files,
        output_files_sha256={p.name: sha(p) for p in sorted(dest.iterdir())},
        loaders="carried283.Observations + restore"
        if experiment == 283
        else "carried284.bindings + load_forecast + load_native",
    )
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment": experiment,
                "own_rule_reference": summary["own_rule_reference"],
                "conditions": summary["conditions"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(283, 284), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument(
        "--out", type=Path, help="New output directory; published records are never overwritten."
    )
    args = parser.parse_args()
    main(
        args.experiment,
        args.run.resolve(),
        args.reference.resolve() if args.reference else None,
        args.upstream.resolve() if args.upstream else None,
        args.out.resolve() if args.out else None,
    )
