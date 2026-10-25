"""Verify public 283/qualified 284b records without importing model libraries.

This checks release integrity, retained numerical qualifications, metadata and
all frozen endpoint decisions. It does not rerun the checkpoint or reconstruct
the omitted arrays. A forecast-only or partly measured 284 release must fail.
Run: python -m research.boolean_coupling.verify_carried_results --experiment 283
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("F", "B", "K", "P", "H", "J", "G0", "G1", "G2", "FZ", "C1", "U", "N", "T")
EVAL = ("B", "K", "P", "H", "J", "G0", "G1", "G2", "C1", "U", "N", "T")
NATIVE = ("C1", "U", "N", "T")
FREEZES = {
    283: "463ece70233d28e2f3bc6d2ae6d08e4e62880817",
    284: "4ddb047b3015cc0b4d9ccf0a4668c172fe9a2eb3",
}
CONTINUATION_FREEZE = "8e2e4c3d5352f0490de69598a5e003dfca1ca88b"
FORECAST_COMMIT = "c2439037904cf6d6c9eb4cb91c38301f4f12f6f5"
FORECAST_SEAL = "cb74a9102547c2777c019f20b8215b9688184740bf8faf273c940ffaa2b2fed0"
QUALIFICATION = "284b-unused-MPS-cache-and-exact-partial-replay"
GROUP_FIELDS = ("assignment", "block", "reverse", "schema", "target")
ENDPOINTS = ("native", "local", "effect", "incremental", "incremental_effect")
COMPONENTS = {
    "gate",
    "up",
    "installation_or_downstream_nonadditivity",
    "unpredicted_remainder",
}


class VerificationError(ValueError):
    """The completed public record does not satisfy its frozen definitions."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_string(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def safe_path(name):
    path = Path(name)
    require(
        isinstance(name, str)
        and bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and ".context" not in path.parts
        and "drafts" not in path.parts
        and path.suffix != ".npz"
        and (ROOT / path).resolve().is_relative_to(ROOT.resolve()),
        f"Unsafe public path: {name}",
    )
    return ROOT / path


def bound_file(path, expected):
    require(digest_string(expected), f"Invalid SHA256 for {path}")
    require(sha(path) == expected, f"Hash mismatch: {path}")


def csv_rows(dest, name):
    with (dest / (name + ".csv")).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"Missing CSV header: {name}")
        require(len(set(reader.fieldnames)) == len(reader.fieldnames), f"Duplicate columns: {name}")
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows), name)
    return rows


def boolean(row, name):
    require(row[name] in ("True", "False"), f"Invalid Boolean {name}: {row[name]}")
    return row[name] == "True"


def number(row, name):
    value = float(row[name])
    require(math.isfinite(value), f"Nonfinite {name}: {row}")
    return value


def close(actual, expected, label, tolerance=1e-10):
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label)


def good(row, prefix="", relative=0.25, cosine=0.95):
    norm = number(row, prefix + "norm")
    error = number(row, prefix + "relative")
    alignment = number(row, prefix + "cosine")
    require(norm >= 0 and error >= 0 and abs(alignment) <= 1 + 1e-12, f"Invalid vector: {row}")
    return norm >= 0.001 and error <= relative and alignment >= cosine


def outcome(row, native=False):
    # Evaluate every metric, including metrics in ineligible rows.
    vectors = [
        good(row, scope + "_", 0.05 if native else 0.25, 0.99 if native else 0.95)
        for scope in ("all", "verdict")
    ]
    target = number(row, "target_margin")
    error = number(row, "margin_error")
    require(error >= 0, f"Negative margin error: {row}")
    eligible = native or abs(target) >= 0.02
    require(boolean(row, "margin_eligible") == eligible, f"Changed margin eligibility: {row}")
    return all(vectors) and eligible and error <= (0.1 if native else 0.005 + 0.25 * abs(target))


def case_index(design):
    cases = {case["key"]: case for case in design["cases"]}
    require(len(cases) == len(design["cases"]) == 32, "Expected 32 unique cases")
    groups = defaultdict(dict)
    for key, case in cases.items():
        require(digest_string(key), f"Invalid case key: {key}")
        f, t, position = case["false_ids"], case["true_ids"], case["position"]
        require(len(f) == len(t) and 0 <= position < len(f), key)
        require([i for i, (a, b) in enumerate(zip(f, t, strict=True)) if a != b] == [position], key)
        require(f[position] == 895 and t[position] == 830, f"Boolean edit changed: {key}")
        hashes = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in (f, t)
        ]
        require(hashes == case["input_hashes"], f"Input hashes: {key}")
        group = case["group"]
        require(set(group) == set(GROUP_FIELDS), f"Unexpected group fields: {key}")
        index = tuple(group[n] for n in ("assignment", "block", "reverse", "target"))
        schema = group["schema"]
        require(schema in ("generic", "descriptive") and schema not in groups[index], key)
        groups[index][schema] = key
    require(
        len(groups) == 16 and all(len(pair) == 2 for pair in groups.values()), "Dependent pairs"
    )
    for schema in ("generic", "descriptive"):
        for reverse in (0, 1):
            require(
                sum(
                    c["group"]["schema"] == schema and c["group"]["reverse"] == reverse
                    for c in cases.values()
                )
                == 8,
                "Unbalanced frozen stratum",
            )
    return cases, groups


def grouped_row(row, cases):
    key = row["key"]
    require(key in cases, f"Unknown case {key}")
    require(
        all(row[n] == str(cases[key]["group"][n]) for n in GROUP_FIELDS), f"Group mismatch: {row}"
    )


def indexed_rows(rows, cases, arms=EVAL):
    result = {}
    for row in rows:
        grouped_row(row, cases)
        index = row["key"], row["arm"]
        require(index not in result and row["arm"] in arms, f"Duplicate/unknown case-arm: {index}")
        result[index] = row
    require(set(result) == {(key, arm) for key in cases for arm in arms}, "Missing case-arm rows")
    return result


def verify_decisions(dest, design, audit):
    """Recompute all decisions from public metrics, independently of saved flags."""
    cases, groups = case_index(design)
    tables, decisions = {}, {}
    for name in ENDPOINTS:
        tables[name] = indexed_rows(csv_rows(dest, name), cases)
        decisions[name] = {}
        for index, row in tables[name].items():
            actual = (
                good(row) if name in ("local", "incremental") else outcome(row, name == "native")
            )
            require(boolean(row, "passes") == actual, f"Changed {name} decision: {index}")
            decisions[name][index] = actual
        # Every arm is compared with the same actual target within a case.
        fields = (
            ("norm",)
            if name in ("local", "incremental")
            else ("all_norm", "verdict_norm", "target_margin")
        )
        for key in cases:
            for field in fields:
                require(
                    len({tables[name][key, arm][field] for arm in EVAL}) == 1,
                    f"Target mismatch: {name}/{key}/{field}",
                )

    pairs, pair_decisions = {}, {}
    for row in csv_rows(dest, "paired"):
        generic, descriptive = row["generic_key"], row["descriptive_key"]
        require(generic in cases and descriptive in cases, "Unknown paired case")
        group = cases[generic]["group"]
        index = tuple(group[n] for n in ("assignment", "block", "reverse", "target"))
        require(
            groups[index] == {"generic": generic, "descriptive": descriptive},
            "Mismatched schema pair",
        )
        require(row["pair_key"] == "|".join(map(str, index)), "Changed pair key")
        arm, pair_key = row["arm"], (row["pair_key"], row["arm"])
        require(arm in EVAL and pair_key not in pairs, "Duplicate/unknown paired arm")
        passed = good(row)
        require(boolean(row, "passes") == passed, f"Paired decision: {pair_key}")
        pairs[pair_key] = row
        pair_decisions[generic, arm] = pair_decisions[descriptive, arm] = passed
    require(
        len(pairs) == 16 * len(EVAL) and len(pair_decisions) == 32 * len(EVAL), "Incomplete pairs"
    )

    joint = indexed_rows(csv_rows(dest, "joint"), cases)
    joint_decisions = {}
    for index, row in joint.items():
        expected = {
            "native_pass": decisions["native"][index],
            "local_pass": decisions["local"][index],
            "effect_pass": decisions["effect"][index],
            "incremental_local_pass": decisions["incremental"][index],
            "paired_local_pass": pair_decisions[index],
        }
        require(
            all(boolean(row, name) == value for name, value in expected.items()),
            f"Joint components: {index}",
        )
        passed = all(expected.values())
        require(boolean(row, "passes") == passed, f"Joint decision: {index}")
        joint_decisions[index] = passed

    removal = indexed_rows(csv_rows(dest, "removal"), cases, ("N",))
    removal_passes = 0
    for index, row in removal.items():
        full, local = outcome(row), good(row, "local_")
        passed = full and local
        require(boolean(row, "passes") == passed, f"Removal decision: {index}")
        removal_passes += passed
        target = tables["incremental_effect"][index[0], "U"]
        for field in ("all_norm", "verdict_norm", "target_margin"):
            close(
                number(row, field),
                number(target, field),
                f"Removal target changed: {index}/{field}",
            )
        close(
            number(row, "local_norm"),
            number(tables["incremental"][index[0], "U"], "norm"),
            f"Removal local target: {index}",
        )

    counts = {
        name: {arm: sum(passed for (key, a), passed in values.items() if a == arm) for arm in EVAL}
        for name, values in {**decisions, "joint": joint_decisions}.items()
    }
    counts["paired"] = {
        arm: sum(boolean(row, "passes") for row in pairs.values() if row["arm"] == arm)
        for arm in EVAL
    }
    for name, values in counts.items():
        require(audit[name + "_counts"] == values, f"Audit count mismatch: {name}")
    strata = {
        f"{schema}/{reverse}": sum(
            joint_decisions[key, "P"]
            for key, case in cases.items()
            if case["group"]["schema"] == schema and case["group"]["reverse"] == reverse
        )
        for schema in ("generic", "descriptive")
        for reverse in (0, 1)
    }
    primary = counts["joint"]["P"] >= 24 and min(strata.values()) >= 6
    require(
        audit["P_strata"] == strata and audit["primary_passes"] is primary,
        "Primary/strata mismatch",
    )
    eligible = sum(
        boolean(tables["incremental_effect"][key, "P"], "margin_eligible") for key in cases
    )
    require(audit["increment_margin_eligible"] == eligible, "Increment margin eligibility count")
    require(audit["removal_passes"] == removal_passes, "Removal count mismatch")
    # Calibration is conditional C1, not native T: T may fail its C1-K local target.
    for name in ENDPOINTS:
        for key in cases:
            left, right = tables[name][key, "C1"], tables[name][key, "U"]
            require(
                all(left[n] == right[n] for n in left if n not in (*GROUP_FIELDS, "key", "arm")),
                f"Full product calibration differs from C1: {name}/{key}",
            )
    require(counts["native"]["T"] == 32, "Native T self-comparison failed")
    if design["experiment"] == 284:
        verify_sealed_score_margins(dest, cases, tables)
    verify_ledger(dest, cases)
    return {
        "counts": counts,
        "P_strata": strata,
        "primary_passes": primary,
        "increment_margin_eligible": eligible,
        "removal_passes": removal_passes,
    }


def verify_sealed_score_margins(dest, cases, tables):
    """Cross-check fresh endpoint margin errors against pre-native scored forecasts."""
    scores = indexed_rows(csv_rows(dest, "forecast_scores"), cases, ARMS[:10])
    for row in scores.values():
        close(
            number(row, "margin"),
            number(row, "score_true") - number(row, "score_false"),
            "Sealed score-to-margin conversion",
        )
    for key in cases:
        base = number(scores[key, "B"], "margin")
        carry = number(scores[key, "K"], "margin")
        native_target = number(tables["native"][key, "P"], "target_margin")
        effect_target = number(tables["effect"][key, "P"], "target_margin")
        incremental_target = number(tables["incremental_effect"][key, "P"], "target_margin")
        close(incremental_target, effect_target + base - carry, "Full/incremental margin targets")
        c1_margin = effect_target + base
        for arm in EVAL:
            if arm in ("C1", "U"):
                predicted = c1_margin
            elif arm == "T":
                predicted = native_target
            elif (key, arm) in scores:
                predicted = number(scores[key, arm], "margin")
            else:
                continue  # N has no pre-native scored forecast.
            for name, origin, target in (
                ("native", 0, native_target),
                ("effect", base, effect_target),
                ("incremental_effect", carry, incremental_target),
            ):
                close(
                    number(tables[name][key, arm], "margin_error"),
                    abs(predicted - origin - target),
                    f"Sealed score/endpoint mismatch: {name}/{key}/{arm}",
                )


def verify_ledger(dest, cases):
    ledger = defaultdict(dict)
    for row in csv_rows(dest, "ledger"):
        grouped_row(row, cases)
        index = row["key"], row["quantity"]
        component = row["component"]
        require(
            row["quantity"] in ("write", "receiving", "verdict")
            and component in COMPONENTS
            and component not in ledger[index],
            "Duplicate/unknown ledger entry",
        )
        good(row)  # Validate finite vector metrics without treating them as successes.
        target, size = number(row, "norm"), number(row, "component_norm")
        projection = number(row, "signed_projection")
        require(size >= 0, "Negative component norm")
        if target > 1e-15:
            ratio = size / target
            close(
                number(row, "relative") ** 2,
                ratio**2 + 1 - 2 * projection,
                "Signed ledger geometry",
                1e-8,
            )
            if ratio > 1e-15:
                close(number(row, "cosine"), projection / ratio, "Signed ledger cosine", 1e-8)
        ledger[index][component] = row
    require(
        set(ledger) == {(key, q) for key in cases for q in ("write", "receiving", "verdict")},
        "Incomplete signed ledger",
    )
    for index, parts in ledger.items():
        require(
            set(parts) == COMPONENTS and len({r["norm"] for r in parts.values()}) == 1,
            f"Ledger parts: {index}",
        )
        if number(next(iter(parts.values())), "norm") > 1e-15:
            close(
                sum(number(row, "signed_projection") for row in parts.values()),
                1.0,
                f"Signed closure: {index}",
                1e-8,
            )


def verify_manifest(dest, experiment):
    manifest = read(dest / "manifest.json")
    require(
        manifest["stage"]
        == ("completed-old-retrospective" if experiment == 283 else "completed-fresh-literal"),
        "Release is not complete",
    )
    require(manifest["scientific_freeze"] == FREEZES[experiment], "Wrong scientific freeze")
    paths = manifest["files"]
    require(paths, "Empty manifest")
    for name, expected in paths.items():
        bound_file(safe_path(name), expected)
    expected_files = {
        str(p.relative_to(ROOT))
        for p in dest.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    require(expected_files <= set(paths), "Unsealed release files")
    required = {
        str(Path(__file__).resolve().relative_to(ROOT)),
        "research/boolean_coupling/package_carried.py",
        "research/boolean_coupling/describe_carried.py",
        f"docs/analysis/{experiment}-carried-state-response.md",
    }
    require(required <= set(paths), "Unsealed presentation/verifier code")
    require(not list(dest.rglob("*.npz")), "Raw model arrays must remain omitted")
    return manifest


def verify_sources(design):
    for name, expected in design["source_hashes"].items():
        bound_file(safe_path(name), expected)


def verify_csv_hashes(dest, audit, required):
    require(set(audit["csv_sha256"]) == set(required), "Changed numerical/endpoint CSV inventory")
    for name, expected in audit["csv_sha256"].items():
        require(Path(name).name == name and name.endswith(".csv"), "Invalid CSV path")
        bound_file(dest / name, expected)


def logical_arrays(meta, reference):
    logical, aliases = meta["logical_arrays"], meta["array_aliases"]
    require(logical and set(aliases) <= set(logical), "Invalid logical array inventory")
    for name, record in logical.items():
        require(
            record["dtype"] in ("float32", "float64")
            and isinstance(record["shape"], list)
            and all(type(n) is int and n >= 0 for n in record["shape"])
            and digest_string(record["sha256"]),
            f"Invalid logical array: {name}",
        )
    for name, alias in aliases.items():
        require(
            len(alias) == 1 and next(iter(alias)) in ("local", "reference"),
            f"Invalid alias: {name}",
        )
        if "local" in alias:
            target = alias["local"]
            require(target in logical and target not in aliases, f"Alias chain: {name}")
            expected = logical[target]
        else:
            require(alias["reference"] in reference, f"Unknown reference alias: {name}")
            expected = reference[alias["reference"]]
        require(logical[name] == expected, f"Non-lossless alias: {name}")


def verify_inventory(dest, filename, status, folder):
    record = read(dest / filename)
    require(record["omitted"] is True, "Raw arrays unexpectedly included")
    entries = status["cases"]
    indexed = {entry["key"]: entry for entry in entries}
    inventory = {row["path"]: row for row in record["arrays"]}
    require(
        len(indexed) == len(entries) == len(inventory) == len(record["arrays"]) == 32,
        "Raw array inventory count",
    )
    require(
        set(inventory) == {f"{folder}/{key}.npz" for key in indexed}, "Raw array inventory paths"
    )
    require(
        status["raw_bytes"] == sum(row["bytes"] for row in inventory.values()), "Raw byte total"
    )
    metadata = {}
    for key, entry in indexed.items():
        row = inventory[f"{folder}/{key}.npz"]
        require(
            row["sha256"] == entry["arrays_sha256"]
            and digest_string(row["sha256"])
            and row["bytes"] == entry["bytes"] > 0,
            f"Raw inventory hash/size: {key}",
        )
        path = dest / folder / (key + ".json")
        bound_file(path, entry["metadata_sha256"])
        meta = read(path)
        require(
            meta["key"] == key and meta["arrays_sha256"] == entry["arrays_sha256"],
            f"Case record: {key}",
        )
        metadata[key] = meta
    require(
        {p.stem for p in (dest / folder).glob("*.json")} == set(indexed), "Missing/extra metadata"
    )
    return metadata


def verify_numeric(dest, audit, cases, prefix="", head_arms=ARMS, raw_failed=None):
    checks = csv_rows(dest, prefix + "checks")
    require(len(checks) == audit["numerical_checks"], "Numerical check count")
    by_case = defaultdict(Counter)
    for row in checks:
        require(row["key"] in cases, "Unknown numerical case")
        error, bound = number(row, "error"), number(row, "bound")
        require(0 <= error <= bound, f"Failed retained numerical check: {row}")
        by_case[row["key"]][row["name"]] += 1
    require(
        set(by_case) == set(cases)
        and len({tuple(sorted(v.items())) for v in by_case.values()}) == 1,
        "Unequal numerical check coverage",
    )
    heads = csv_rows(dest, prefix + "heads")
    head_keys = {(r["key"], r["arm"]) for r in heads}
    require(
        len(head_keys) == len(heads) == audit["finite_head_maps"] == len(cases) * len(head_arms)
        and head_keys == {(key, arm) for key in cases for arm in head_arms},
        "Finite head coverage",
    )
    require(
        all(
            number(r, "public_error") == 0 and 0 <= number(r, "CPU64_error") <= 0.005 for r in heads
        ),
        "Finite head check failed",
    )
    if raw_failed is None:
        return
    coefficients, derivatives, raw = (
        csv_rows(dest, prefix + n) for n in ("coefficients", "derivatives", "raw")
    )
    require(
        len(coefficients) == audit["False_anchors"] == 64
        and audit["False_coefficient_vectors"] == 192,
        "False coefficient coverage",
    )
    require(
        {(r["key"], r["site"]) for r in coefficients}
        == {(key, str(site)) for key in cases for site in (1, 2)},
        "Coefficient anchor identity",
    )
    require(
        len(derivatives) == audit["staged_derivative_anchors"] == 32
        and {r["key"] for r in derivatives} == set(cases),
        "Staged derivative coverage",
    )
    for row in (*coefficients, *derivatives):
        require(number(row, "bound") == 0.0001, "Changed derivative/coefficient tolerance")
        require(
            all(
                0 <= number(row, name) <= 0.0001
                for name in row
                if name not in ("key", "site", "bound")
            ),
            f"Failed derivative/coefficient check: {row}",
        )
    nodes = {"norm", "gate", "up", "phi", "finite_product", "finite_write", "cubic_write"}
    require(
        len(raw) == audit["raw_checks"]["total"] == 224
        and {(r["key"], r["node"]) for r in raw}
        == {(key, node) for key in cases for node in nodes},
        "Raw bridge coverage",
    )
    failed = []
    for row in raw:
        error = number(row, "error")
        require(error >= 0 and number(row, "original_bound") == 0.0001, "Changed raw bound")
        require(boolean(row, "passes") == (error <= 0.0001), "Raw decision mismatch")
        required = row["node"] in ("norm", "gate", "phi")
        require(
            boolean(row, "required_bridge") == required
            and (not required or boolean(row, "passes")),
            "Failed required bridge",
        )
        if not boolean(row, "passes"):
            failed.append(row)
    require(
        len(failed) == audit["raw_checks"]["failed"] == raw_failed
        and all(r["node"] == "up" for r in failed),
        "Original raw UP failures not preserved",
    )
    require(audit["raw_checks"]["original_bound"] == 0.0001, "Raw audit bound changed")


def verify_figures(dest, experiment, audit_name):
    folder = ROOT / f"docs/figures/carried-response/{experiment}"
    provenance = read(folder / "provenance.json")
    require(
        provenance["experiment"] == experiment
        and provenance["posthoc"] is True
        and provenance["model_forwards"] == 0
        and provenance["predictions_modified"] is False,
        "Figure scope changed",
    )
    bound_file(ROOT / "research/boolean_coupling/describe_carried.py", provenance["script_sha256"])
    bound_file(dest / audit_name, provenance["audit_sha256"])
    require(
        set(provenance["csv_sha256"])
        == {n + ".csv" for n in ("incremental", "paired", "native", "joint", "ledger")},
        "Figure inputs changed",
    )
    for name, expected in provenance["csv_sha256"].items():
        bound_file(dest / name, expected)
    require(
        set(provenance["figure_sha256"]) == {"carried-response.png", "carried-response.svg"},
        "Figure inventory",
    )
    for name, expected in provenance["figure_sha256"].items():
        bound_file(folder / name, expected)


def verify_logs(dest, name):
    record = read(dest / name)
    stages = set()
    for row in record["logs"]:
        stage = row["stage"]
        require(
            stage not in stages and Path(stage).name == stage and digest_string(row["raw_sha256"]),
            "Log provenance",
        )
        stages.add(stage)
        bound_file(dest / (stage + ".log"), row["published_sha256"])
    require(stages, "Missing execution logs")
    return {row["stage"]: row for row in record["logs"]}


def verify_old(dest):
    design, status, audit = (
        read(dest / (name + ".json")) for name in ("design", "status", "audit")
    )
    require(
        design["experiment"] == audit["experiment"] == 283
        and design["retrospective"] is True
        and audit["retrospective"] is True,
        "Old study scope",
    )
    require(
        design["source_commit"] == FREEZES[283]
        and status["status"] == "complete"
        and audit["status"] == "passed",
        "Old study incomplete",
    )
    require(tuple(design["arms"]) == ARMS, "Changed old arm inventory")
    verify_sources(design)
    bound_file(ROOT / "research/boolean_coupling/audit283.py", audit["source_sha256"])
    bound_file(dest / "design.json", audit["design_sha256"])
    bound_file(dest / "status.json", audit["status_sha256"])
    cases, _ = case_index(design)
    references = ROOT / "results/source2-origin-282"
    for folder, field in (
        (references, "reference_hashes"),
        (ROOT / "results/source2-products-281d", "upstream_hashes"),
    ):
        for name, expected in design[field].items():
            require(Path(name).name == name, "Invalid reference path")
            bound_file(folder / name, expected)
    archived = {entry["key"]: entry for entry in read(references / "status.json")["cases"]}
    metadata = verify_inventory(dest, "raw-array-inventory.json", status, "cases")
    require(set(metadata) == set(cases) and status["raw_bytes"] <= 64 * 1024**2, "Old raw budget")
    expected_flags = (
        "own_B_inputs",
        "candidates_before_full_calibration",
        "archive_replays_exact",
        "staged_not_whole_model_derivative",
        "zero_exact",
        "full_control_exact",
    )
    for key, meta in metadata.items():
        require(
            meta["reference_entry"] == archived[key]
            and all(meta[name] is True for name in expected_flags),
            f"Old case qualifications: {key}",
        )
        reference_path = references / "cases" / (key + ".json")
        bound_file(reference_path, archived[key]["metadata_sha256"])
        logical_arrays(meta, read(reference_path)["logical_arrays"])
        require(set(meta["arms"]) == set(ARMS), f"Old case arms: {key}")
        base = meta["arms"]["F"]
        for arm, data in meta["arms"].items():
            require(
                data["prefix_hashes"] == base["prefix_hashes"]
                and len(data["prefix_hashes"]) == 28
                and all(digest_string(h) for h in data["prefix_hashes"])
                and data["other_rows_unchanged"] is True,
                f"Prefix invariance: {key}/{arm}",
            )
            require(
                data["product_delta_installed"]
                is (arm in ("K", "P", "H", "J", "G0", "G1", "G2", "U", "N")),
                f"Installed products: {key}/{arm}",
            )
            if arm not in ("F", "C1", "T"):
                require(
                    data["actual_gate_hashes"] == base["actual_gate_hashes"]
                    and data["primal_hashes"] == base["primal_hashes"]
                    and data["current_MLP2_finite_coordinates"] == 0,
                    f"Unexpected current source2 primal: {key}/{arm}",
                )
    require(
        status["counts"]
        == {
            "full_forwards": 448,
            "finite_C_corner_maps": 32,
            "local_down_maps": 64,
            "staged_norm_jvps": 32,
            "staged_gate_up_matrix_products": 64,
            "other_source_sites_finite": 26,
            "all_receiving_operations_finite": True,
        },
        "Old prediction budget changed",
    )
    required = {
        n + ".csv"
        for n in (
            *ENDPOINTS,
            "checks",
            "raw",
            "coefficients",
            "derivatives",
            "heads",
            "removal",
            "paired",
            "joint",
            "ledger",
        )
    }
    verify_csv_hashes(dest, audit, required)
    verify_numeric(dest, audit, cases, raw_failed=9)
    require(
        audit["derivative_bound"] == 0.0001
        and audit["public_corner_maps"] == 32
        and audit["public_down_maps"] == 480,
        "Old numerical scope",
    )
    notes = read(dest / "execution-notes.json")
    require(
        notes["initial_unsealed_CPU_assertion_failure"] is True
        and digest_string(notes["failed_unsealed_preflight_sha256"])
        and notes["source_or_bound_changes_after_freeze"] is False,
        "Initial failed fixture not preserved",
    )
    for name in (
        "independent_research_team",
        "fresh_confirmation",
        "learned_circuit",
        "efficiency_claim",
        "general_Question_B_solved",
        "private_article_included",
    ):
        require(notes[name] is False, f"Unsupported old release claim: {name}")
    verify_logs(dest, "log-provenance.json")
    return verify_decisions(dest, design, audit)


def verify_continuation(dest, design, native_status, native_audit):
    qualification, status, audit, old = (
        read(dest / (n + ".json"))
        for n in (
            "native-continuation-design",
            "native-continuation-status",
            "native-continuation-audit",
            "retained-native-stop/inventory",
        )
    )
    require(
        qualification["qualification"] == audit["qualification"] == QUALIFICATION
        and qualification["source_commit"] == CONTINUATION_FREEZE,
        "Unqualified native continuation",
    )
    sources = {
        "research/boolean_coupling/continue284b.py",
        "research/boolean_coupling/audit284b.py",
        "research/boolean_coupling/preflight284b.py",
        "docs/experiments/284b-native-cache-continuation.md",
    }
    require(
        set(qualification["source_hashes"]) == sources, "Continuation scientific source inventory"
    )
    verify_sources(qualification)
    for field, filename in (
        ("design_sha256", "design.json"),
        ("forecast_seal_sha256", "forecast-seal.json"),
        ("original_failure_sha256", "retained-native-stop/failure-native.json"),
        ("original_native_started_sha256", "native-started.json"),
        ("original_partial_inventory_sha256", "retained-native-stop/inventory.json"),
    ):
        bound_file(dest / filename, qualification[field])
    bound_file(
        dest / "retained-native-stop/native-started.json",
        qualification["original_native_started_sha256"],
    )
    require(
        all(
            qualification[name] is False
            for name in (
                "predictions_changed",
                "scientific_thresholds_changed",
                "numerical_validation_scope_changed",
            )
        ),
        "Continuation changed the science",
    )
    require(
        qualification["per_case_reserve_MiB"] == 512
        and qualification["combined_raw_cap_MiB"] == 96
        and qualification["replay_every_completed_case_before_new_native"] is True
        and qualification["previously_measured_native_cases"] == 17,
        "Continuation guard/replay policy",
    )
    require(
        old["status"] == "incomplete_native_storage_guard"
        and old["full_forwards"] == 68
        and old["no_orphan_case18"] is True
        and old["guard_expression"] == "assert shutil.disk_usage(run).free >= 512 * 1024**2",
        "Original storage stop erased",
    )
    require(
        read(dest / "retained-native-stop/failure-native.json")["error"] == "AssertionError()"
        and digest_string(old["failure_log_sha256"]),
        "Original failure not preserved",
    )
    bound_file(ROOT / "research/boolean_coupling/run284.py", old["source_sha256"])
    require(
        status["status"] == "complete"
        and status["all_replay_arrays_and_metadata_exact"] is True
        and len(status["replays"]) == len(old["cases"]) == 17,
        "Exact partial replay incomplete",
    )
    require(
        [r["key"] for r in old["cases"]] == [c["key"] for c in design["cases"][:17]],
        "Original partial case order changed",
    )
    native_entries = {entry["key"]: entry for entry in native_status["cases"]}
    replay_record = read(dest / "replay-array-inventory.json")
    replay_inventory = {row["path"]: row for row in replay_record["arrays"]}
    require(
        replay_record["omitted"] is True
        and len(replay_inventory) == len(replay_record["arrays"]) == 17
        and set(replay_inventory) == {f"replay/{prior['key']}.npz" for prior in old["cases"]},
        "Missing/extra replay-array inventory",
    )
    require(
        {p.stem for p in (dest / "replay").glob("*.json")}
        == {prior["key"] for prior in old["cases"]},
        "Missing/extra replay metadata",
    )
    for replay, prior in zip(status["replays"], old["cases"], strict=True):
        require(
            replay["exact"] is True
            and all(
                replay[name] == prior[name] == native_entries[prior["key"]][name]
                for name in ("key", "arrays_sha256", "metadata_sha256")
            ),
            "Replay/new native/retained hashes disagree",
        )
        bound_file(dest / "native" / (prior["key"] + ".json"), replay["metadata_sha256"])
        bound_file(dest / "replay" / (prior["key"] + ".json"), replay["metadata_sha256"])
        omitted = replay_inventory[f"replay/{prior['key']}.npz"]
        require(
            omitted["sha256"] == prior["arrays_sha256"]
            and omitted["bytes"] == prior["bytes"] == native_entries[prior["key"]]["bytes"],
            "Replay-array inventory disagrees with original/native arrays",
        )
    expected_counts = {
        "original_partial_full_forwards": 68,
        "replay_full_forwards": 68,
        "new_unmeasured_full_forwards": 60,
        "total_native_producer_full_forwards": 196,
        "unique_native_cases": 32,
        "actual_native_True_producer_forwards": 49,
    }
    require(
        all(status[n] == v for n, v in expected_counts.items()),
        "Actual continuation forward counts",
    )
    bound_file(dest / "native-cache-events.csv", status["cache_events_sha256"])
    events = csv_rows(dest, "native-cache-events")
    sequence = [("initial", "loaded")]
    for case in design["cases"]:
        sequence += [(case["key"], arm) for arm in ("before_case", *NATIVE)]
    require(
        [(r["key"], r["arm"]) for r in events] == sequence,
        "Missing/reordered cache scheduling evidence",
    )
    for row in events:
        require(
            row["device"] in ("cpu", "mps")
            and all(
                number(row, name) >= 0 and number(row, name).is_integer()
                for name in ("driver_before", "driver_after", "live_before", "live_after")
            ),
            "Invalid cache diagnostics",
        )
    require(audit["status"] == "passed", "Continuation native audit incomplete")
    for field, path in (
        ("source_sha256", ROOT / "research/boolean_coupling/audit284b.py"),
        ("qualification_sha256", dest / "native-continuation-design.json"),
        ("continuation_status_sha256", dest / "native-continuation-status.json"),
        ("original_native_audit_sha256", dest / "native-audit.json"),
        ("original_native_audit_source_sha256", ROOT / "research/boolean_coupling/audit284.py"),
    ):
        bound_file(path, audit[field])
    require(
        all(
            audit[n] is True
            for n in (
                "scientific_predictions_and_bounds_unchanged",
                "all17_replay_arrays_and_metadata_exact",
                "fresh_literals_in_dependent_templates",
            )
        )
        and audit["private_article_included"] is False,
        "Continuation attribution scope",
    )
    require(native_audit["ordinary_True_replay_forwards"] == 32, "Independent native replay count")
    forecast_bytes = read(dest / "forecast-status.json")["raw_bytes"]
    require(
        forecast_bytes + native_status["raw_bytes"] + sum(r["bytes"] for r in old["cases"])
        <= 96 * 1024**2,
        "Combined raw-array cap exceeded",
    )
    verify_failed_continuations(dest, qualification)


def verify_failed_continuations(dest, successful_qualification):
    """Bind both stopped284b attempts; neither generated another native outcome."""
    folder = dest / "retained-continuation-stops"
    record = read(folder / "inventory.json")
    expected = {
        "carried-state-forecast-284b": "284b-native",
        "carried-state-forecast-284b-retry": "284b-retry-native",
    }
    attempts = {row["attempt"]: row for row in record["attempts"]}
    require(
        len(record["attempts"]) == len(attempts) == 2
        and set(attempts) == set(expected)
        and {p.name for p in folder.iterdir() if p.is_dir()} == set(expected),
        "The two zero-forward continuation failures were not retained",
    )
    logs = verify_logs(dest, "native-log-provenance.json")
    for name, entry in attempts.items():
        require(
            entry["replay_forwards"] == entry["new_native_forwards"] == 0
            and entry["retained_native_cases"] == 17
            and entry["log_stage"] == expected[name],
            "Stopped continuation forward counts changed",
        )
        attempt = folder / name
        require(
            {p.name for p in attempt.iterdir()}
            == {"failure-native-continuation.json", "native-continuation-design.json"},
            "Unexpected stopped-attempt files",
        )
        bound_file(attempt / "failure-native-continuation.json", entry["failure_sha256"])
        failure = read(attempt / "failure-native-continuation.json")
        qualification = read(attempt / "native-continuation-design.json")
        require(
            set(qualification) == set(successful_qualification)
            and all(
                qualification[field] == value
                for field, value in successful_qualification.items()
                if field != "began"
            ),
            "Stopped attempt used another scientific qualification",
        )
        require(
            failure["error"] == "AssertionError()"
            and datetime.fromisoformat(qualification["began"])
            <= datetime.fromisoformat(failure["time"])
            <= datetime.fromisoformat(successful_qualification["began"]),
            "Stopped continuation chronology/failure",
        )
        stage = entry["log_stage"]
        require(stage in logs, "Missing stopped-attempt log provenance")
        lines = (dest / (stage + ".log")).read_text().splitlines()
        require(
            any(line.startswith("Traceback") for line in lines)
            and any(line.startswith("AssertionError") for line in lines)
            and not any(line.startswith(("284b replayed", "284b new native")) for line in lines),
            "Stopped-attempt log contradicts zero completed replay/new native forwards",
        )
    require(
        {"284-native", "284b-retry2-native", "284b-retry2-audit"} <= set(logs),
        "Missing original stop/completed continuation/audit logs",
    )
    original_lines = (dest / "284-native.log").read_text().splitlines()
    completed_lines = (dest / "284b-retry2-native.log").read_text().splitlines()
    require(
        [line for line in original_lines if line.startswith("284 native")]
        == [f"284 native {i} /32" for i in range(1, 18)],
        "Original producer log does not retain exactly17 completed cases",
    )
    require(
        [line for line in completed_lines if line.startswith("284b")]
        == [f"284b replayed {i} /32" for i in range(1, 18)]
        + [f"284b new native {i} /32" for i in range(18, 33)],
        "Continuation log does not replay17 before measuring remaining15",
    )


def verify_fresh(dest):
    # Reuse the separately published, dependency-free pre-native checker without
    # changing its manifest or importing any scientific/model implementation.
    forecast_manifest_before = sha(dest / "forecast-manifest.json")
    path = ROOT / "research/boolean_coupling/verify_carried_forecasts.py"
    spec = importlib.util.spec_from_file_location("sealed_carried_forecasts", path)
    require(spec is not None and spec.loader is not None, "Forecast verifier unavailable")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.main()
    require(
        sha(dest / "forecast-manifest.json") == forecast_manifest_before,
        "Immutable forecast manifest changed",
    )
    design, status, audit, start = (
        read(dest / (n + ".json"))
        for n in ("design", "native-status", "native-audit", "native-started")
    )
    require(
        status["status"] == "complete"
        and audit["status"] == "passed"
        and audit["experiment"] == design["experiment"] == 284,
        "Fresh native measurements/audit incomplete",
    )
    require(
        design["source_commit"] == FREEZES[284]
        and status["counts"] == {"full_forwards": 128, "ordinary_true": 32},
        "Fresh unique native counts",
    )
    require(
        sha(dest / "forecast-seal.json")
        == FORECAST_SEAL
        == start["forecast_seal_sha256"]
        == start["receipt"]["forecast_seal_sha256"],
        "Forecast/public/native seal mismatch",
    )
    receipt = start["receipt"]
    require(
        receipt["public_forecast_commit"] == FORECAST_COMMIT
        and receipt["public_seal_path"] == "results/carried-state-forecast-284/forecast-seal.json"
        and receipt["remote_head_verified"] is True,
        "No pre-native public forecast receipt",
    )
    require(
        datetime.fromisoformat(read(dest / "forecast-seal.json")["ended"])
        <= datetime.fromisoformat(start["began"]),
        "Native measurement preceded seal",
    )
    require(
        start["native_before_forecast_seal"]
        == start["full_current_source2_before_forecast_seal"]
        == 0,
        "Native leakage before seal",
    )
    for field, name in (
        ("forecast_seal_sha256", "forecast-seal"),
        ("native_status_sha256", "native-status"),
        ("native_started_sha256", "native-started"),
    ):
        bound_file(dest / (name + ".json"), audit[field])
    bound_file(ROOT / "research/boolean_coupling/audit284.py", audit["source_sha256"])
    cases, _ = case_index(design)
    metadata = verify_inventory(dest, "native-array-inventory.json", status, "native")
    require(set(metadata) == set(cases), "Fresh native case inventory")
    forecasts = {entry["key"]: entry for entry in read(dest / "forecast-status.json")["cases"]}
    for key, meta in metadata.items():
        fm = read(dest / "forecasts" / (key + ".json"))
        require(
            meta["forecast_arrays_sha256"] == forecasts[key]["arrays_sha256"]
            and meta["forecast_metadata_sha256"] == forecasts[key]["metadata_sha256"]
            and meta["full_control_exact"] is True,
            f"Native-to-forecast binding: {key}",
        )
        logical_arrays(meta, fm["logical_arrays"])
        require(set(meta["arms"]) == set(NATIVE), f"Native arm inventory: {key}")
        for arm, data in meta["arms"].items():
            require(
                data["prefix_hashes"] == fm["arms"]["F"]["prefix_hashes"]
                and data["other_rows_unchanged"] is True
                and data["product_delta_installed"] is (arm in ("U", "N")),
                f"Native intervention policy: {key}/{arm}",
            )
    required = {
        n + ".csv"
        for n in (
            *ENDPOINTS,
            "native_checks",
            "native_heads",
            "removal",
            "paired",
            "joint",
            "ledger",
        )
    }
    verify_csv_hashes(dest, audit, required)
    verify_numeric(dest, audit, cases, "native_", NATIVE)
    # Retain the original11/224 diagnostics, even when the endpoint succeeds.
    fa = read(dest / "forecast-audit.json")
    verify_numeric(dest, fa, cases, "forecast_", ARMS[:10], 11)
    verify_continuation(dest, design, status, audit)
    notes = read(dest / "native-execution-notes.json")
    require(
        notes["forecast_full_forwards"] == 320
        and notes["native_control_full_forwards"] == 128
        and notes["actual_native_producer_full_forwards_including_original_and_replay"] == 196
        and notes["actual_native_True_producer_forwards_including_replay"] == 49
        and notes["original_partial_native_cases"] == 17
        and notes["resumed_cases_after_all17_exact_replays"] == 15
        and notes["retained_zero_forward_continuation_stops"] == 2
        and notes["native_runtime_policy_changed_under_separate_284b_qualification"] is True
        and notes["independent_ordinary_True_audit_replay_forwards"] == 32
        and notes["forecasts_and_qualification_public_before_native"] is True,
        "Fresh execution budget",
    )
    require(
        notes["forecast_manifest_sha256"] == forecast_manifest_before,
        "Original forecast manifest replaced",
    )
    require(
        all(
            notes[n] is False
            for n in (
                "source_or_bound_changes_after_forecasts",
                "learned_circuit",
                "efficiency_claim",
                "general_Question_B_solved",
                "private_article_included",
            )
        )
        and notes["templates_dependent"] is True,
        "Fresh release overclaims",
    )
    verify_logs(dest, "native-log-provenance.json")
    result = verify_decisions(dest, design, audit)
    require(
        sha(dest / "forecast-manifest.json") == forecast_manifest_before,
        "Forecast record modified during verification",
    )
    return result


def verify_transport_inputs(dest, experiment, provenance, design):
    """Check provenance via public metadata; omitted arrays remain hash-only."""
    sources = dict(design["source_hashes"])
    if experiment == 284:
        sources.update(read(dest / "native-continuation-design.json")["source_hashes"])
    by_basename = defaultdict(list)
    for name in sources:
        by_basename[Path(name).name].append(name)
    folders = {
        str(experiment): dest,
        "282-reference": ROOT / "results/source2-origin-282",
        "281-upstream": ROOT / "results/source2-products-281d",
    }
    statuses = {}
    for label, directory in folders.items():
        for folder, filename in (
            ("cases", "status.json"),
            ("forecasts", "forecast-status.json"),
            ("native", "native-status.json"),
        ):
            path = directory / filename
            if path.exists():
                statuses[label, folder] = {entry["key"]: entry for entry in read(path)["cases"]}
    require(provenance["input_files_sha256"], "Missing transport input provenance")
    for label, expected in provenance["input_files_sha256"].items():
        path = Path(label)
        require(
            not path.is_absolute() and ".." not in path.parts and digest_string(expected),
            "Invalid transport provenance path/hash",
        )
        first, *rest = path.parts
        if first == "scientific":
            name = str(Path(*rest))
            if len(rest) == 1:
                name = "research/boolean_coupling/" + name
            bound_file(safe_path(name), expected)
        elif first in folders:
            require(rest, "Missing transport provenance filename")
            if rest[0] in ("sources", "native-continuation-sources"):
                require(
                    len(rest) == 2 and len(by_basename[rest[1]]) == 1, "Ambiguous transport source"
                )
                name = by_basename[rest[1]][0]
                require(sources[name] == expected, "Transport scientific source seal")
                bound_file(safe_path(name), expected)
            elif path.suffix == ".npz":
                require(
                    len(rest) == 2 and (first, rest[0]) in statuses,
                    "Unknown transport raw-array reference",
                )
                key = Path(rest[1]).stem
                require(
                    key in statuses[first, rest[0]]
                    and statuses[first, rest[0]][key]["arrays_sha256"] == expected,
                    "Transport omitted-array seal",
                )
            else:
                bound_file(folders[first] / Path(*rest), expected)
        else:
            raise VerificationError(f"Unknown transport provenance namespace: {label}")


def verify_transport(dest, experiment):
    """Recompute the posthoc donor decisions and summaries from all public rows."""
    folder = dest / "transport-posthoc"
    provenance, summary, design = (
        read(path)
        for path in (folder / "provenance.json", folder / "summary.json", dest / "design.json")
    )
    require(
        provenance["experiment"] == summary["experiment"] == design["experiment"] == experiment
        and provenance["scientific_freeze"] == design["source_commit"],
        "Transport experiment/freeze mismatch",
    )
    for record in (provenance, summary):
        require(
            record["posthoc"] is True
            and record["model_forwards"] == 0
            and record["frozen_primary"] is False
            and record["causal_installation"] is False,
            "Transport diagnostic scope changed",
        )
    require(
        provenance["predictions_modified"] is False
        and summary["learned_circuit"] is False
        and summary["general_Question_B_solved"] is False,
        "Transport diagnostic overclaims",
    )
    bound_file(
        ROOT / "research/boolean_coupling/diagnose_carried_transport.py",
        provenance["script_sha256"],
    )
    outputs = provenance["output_files_sha256"]
    require(
        set(outputs) == {"donor-transfer.csv", "own-reference.csv", "summary.json"},
        "Transport output inventory",
    )
    for name, expected in outputs.items():
        bound_file(folder / name, expected)
    verify_transport_inputs(dest, experiment, provenance, design)
    cases, _ = case_index(design)
    require(
        summary["thresholds"]
        == {"target_norm_min": 0.001, "relative_error_max": 0.25, "cosine_min": 0.95},
        "Changed transport decision thresholds",
    )
    require(
        summary["context_cases"] == summary["own_reference_comparisons"] == 32
        and summary["dependent_schema_pairs"] == 16
        and summary["donor_comparisons"] == 128,
        "Transport reused sample accounting",
    )
    require(
        summary["literal_blocks"] == sorted({c["group"]["block"] for c in cases.values()})
        and len(summary["literal_blocks"]) == 2,
        "Transport literal groups",
    )
    require(
        summary["prediction"] == "float64(P_write) - float64(K_write)"
        and summary["actual_target"] == "float64(C1_write) - float64(K_write)",
        "Transport prediction/target changed",
    )
    conditions = ("same_schema_other_literal", "cross_schema_same_literal")
    magnitudes = ("raw", "own_forecast_norm")
    require(
        set(summary["conditions"]) == set(conditions)
        and all(set(summary["conditions"][name]) == set(magnitudes) for name in conditions),
        "Transport condition inventory",
    )
    own, donors = {}, {}
    incremental = indexed_rows(csv_rows(dest, "incremental"), cases)

    def validate(row):
        recipient = row["recipient_key"]
        require(
            recipient in cases
            and row["experiment"] == str(experiment)
            and all(row[name] == str(cases[recipient]["group"][name]) for name in GROUP_FIELDS),
            "Transport recipient group",
        )
        passed = good(row)
        require(boolean(row, "passes") == passed, "Transport endpoint decision")
        target, predicted = number(row, "norm"), number(row, "prediction_norm")
        own_size = number(row, "own_forecast_norm")
        require(target > 0 and predicted > 0 and own_size > 0, "Degenerate transport vector")
        ratio = predicted / target
        close(
            number(row, "relative") ** 2,
            1 + ratio**2 - 2 * ratio * number(row, "cosine"),
            "Transport vector geometry",
            1e-8,
        )

    for row in csv_rows(folder, "own-reference"):
        validate(row)
        key = row["recipient_key"]
        require(
            key not in own
            and row["donor_key"] == key
            and row["condition"] == "own_rule_reference"
            and row["magnitude"] == "raw",
            "Transport own-reference identity",
        )
        close(number(row, "prediction_norm"), number(row, "own_forecast_norm"), "Own forecast norm")
        for name in ("norm", "relative", "cosine"):
            close(
                number(row, name),
                number(incremental[key, "P"], name),
                "Own reference vs primary incremental metrics",
            )
        require(
            boolean(row, "passes") == boolean(incremental[key, "P"], "passes"),
            "Own reference vs incremental decision",
        )
        own[key] = row
    require(set(own) == set(cases), "Transport own-reference coverage")
    for row in csv_rows(folder, "donor-transfer"):
        validate(row)
        key, donor, condition, magnitude = (
            row[name] for name in ("recipient_key", "donor_key", "condition", "magnitude")
        )
        require(condition in conditions and magnitude in magnitudes, "Unknown transport condition")
        fields = (
            ("schema", "assignment", "reverse", "target")
            if condition == conditions[0]
            else ("block", "assignment", "reverse", "target")
        )
        selected = [
            other
            for other, case in cases.items()
            if other != key
            and all(case["group"][name] == cases[key]["group"][name] for name in fields)
        ]
        require(selected == [donor], "Transport donor is not unique OTHER matched context")
        changing = "block" if condition == conditions[0] else "schema"
        require(
            cases[donor]["group"][changing] != cases[key]["group"][changing]
            and row["donor_schema"] == cases[donor]["group"]["schema"]
            and row["donor_block"] == cases[donor]["group"]["block"],
            "Transport donor contrast",
        )
        index = key, condition, magnitude
        require(index not in donors, "Duplicate transport recipient-condition")
        own_size, donor_size = (
            number(own[key], "own_forecast_norm"),
            number(own[donor], "own_forecast_norm"),
        )
        close(number(row, "own_forecast_norm"), own_size, "Transport recipient forecast norm")
        close(number(row, "donor_forecast_norm"), donor_size, "Transport donor forecast norm")
        factor = 1.0 if magnitude == "raw" else own_size / donor_size
        close(number(row, "rescale_factor"), factor, "Transport uses forecast-only norm matching")
        close(
            number(row, "prediction_norm"),
            donor_size * factor,
            "Transport installed comparison magnitude",
        )
        close(number(row, "norm"), number(own[key], "norm"), "Transport actual target norm")
        donors[index] = row
    require(
        set(donors)
        == {
            (key, condition, magnitude)
            for key in cases
            for condition in conditions
            for magnitude in magnitudes
        },
        "Transport comparisons must reuse32 recipients in each condition",
    )
    for key in cases:
        for condition in conditions:
            close(
                number(donors[key, condition, "raw"], "cosine"),
                number(donors[key, condition, "own_forecast_norm"], "cosine"),
                "Positive norm matching changed direction",
            )

    def summarized(rows):
        return {
            "cases": 32,
            "dependent_schema_pairs": 16,
            "passes": sum(boolean(row, "passes") for row in rows),
            "target_norm_eligible": sum(number(row, "norm") >= 0.001 for row in rows),
            "median_relative": median(number(row, "relative") for row in rows),
            "median_cosine": median(number(row, "cosine") for row in rows),
            "worst_relative": max(number(row, "relative") for row in rows),
            "minimum_cosine": min(number(row, "cosine") for row in rows),
        }

    def compare(saved, actual):
        require(set(saved) == set(actual), "Transport summary fields")
        for name, value in actual.items():
            if name in ("cases", "dependent_schema_pairs", "passes", "target_norm_eligible"):
                require(saved[name] == value, f"Transport summary count: {name}")
            else:
                close(float(saved[name]), value, f"Transport summary statistic: {name}")

    compare(summary["own_rule_reference"], summarized(list(own.values())))
    for condition in conditions:
        for magnitude in magnitudes:
            compare(
                summary["conditions"][condition][magnitude],
                summarized([donors[key, condition, magnitude] for key in cases]),
            )
    audit_name, raw_label = (
        ("audit.json", "original") if experiment == 283 else ("forecast-audit.json", "forecast")
    )
    require(
        summary["retained_raw_diagnostics"] == {raw_label: read(dest / audit_name)["raw_checks"]},
        "Transport discarded raw numerical caveats",
    )


def main(experiment):
    dest = ROOT / (
        "results/carried-state-response-283"
        if experiment == 283
        else "results/carried-state-forecast-284"
    )
    verify_manifest(dest, experiment)
    result = verify_old(dest) if experiment == 283 else verify_fresh(dest)
    verify_transport(dest, experiment)
    verify_figures(dest, experiment, "audit.json" if experiment == 283 else "native-audit.json")
    print(
        json.dumps(
            {
                "status": "passed",
                "experiment": experiment,
                "scope": "Sealed public records and frozen decisions; no model replication or learned-circuit claim.",
                **result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(283, 284), required=True)
    main(parser.parse_args().experiment)
