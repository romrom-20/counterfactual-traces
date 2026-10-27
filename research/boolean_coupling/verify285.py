"""Dependency-free verification of the public 285 forecast or complete release.

Checks hashes, retained numerical records, information budget, chronology and
all endpoint decisions. It neither loads omitted arrays nor reruns a checkpoint.
Synthetic fixtures cannot satisfy either production command.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path("results/native-structure-285")
ARMS = ("F", "C", "L", "D", "M", "Z", "FZ")
EVAL = ("C", "L", "D", "M", "Z", "T")
GROUP = ("assignment", "block", "reverse", "schema", "target")
HASH = re.compile(r"[0-9a-f]{64}\Z")


class VerificationError(ValueError):
    """A published record does not meet its declared definition."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {path.name}/{key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            VerificationError(f"Nonfinite JSON value: {value}")
        ),
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(row, field):
    value = float(row[field])
    require(math.isfinite(value), f"Nonfinite {field}")
    return value


def boolean(row, field):
    value = row[field]
    require(value in ("True", "False"), f"Invalid CSV Boolean: {field}/{value}")
    return value == "True"


def close(left, right, label):
    require(math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-10), label)


def rows(dest, name):
    with (dest / (name + ".csv")).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
            f"Duplicate or absent CSV columns: {name}",
        )
        result = list(reader)
    require(
        result and all(None not in row and None not in row.values() for row in result),
        f"Malformed or empty CSV: {name}",
    )
    return result


def time(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Timestamp lacks timezone")
    return parsed


def relative(root, name):
    p = Path(name)
    require(
        not p.is_absolute()
        and ".." not in p.parts
        and not any(part.startswith(".") for part in p.parts),
        f"Unsafe publication path: {name}",
    )
    require("drafts" not in p.parts and p.suffix != ".npz", f"Private/array path: {name}")
    require((root / p).resolve().is_relative_to(root.resolve()), "Path leaves publication root")
    return root / p


def declared_sources(root, module="structure285"):
    """Read tuple declarations as AST, without importing scientific dependencies."""
    tree = ast.parse((root / "research/boolean_coupling" / (module + ".py")).read_text())
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.name == "SOURCES":
                    aliases[name.asname or name.name] = node.module

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            result = []
            for child in node.elts:
                if isinstance(child, ast.Starred):
                    result.extend(evaluate(child.value))
                else:
                    result.append(evaluate(child))
            return result
        if isinstance(node, ast.Name) and node.id in aliases:
            return declared_sources(root, aliases[node.id])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dict"
            and node.func.attr == "fromkeys"
            and len(node.args) == 1
            and not node.keywords
        ):
            return list(dict.fromkeys(evaluate(node.args[0])))
        raise VerificationError(f"Unsupported scientific SOURCES declaration: {module}")

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets
        ):
            return evaluate(node.value)
    raise VerificationError(f"Missing scientific SOURCES: {module}")


def verify_manifest(root, dest, phase, sources):
    path = dest / (phase + "-manifest.json")
    manifest = read(path)
    require(manifest["experiment"] == 285 and manifest["phase"] == phase, "Manifest identity")
    require(manifest["scientific_sources"] == sources, "Manifest scientific source map")
    files = manifest["files"]
    for name, expected in files.items():
        p = relative(root, name)
        require(HASH.fullmatch(expected) and sha(p) == expected, f"Manifest hash: {name}")
    required = {str(p.relative_to(root)) for p in dest.rglob("*") if p.is_file() and p != path}
    required.update(sources)
    require(required <= files.keys(), f"Manifest missing files: {sorted(required - files.keys())}")
    if phase == "forecast":
        require(set(files) == required, "Forecast manifest includes later delivery/presentation")
    else:
        require(
            {
                "research/boolean_coupling/publish285.py",
                "research/boolean_coupling/verify285.py",
                "docs/analysis/285-native-structure.md",
            }
            <= files.keys(),
            "Final deliverables absent",
        )
        require(any(p.startswith("docs/figures/") for p in files), "Final figure absent")
    return manifest


def verify_cases(root, design, synthetic_test=False):
    cases = design["cases"]
    require(len(cases) == 32 and len({c["key"] for c in cases}) == 32, "32 unique cases")
    history = design["historical_input_hashes"]
    if not synthetic_test:
        catalog = read(root / "research/boolean_coupling/history285.json")
        require(
            history == catalog["hashes"] and len(set(history)) == len(history) == 2112,
            "Historical 2112-input catalog",
        )
        require(
            cases == read(root / "research/boolean_coupling/cases285.json"), "Frozen cases differ"
        )
        for item in catalog["provenance"]:
            require(
                sha(root / "research/boolean_coupling" / item["path"]) == item["sha256"],
                f"Historical catalog source hash: {item['path']}",
            )
    require(all(HASH.fullmatch(h) for h in history), "Historical hash format")
    seen, groups = set(), {}
    for case in cases:
        require(HASH.fullmatch(case["key"]), "Case key format")
        f, t, start = case["false_ids"], case["true_ids"], case["position"]
        require(len(f) == len(t) and 0 < start < len(f) - 1, "Case geometry")
        require(all(type(i) is int and i >= 0 for i in (*f, *t)), "Input token type")
        require(
            [i for i, pair in enumerate(zip(f, t)) if pair[0] != pair[1]] == [start]
            and f[start] == 895
            and t[start] == 830,
            "Sole False/True edit",
        )
        hashes = [
            hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            for ids in (f, t)
        ]
        require(
            hashes == case["input_hashes"] and not set(hashes) & (seen | set(history)),
            "Fresh input hashes",
        )
        seen.update(hashes)
        g = case["group"]
        require(set(g) == set(GROUP) and g["assignment"] == 0, "Case group fields")
        combination = (g["schema"], g["reverse"], g["target"])
        bucket = groups.setdefault(g["block"], set())
        require(combination not in bucket, "Repeated layout combination")
        bucket.add(combination)
    expected = {(s, r, t) for s in ("generic", "descriptive") for r in (0, 1) for t in (0, 1)}
    require(
        len(seen) == 64 and len(groups) == 4 and all(g == expected for g in groups.values()),
        "Four layouts crossed with eight cases",
    )
    return {c["key"]: c for c in cases}


def verify_inventory(dest, folder, status, cases):
    entries = status["cases"]
    require([e["key"] for e in entries] == list(cases), "Inventory case order")
    inventory = read(dest / "omitted-array-inventory.json")
    require(
        inventory["experiment"] == 285 and inventory["omitted"] is True, "Omitted-array identity"
    )
    expected = [
        dict(path=f"{folder}/{e['key']}.npz", sha256=e["arrays_sha256"], bytes=e["bytes"])
        for e in entries
    ]
    require(inventory["arrays"] == expected, "Omitted-array hashes/sizes/status binding")
    require(
        all(
            HASH.fullmatch(e["sha256"]) and type(e["bytes"]) is int and e["bytes"] > 0
            for e in expected
        ),
        "Array hash/size format",
    )
    require(
        status["raw_bytes"] == inventory["raw_bytes"] == sum(e["bytes"] for e in expected)
        and status["raw_bytes"] <= 96 * 1024**2,
        "Array byte count/cap",
    )
    require(
        {p.name for p in (dest / folder).iterdir()} == {k + ".json" for k in cases},
        "Exactly 32 metadata files, no arrays",
    )
    metadata = {}
    for entry in entries:
        path = dest / folder / (entry["key"] + ".json")
        require(sha(path) == entry["metadata_sha256"], "Case metadata hash")
        meta = read(path)
        require(
            meta["key"] == entry["key"] and meta["arrays_sha256"] == entry["arrays_sha256"],
            "Case metadata array binding",
        )
        metadata[entry["key"]] = meta
    return metadata


def indexed(data, cases, arms, label, groups=True):
    require(len(data) == len(cases) * len(arms), f"{label}: row count")
    result = {}
    for row in data:
        index = (row["key"], row["arm"])
        require(
            index not in result and index[0] in cases and index[1] in arms,
            f"{label}: unknown or duplicate row",
        )
        if groups:
            require(
                all(str(cases[index[0]]["group"][f]) == row[f] for f in GROUP),
                f"{label}: group mismatch",
            )
        result[index] = row
    require(set(result) == {(k, a) for k in cases for a in arms}, f"{label}: coverage")
    return result


def expected_checks(native=False):
    if native:
        return {
            **{f"prepared_native_{n}": 0 for n in ("before", "mid", "gate", "up", "z")},
            **{f"native_public_{n}": 0 for n in ("z", "gate", "up", "phi", "raw_write", "write")},
            "native_source_addition": 0,
            **{f"ordinary_True_{n}": 0 for n in ("source", "final", "scores")},
            **{f"public_head_{a}": 0 for a in (*ARMS, "T")},
            **{f"CPU64_head_{a}": 0.005 for a in (*ARMS, "T")},
        }
    names = {
        "independent_bank": 0,
        "observed_False_SiLU": 0,
        "CPU64_False_SiLU": 0.0001,
        "independent_C_phi": 0.0001,
        "independent_L_phi": 0.0001,
    }
    errors = [f"d{i}_{suffix}" for suffix in ("AD", "FD_0.0001", "FD_1e-05") for i in (1, 2, 3)]
    names.update({"False_coefficient_" + n: 0.0001 for n in errors})
    for a in ("F", "C"):
        names.update(
            {f"known_{a}_{n}": 0 for n in ("before", "norm", "q", "K", "V", "full_k", "full_v")}
        )
        names.update({f"attention_{a}_{n}": 0.0001 for n in ("probability", "PV", "write", "mid")})
        names["saved_mid_" + a] = 0
        names.update({f"public_{a}_{n}": 0 for n in ("z", "gate", "up")})
        names["CPU64_norm_" + a] = 0.0001
    for a in ARMS:
        names.update({f"same_incoming_{a}_{n}": 0 for n in ("before", "mid", "gate", "up", "z")})
        names.update(
            {
                n + a: 0
                for n in (
                    "same_installed_phi_",
                    "public_raw_write_",
                    "installed_write_",
                    "source_addition_",
                    "public_head_",
                )
            }
        )
        names["CPU64_head_" + a] = 0.005
    names.update(
        {
            "False_zero_" + n: 0
            for n in (
                "before",
                "mid",
                "gate",
                "up",
                "phi",
                "raw_write",
                "write",
                "source",
                "final",
                "scores",
                "z",
            )
        }
    )
    return names


def verify_numerical(dest, prefix, audit, cases):
    data = rows(dest, prefix + "_checks")
    expected = expected_checks(prefix == "native")
    require(len(data) == len(cases) * len(expected) == audit["numerical_checks"], "Numerical count")
    seen = set()
    for row in data:
        index = (row["key"], row["name"])
        require(
            index not in seen and index[0] in cases and index[1] in expected,
            "Numerical row name/key",
        )
        seen.add(index)
        error, bound = number(row, "error"), number(row, "bound")
        require(
            bound == expected[index[1]] and 0 <= error <= bound, "Required numerical check failed"
        )
    heads = indexed(
        rows(dest, prefix + "_heads"),
        cases,
        (*ARMS, "T") if prefix == "native" else ARMS,
        "heads",
        groups=False,
    )
    require(len(heads) == audit["finite_head_maps"], "Head count")
    numeric = {(r["key"], r["name"]): r for r in data}
    for (key, arm), row in heads.items():
        for kind, field, bound in (("public", "public_error", 0), ("CPU64", "CPU64_error", 0.005)):
            value = number(row, field)
            require(0 <= value <= bound, "Head qualification failed")
            close(
                value,
                number(numeric[key, kind + "_head_" + arm], "error"),
                "Head/check disagreement",
            )


def verify_sources(root, design):
    sources = design["source_hashes"]
    require(set(sources) == set(declared_sources(root)), "Complete frozen scientific SOURCES")
    for name, expected in sources.items():
        require(
            HASH.fullmatch(expected) and sha(relative(root, name)) == expected,
            f"Frozen scientific source: {name}",
        )
    require(re.fullmatch(r"[0-9a-f]{40}", design["source_commit"]), "Scientific commit format")
    return sources


def verify_reference(root, config):
    """Bind fixed donors to public historical forecast records, without arrays."""
    reference = root / "results/source-activation-forecast-279"
    for name, expected in config["reference_hashes"].items():
        require(sha(reference / name) == expected, "Historical forecast reference hash")
    old, status, audit, seal = (
        read(reference / name)
        for name in (
            "design.json",
            "forecast-status.json",
            "forecast-audit.json",
            "forecast-seal.json",
        )
    )
    require(
        old["experiment"] == 279
        and status["status"] == "complete"
        and audit["status"] == "passed"
        and status["ordinary_true"] == audit["ordinary_true"] == seal["ordinary_true"] == 0,
        "Historical donor forecast-only status",
    )
    require(
        seal["cases"] == status["cases"]
        and seal["design_sha256"] == audit["design_sha256"] == sha(reference / "design.json")
        and seal["status_sha256"]
        == audit["status_sha256"]
        == sha(reference / "forecast-status.json")
        and seal["audit_sha256"] == sha(reference / "forecast-audit.json"),
        "Historical seal bindings",
    )
    entries, cases = ({e["key"]: e for e in status["cases"]}, {c["key"]: c for c in old["cases"]})
    require(len(entries) == len(cases) == 32, "Historical case coverage")
    for item in config["groups"]:
        blocks = set()
        for donor in item["donors"]:
            key = donor["key"]
            require(
                entries[key] == donor
                and all(
                    cases[key]["group"][n] == item["group"][n]
                    for n in ("schema", "reverse", "target", "assignment")
                ),
                "Fixed historical donor match",
            )
            blocks.add(cases[key]["group"]["block"])
            path = reference / "forecasts" / (key + ".json")
            require(sha(path) == donor["metadata_sha256"], "Historical donor metadata hash")
            meta = read(path)
            require(
                meta["ordinary_true"] == 0 and meta["arrays_sha256"] == donor["arrays_sha256"],
                "Historical donor omitted-array binding",
            )
        require(len(blocks) == 2, "Two old literal blocks per donor mean")


def verify_forecast(root=ROOT, destination=DESTINATION, synthetic_test=False, manifest=True):
    dest = root / destination / "pre-native"
    design, status, audit, seal, config, runtime = (
        read(dest / n)
        for n in (
            "design.json",
            "forecast-status.json",
            "forecast-audit.json",
            "forecast-seal.json",
            "bank-config.json",
            "runtime.json",
        )
    )
    require(
        design["experiment"] == 285 and design["globally_pre_native"] is True, "Forecast design"
    )
    require(
        type(design["synthetic"]) is bool and (not design["synthetic"] or synthetic_test),
        "Synthetic fixture cannot satisfy a production release",
    )
    require(tuple(design["arms"]) == ARMS, "Seven-arm declared generation order")
    sources = verify_sources(root, design)
    cases = verify_cases(root, design, synthetic_test)
    require(status["status"] == "complete" and audit["status"] == "passed", "Forecast status/audit")
    require(
        status["ordinary_true"] == audit["ordinary_true"] == seal["ordinary_true"] == 0,
        "No native True before seal",
    )
    require(
        audit["full_current_first_source_SiLU"] == seal["full_current_first_source_SiLU"] == 0,
        "No full current first-source activation",
    )
    require(
        seal["experiment"] == audit["experiment"] == 285 and seal["globally_pre_native"] is True,
        "Forecast audit/seal identity",
    )
    for filename, field in (
        ("design.json", "design_sha256"),
        ("forecast-status.json", "forecast_status_sha256"),
        ("bank-config.json", "bank_config_sha256"),
    ):
        require(
            seal[field] == audit[field] == sha(dest / filename), f"Forecast binding: {filename}"
        )
    require(design["bank_config_sha256"] == sha(dest / "bank-config.json"), "Design bank binding")
    require(seal["forecast_audit_sha256"] == sha(dest / "forecast-audit.json"), "Sealed audit")
    require(
        audit["source_sha256"] == sources["research/boolean_coupling/audit285.py"], "Audit source"
    )
    require(
        seal["source_hashes"] == sources and seal["cases"] == status["cases"], "Seal sources/cases"
    )
    require(
        seal["requires_public_push_before_native"] is (not design["synthetic"]), "Public gate scope"
    )
    require(
        time(design["began"])
        <= time(status["ended"])
        <= time(audit["ended"])
        <= time(seal["ended"]),
        "Forecast/seal chronology",
    )
    require(
        set(audit["csv_sha256"])
        == {
            n + ".csv"
            for n in (
                "forecast_checks",
                "forecast_coefficients",
                "forecast_raw",
                "forecast_heads",
                "forecast_scores",
            )
        }
        and seal["csv_sha256"] == audit["csv_sha256"],
        "Five sealed forecast tables",
    )
    for name, expected in audit["csv_sha256"].items():
        require(sha(dest / name) == expected, f"Forecast CSV hash: {name}")
    counts = status["counts"]
    require(
        counts["full_forwards"] == 224
        and counts["full_false"] == 32
        and counts["candidate_full_forwards"] == 192
        and counts["first_source_current_SiLU_coordinates"] == 0
        and counts["all_receiving_operations_finite"] is True,
        "Forecast execution budget",
    )
    layers = counts["other_source_sites_finite"] + 1
    require(layers >= 2 and (synthetic_test or layers == 28), "Finite source-site count")
    require(
        runtime["synthetic"] is design["synthetic"]
        and runtime["dtype"] == "torch.float32"
        and runtime["metric_dtype"] == "cpu_float64"
        and runtime["attention"] == "eager"
        and runtime["training"] is False,
        "Runtime declaration",
    )
    if not synthetic_test:
        require(
            runtime["device"] in ("mps", "mps:0")
            and design["checkpoint"]["revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "Pinned production checkpoint/runtime",
        )
        require(
            config == read(root / "research/boolean_coupling/bank285.json"), "Frozen bank config"
        )
    require(
        config["experiment"] == 285
        and config["reference_experiment"] == 279
        and config["match_fields"] == ["schema", "reverse", "target", "assignment"]
        and config["reference_hashes"] == design["reference_hashes"],
        "Historical bank identity",
    )
    require(
        set(config["reference_hashes"])
        == {"design.json", "forecast-status.json", "forecast-audit.json", "forecast-seal.json"},
        "Historical forecast-only references",
    )
    require(len(config["groups"]) == len(design["bank_bindings"]) == 8, "Eight donor groups")
    banks, selected = {}, set()
    for item, binding in zip(config["groups"], design["bank_bindings"]):
        g = item["group"]
        key = tuple(g[n] for n in ("schema", "reverse", "target", "assignment"))
        require(
            key not in banks and g["assignment"] == 0 and len(item["donors"]) == 2,
            "Donor group selection",
        )
        donor_keys = [d["key"] for d in item["donors"]]
        require(not set(donor_keys) & selected and len(set(donor_keys)) == 2, "Unique fixed donors")
        selected.update(donor_keys)
        require(
            binding["group"] == g
            and binding["donor_keys"] == donor_keys
            and HASH.fullmatch(binding["delta_sha256"]),
            "Reconstructed bank binding",
        )
        banks[key] = binding
    require(len(selected) == 16, "16 historical forecast donors")
    if not synthetic_test:
        verify_reference(root, config)
    metadata = verify_inventory(dest, "forecasts", status, cases)
    for key, meta in metadata.items():
        require(
            meta["ordinary_true"] == meta["full_current_first_source_SiLU"] == 0
            and meta["False_zero_exact"] is True
            and meta["all_prefixes_exact"] is True,
            "Forecast metadata qualification",
        )
        require(tuple(meta["arms"]) == ARMS, "Actual seven-arm generation order")
        group = cases[key]["group"]
        binding = banks[tuple(group[n] for n in ("schema", "reverse", "target", "assignment"))]
        require(
            meta["donor_keys"] == binding["donor_keys"]
            and meta["bank_delta_sha256"] == binding["delta_sha256"],
            "Case donor binding",
        )
        require(
            time(design["began"]) <= time(meta["measured"]) <= time(status["ended"]),
            "Forecast measurement time",
        )
        for arm, am in meta["arms"].items():
            require(
                am["ordinary_true"] == 0
                and am["only_first_source_policy"] is True
                and am["false_primal_checks"] == int(arm != "F")
                and am["current_source_finite_sites"] == layers - int(arm != "F")
                and am["full_current_first_source_activation"] == int(arm == "F"),
                "False primal/current activation/source-site budget",
            )
            for field in ("prefix_hashes", "K_prefix", "V_prefix", "product_hashes"):
                require(
                    len(am[field]) == layers and all(HASH.fullmatch(v) for v in am[field]),
                    "Layer hash inventory",
                )
                if field != "product_hashes":
                    require(
                        am[field] == meta["arms"]["F"][field], "Exact causal prefix/KV agreement"
                    )
    verify_numerical(dest, "forecast", audit, cases)
    require(
        audit["False_anchors"] == 32 and audit["False_coefficient_vectors"] == 96,
        "False anchors/coefficients",
    )
    coefficients = rows(dest, "forecast_coefficients")
    require(
        len(coefficients) == 32 and {r["key"] for r in coefficients} == set(cases),
        "32 coefficient rows",
    )
    for row in coefficients:
        require(
            number(row, "bound") == 0.0001
            and all(0 <= number(row, k) <= 0.0001 for k in row if k not in ("key", "bound")),
            "Independent coefficient check",
        )
    raw = rows(dest, "forecast_raw")
    require(
        len(raw) == 320 and Counter(r["key"] for r in raw) == Counter({k: 10 for k in cases}),
        "320 raw diagnostics",
    )
    raw_indices = set()
    for row in raw:
        index = row["key"], row["arm"], row["node"]
        require(
            index not in raw_indices
            and (
                (index[1] in ("F", "C") and index[2] in ("gate", "up"))
                or (index[1] in ("C", "L") and index[2] in ("phi", "product", "write"))
            ),
            "Raw diagnostic coverage",
        )
        raw_indices.add(index)
        require(
            number(row, "original_bound") == 0.0001
            and number(row, "error") >= 0
            and boolean(row, "passes") == (number(row, "error") <= 0.0001),
            "Original raw 1e-4 diagnostic flag",
        )
    raw_counts = dict(
        total=len(raw), failed=sum(not boolean(r, "passes") for r in raw), original_bound=0.0001
    )
    require(audit["raw_checks"] == raw_counts, "Raw diagnostic failures preserved")
    scores = indexed(rows(dest, "forecast_scores"), cases, ARMS, "forecast scores")
    for (key, arm), row in scores.items():
        close(
            number(row, "margin"),
            number(row, "true_score") - number(row, "false_score"),
            "Sealed margin arithmetic",
        )
        if arm == "FZ":
            require(
                all(row[f] == scores[key, "F"][f] for f in ("false_score", "true_score", "margin")),
                "FZ exact scored False",
            )
    if manifest:
        verify_manifest(root, dest, "forecast", sources)
    return dict(
        design=design,
        cases=cases,
        status=status,
        audit=audit,
        seal=seal,
        metadata=metadata,
        scores=scores,
        layers=layers,
        raw_counts=raw_counts,
    )


def good(row, prefix="", relative_bound=0.25, cosine_bound=0.95):
    norm, error, cosine = (number(row, prefix + f) for f in ("norm", "relative", "cosine"))
    require(
        norm >= 0 and error >= 0 and -1.000000000001 <= cosine <= 1.000000000001,
        "Vector metric range",
    )
    return norm >= 0.001 and error <= relative_bound and cosine >= cosine_bound


def endpoint(row, native):
    relative_bound, cosine_bound = (0.05, 0.99) if native else (0.25, 0.95)
    vectors = [good(row, prefix, relative_bound, cosine_bound) for prefix in ("all_", "verdict_")]
    target, error = number(row, "target_margin"), number(row, "margin_error")
    eligible = native or abs(target) >= 0.02
    require(error >= 0 and boolean(row, "margin_eligible") == eligible, "Frozen margin materiality")
    return all(vectors) and eligible and error <= (0.1 if native else 0.005 + 0.25 * abs(target))


def verify_complete(
    root=ROOT, destination=DESTINATION, synthetic_test=False, manifest=True, public_receipt=True
):
    forecast = verify_forecast(root, destination, synthetic_test, manifest)
    base, dest = root / destination / "pre-native", root / destination / "native-phase"
    design, cases = forecast["design"], forecast["cases"]
    status, start, audit = (
        read(dest / n) for n in ("native-status.json", "native-started.json", "native-audit.json")
    )
    require(
        status["status"] == "complete"
        and status["ordinary_true"] == 32
        and status["counts"]["full_forwards"] == 32
        and audit["status"] == "passed"
        and audit["experiment"] == 285
        and audit["ordinary_True_replay_forwards"] == 32,
        "32 native calls plus 32 unhooked replays",
    )
    for filename, field in (
        ("native-status.json", "native_status_sha256"),
        ("native-started.json", "native_started_sha256"),
    ):
        require(audit[field] == sha(dest / filename), "Native audit/status/start binding")
    require(
        status["native_started_sha256"] == sha(dest / "native-started.json"),
        "Native status/start binding",
    )
    require(
        audit["design_sha256"] == sha(base / "design.json")
        and audit["source_sha256"]
        == design["source_hashes"]["research/boolean_coupling/audit285.py"],
        "Native scientific bindings",
    )
    seal_hash = sha(base / "forecast-seal.json")
    require(
        start["forecast_seal_sha256"]
        == status["forecast_seal_sha256"]
        == audit["forecast_seal_sha256"]
        == seal_hash,
        "Native phase binds pre-native seal",
    )
    require(
        start["synthetic"] is design["synthetic"] and start["ordinary_true_before"] == 0,
        "Native start information budget",
    )
    require(
        time(forecast["seal"]["ended"])
        <= time(start["began"])
        <= time(status["ended"])
        <= time(audit["ended"]),
        "Native chronology",
    )
    if not synthetic_test:
        require(
            start["remote_head_verified"] is True
            and re.fullmatch(r"[0-9a-f]{40}", start["public_commit"])
            and start["public_branch"]
            and not start["public_branch"].startswith("synthetic"),
            "Production public receipt",
        )
        if public_receipt:
            published = subprocess.check_output(
                [
                    "git",
                    "show",
                    start["public_commit"]
                    + ":"
                    + str(destination / "pre-native/forecast-seal.json"),
                ],
                cwd=root,
            )
            require(
                hashlib.sha256(published).hexdigest() == seal_hash,
                "Exact seal in public receipt commit",
            )
            subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    design["source_commit"],
                    start["public_commit"],
                ],
                cwd=root,
                check=True,
            )
    metadata = verify_inventory(dest, "native", status, cases)
    require(
        status["raw_bytes"] + forecast["status"]["raw_bytes"] <= 96 * 1024**2, "Combined raw cap"
    )
    for key, meta in metadata.items():
        require(
            meta["ordinary_true"] is True and meta["editing_hooks_applied"] is False,
            "Ordinary native measurement",
        )
        am = meta["arm"]
        require(
            am["false_primal_checks"] == 0
            and am["current_source_finite_sites"] == forecast["layers"]
            and am["ordinary_true"] == am["full_current_first_source_activation"] == 1,
            "Native first-source finite activation",
        )
        require(
            time(start["began"]) <= time(meta["measured"]) <= time(status["ended"]),
            "Native measurement time",
        )
        for field in ("prefix_hashes", "K_prefix", "V_prefix"):
            require(
                am[field] == forecast["metadata"][key]["arms"]["F"][field],
                "Native causal prefix/KV agreement",
            )
    expected_csvs = {
        n + ".csv"
        for n in (
            "native_checks",
            "native_heads",
            "native",
            "local",
            "h1",
            "effect",
            "repair",
            "h2",
        )
    }
    require(set(audit["csv_sha256"]) == expected_csvs, "Eight completed tables")
    for name, expected in audit["csv_sha256"].items():
        require(sha(dest / name) == expected, f"Native CSV hash: {name}")
    verify_numerical(dest, "native", audit, cases)
    require(
        audit["retained_forecast_raw_checks"] == forecast["raw_counts"],
        "Original raw failures retained after native",
    )
    tables, decisions = {}, {}
    for name in ("native", "local", "h1", "effect", "repair", "h2"):
        tables[name] = indexed(rows(dest, name), cases, EVAL, name)
        decisions[name] = {}
    for key in cases:
        for arm in EVAL:
            index = key, arm
            checks = {
                "native": endpoint(tables["native"][index], True),
                "local": good(tables["local"][index], relative_bound=0.1, cosine_bound=0.995),
                "effect": endpoint(tables["effect"][index], False),
                "repair": good(tables["repair"][index]),
            }
            checks["h1"] = checks["native"] and checks["local"]
            checks["h2"] = checks["effect"] and checks["repair"]
            for name, passed in checks.items():
                require(
                    boolean(tables[name][index], "passes") == passed,
                    f"Frozen {name} decision: {index}",
                )
                decisions[name][index] = passed
            # All arms have identical targets; no eligibility-based case removal.
            for name, fields in (
                ("native", ("all_norm", "verdict_norm", "target_margin")),
                ("effect", ("all_norm", "verdict_norm", "target_margin")),
                ("local", ("norm",)),
                ("repair", ("norm",)),
            ):
                for field in fields:
                    close(
                        number(tables[name][index], field),
                        number(tables[name][key, "T"], field),
                        "Target common to all arms",
                    )
        fmargin = number(forecast["scores"][key, "F"], "margin")
        mmargin = number(forecast["scores"][key, "M"], "margin")
        tmargin = fmargin + number(tables["native"][key, "T"], "target_margin")
        for arm in EVAL:
            predicted = tmargin if arm == "T" else number(forecast["scores"][key, arm], "margin")
            close(
                number(tables["effect"][key, arm], "target_margin"),
                tmargin - mmargin,
                "Repair target from sealed M scores",
            )
            for name in ("native", "effect"):
                close(
                    number(tables[name][key, arm], "margin_error"),
                    abs(predicted - tmargin),
                    "Outcome margin error from sealed forecasts",
                )
        for name in ("native", "local", "effect", "repair"):
            row = tables[name][key, "T"]
            for prefix in ("all_", "verdict_") if name in ("native", "effect") else ("",):
                close(number(row, prefix + "relative"), 0, "T calibration relative error")
                norm = number(row, prefix + "norm")
                close(
                    number(row, prefix + "cosine"),
                    norm * norm / max(norm * norm, 1e-30),
                    "T calibration cosine",
                )
            if name in ("native", "effect"):
                close(number(row, "margin_error"), 0, "T calibration margin error")
    counts = {
        name: {a: sum(p for (k, arm), p in decision.items() if arm == a) for a in EVAL}
        for name, decision in decisions.items()
    }
    for name, values in counts.items():
        field = {"h1": "H1", "h2": "H2"}.get(name, name) + "_counts"
        require(audit[field] == values, f"Recomputed audit counts: {name}")
    result = {"counts": counts}
    layouts = sorted({c["group"]["block"] for c in cases.values()})
    for name in ("h1", "h2"):
        strata = {
            b: sum(
                decisions[name][key, "C"]
                for key, case in cases.items()
                if case["group"]["block"] == b
            )
            for b in layouts
        }
        passed = counts[name]["C"] >= 24 and min(strata.values()) >= 6
        require(
            audit[name.upper() + "_C_layouts"] == strata
            and audit[name.upper() + "_C_passes"] is passed,
            "Primary threshold/layout strata",
        )
        result[name] = dict(layouts=strata, passes=passed)
    eligible = sum(boolean(tables["effect"][key, "T"], "margin_eligible") for key in cases)
    require(audit["H2_margin_eligible"] == eligible, "All-32 H2 materiality count")
    result["H2_margin_eligible"] = eligible
    if manifest:
        verify_manifest(root, root / destination, "final", design["source_hashes"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("forecast", "complete"), required=True)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.phase == "forecast":
        checked = verify_forecast(destination=args.destination)
        output = dict(
            phase="forecast",
            cases=len(checked["cases"]),
            numerical_checks=checked["audit"]["numerical_checks"],
            raw_checks=checked["raw_counts"],
        )
    else:
        output = verify_complete(destination=args.destination)
    print(json.dumps(dict(status="passed", experiment=285, **output), sort_keys=True))
