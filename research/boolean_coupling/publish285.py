"""Posthoc delivery of immutable 285 records; never runs the checkpoint.

Forecast delivery excludes this helper and presentation files from its manifest.
Complete delivery is sealed separately after the report and figures are ready.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path("results/native-structure-285")
FORECAST = (
    "design.json",
    "runtime.json",
    "bank-config.json",
    "forecast-status.json",
    "forecast-audit.json",
    "forecast-seal.json",
    "forecast_checks.csv",
    "forecast_coefficients.csv",
    "forecast_raw.csv",
    "forecast_heads.csv",
    "forecast_scores.csv",
)
NATIVE = (
    "native-started.json",
    "native-status.json",
    "native-audit.json",
    "native_checks.csv",
    "native_heads.csv",
    "native.csv",
    "local.csv",
    "h1.csv",
    "effect.csv",
    "repair.csv",
    "h2.csv",
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text() == content, f"Refusing to overwrite {path.name}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def copy(source, dest):
    assert source.is_file(), source.name
    if source.suffix in (".json", ".csv"):
        content = source.read_text()
        assert "/Users/" not in content and ".context/" not in content, (
            "Private path in data record; preserve the original locally and prepare an explicit redacted public representation",
            source.name,
        )
    if dest.exists():
        assert sha(source) == sha(dest), f"Refusing to overwrite {dest.name}"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def safe_relative(path, root):
    rel = path.resolve().relative_to(root.resolve())
    assert not any(p.startswith(".") for p in rel.parts) and "drafts" not in rel.parts
    assert path.suffix != ".npz" and "sources" not in rel.parts
    return str(rel)


def manifest(root, dest, phase, sources, artifacts=()):
    paths = {p for p in dest.rglob("*") if p.is_file() and p.name != phase + "-manifest.json"}
    paths.update(root / name for name in sources)
    paths.update(root / name for name in artifacts)
    write(
        dest / (phase + "-manifest.json"),
        dict(
            experiment=285,
            phase=phase,
            files={safe_relative(p, root): sha(p) for p in sorted(paths)},
            scientific_sources=sources,
            scope=(
                "Immutable scientific inputs, all forecast records and qualification; "
                "delivery helpers and presentation finalized separately."
                if phase == "forecast"
                else "Both immutable phases, scientific sources, delivery helpers and final presentation."
            ),
        ),
    )


def array_inventory(run, dest, folder, status_name):
    status = read(run / status_name)
    entries = status["cases"]
    assert status["status"] == "complete" and len(entries) == 32
    assert len({e["key"] for e in entries}) == 32
    expected = {e["key"] + ".npz" for e in entries}
    assert {p.name for p in (run / folder).glob("*.npz")} == expected
    assert {p.name for p in (run / folder).glob("*.json")} == {
        name.replace(".npz", ".json") for name in expected
    }
    arrays = []
    for entry in entries:
        p = run / folder / (entry["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"] and p.stat().st_size == entry["bytes"]
        assert sha(p.with_suffix(".json")) == entry["metadata_sha256"]
        copy(p.with_suffix(".json"), dest / folder / p.with_suffix(".json").name)
        arrays.append(dict(path=f"{folder}/{p.name}", sha256=sha(p), bytes=p.stat().st_size))
    assert status["raw_bytes"] == sum(e["bytes"] for e in arrays)
    write(
        dest / "omitted-array-inventory.json",
        dict(
            experiment=285,
            omitted=True,
            arrays=arrays,
            raw_bytes=status["raw_bytes"],
            boundary="Original compressed arrays retained locally; hashes and sizes verified before delivery. No weights or activation arrays are published.",
        ),
    )


def logs(folder, dest):
    if folder is None:
        return
    records = []
    for p in sorted(folder.glob("285*.log")):
        assert p.is_file()
        # Keep progress, result dictionaries, error types and timing, without
        # publishing traceback source lines, private paths or environment data.
        selected = []
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith(
                (
                    "285 ",
                    "{'",
                    "Traceback",
                    "AssertionError",
                    "ValueError",
                    "RuntimeError",
                    "MemoryError",
                    "real ",
                    "user ",
                    "sys ",
                )
            ):
                line = re.sub(r"/Users/[^\s\"']+", "<local-path>", line)
                line = re.sub(r"\.?context/[^\s\"']+", "<local-path>", line)
                selected.append(line.rstrip())
        target = dest / "logs" / p.name
        content = "\n".join(selected) + "\n"
        if target.exists():
            assert target.read_text() == content
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        records.append(dict(name=p.name, raw_sha256=sha(p), published_sha256=sha(target)))
    write(
        dest / "log-provenance.json",
        dict(
            logs=records,
            representation="Progress/results/error types/timing only; original logs retained locally.",
        ),
    )


def forecast(run, root=ROOT, destination=DESTINATION, log_folder=None, synthetic_test=False):
    design = read(run / "design.json")
    assert design["experiment"] == 285 and (not design["synthetic"] or synthetic_test)
    assert not (run / "native-started.json").exists(), "Forecast package must precede native True"
    assert not (run / "native").exists()
    assert read(run / "forecast-audit.json")["status"] == "passed"
    dest = root / destination / "pre-native"
    for name in FORECAST:
        copy(run / name, dest / name)
    # Include all other JSON/CSV records, including original failure diagnostics.
    # Frozen source copies and raw arrays remain local.
    for p in sorted(run.rglob("*")):
        rel = p.relative_to(run)
        if p.is_file() and p.suffix in (".json", ".csv") and "sources" not in rel.parts:
            copy(p, dest / rel)
    array_inventory(run, dest, "forecasts", "forecast-status.json")
    logs(log_folder, dest)
    for name, expected in design["source_hashes"].items():
        assert sha(root / name) == sha(run / "sources" / Path(name).name) == expected, name
    manifest(root, dest, "forecast", design["source_hashes"])
    return dest


def complete(run, root=ROOT, destination=DESTINATION, log_folder=None):
    base = root / destination
    design = read(run / "design.json")
    assert not design["synthetic"] and read(run / "native-audit.json")["status"] == "passed"
    # Every previously published byte must remain exact.
    for name in FORECAST:
        assert sha(run / name) == sha(base / "pre-native" / name), name
    dest = base / "native-phase"
    for name in NATIVE:
        copy(run / name, dest / name)
    for p in sorted(run.rglob("*")):
        rel = p.relative_to(run)
        if not p.is_file() or p.suffix not in (".json", ".csv") or "sources" in rel.parts:
            continue
        previous = base / "pre-native" / rel
        if previous.exists():
            assert sha(p) == sha(previous), "Previously published data changed"
        else:
            copy(p, dest / rel)
    array_inventory(run, dest, "native", "native-status.json")
    logs(log_folder, dest)
    assert (
        sum(
            read(base / p / "omitted-array-inventory.json")["raw_bytes"]
            for p in ("pre-native", "native-phase")
        )
        <= 96 * 1024**2
    )
    return dest


def seal(root=ROOT, destination=DESTINATION, artifacts=()):
    base = root / destination
    design = read(base / "pre-native/design.json")
    assert not design["synthetic"]
    assert read(base / "native-phase/native-audit.json")["status"] == "passed"
    required = (
        "research/boolean_coupling/publish285.py",
        "research/boolean_coupling/verify285.py",
        "docs/analysis/285-native-structure.md",
    )
    included = tuple(dict.fromkeys((*required, *artifacts)))
    assert any("docs/figures/" in p for p in included), "Final seal requires figures"
    manifest(root, base, "final", design["source_hashes"], included)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forecast", "complete", "seal"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()
    if args.phase == "seal":
        seal(destination=args.destination, artifacts=args.artifact)
    else:
        assert args.run is not None, "--run is required for data delivery"
        operation = forecast if args.phase == "forecast" else complete
        operation(args.run.resolve(), destination=args.destination, log_folder=args.logs)
    print(f"285 {args.phase} delivery passed")
