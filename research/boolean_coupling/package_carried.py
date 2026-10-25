"""Publication metadata only: immutable284 forecast release precedes native data."""

import argparse
import json
import shutil
from pathlib import Path

from .audit229 import sha
from .audit272 import read

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/carried-state-response-283"
FRESH = ROOT / "results/carried-state-forecast-284"


def save(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, indent=2) + "\n")


def copy_file(source, dest):
    if dest.exists():
        assert sha(source) == sha(dest), dest
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def records(run, dest, include):
    arrays = []
    for p in sorted(run.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run)
        if "sources" in rel.parts or not include(rel):
            continue
        if p.suffix == ".npz":
            arrays.append(dict(path=str(rel), sha256=sha(p), bytes=p.stat().st_size))
        else:
            assert p.suffix in (".json", ".csv"), p
            copy_file(p, dest / rel)
    return arrays


def logs(dest, folder, names, filename):
    results = []
    for name in names:
        p = folder / (name + ".log")
        selected = [
            line.rstrip()
            for line in p.read_text().splitlines()
            if line.startswith(
                (
                    "283 ",
                    "284 ",
                    "{'passed':",
                    "{'primary':",
                    "Traceback",
                    "AssertionError",
                    "ValueError",
                    "AttributeError",
                    "KeyError",
                    "real ",
                    "user ",
                    "sys ",
                )
            )
        ]
        q = dest / (name + ".log")
        assert not q.exists()
        q.write_text("\n".join(selected) + "\n")
        results.append(
            dict(
                stage=name,
                raw_sha256=sha(p),
                published_sha256=sha(q),
                representation="Progress/results/errors/timing only; local paths omitted and trailing whitespace removed. Original retained.",
            )
        )
    save(dest / filename, dict(logs=results))


def forecast(run, folder):
    assert read(run / "forecast-audit.json")["status"] == "passed"
    assert (
        not (run / "native-started.json").exists()
        and not (run / "native").exists()
        and not FRESH.exists()
    )
    FRESH.mkdir()
    arrays = records(run, FRESH, lambda rel: True)
    assert len(arrays) == 32 and all(r["path"].startswith("forecasts/") for r in arrays)
    save(
        FRESH / "forecast-array-inventory.json",
        dict(
            arrays=arrays,
            omitted=True,
            regenerate="research/README.md#carried-state-response-283284",
        ),
    )
    logs(
        FRESH,
        folder,
        (
            "284-synthetic-pipeline",
            "284-synthetic-pipeline-v2",
            "284-forecast",
            "284-forecast-audit",
        ),
        "forecast-log-provenance.json",
    )
    save(
        FRESH / "forecast-execution-notes.json",
        dict(
            real_forecast_full_forwards=320,
            ordinary_true_before_publication=0,
            full_current_source2_before_publication=0,
            finite_C_maps=32,
            local_down_maps=64,
            staged_norm_jvps=32,
            staged_gate_up_matrix_products=64,
            full_unequal_CPU_pipeline_passed_before_freeze=True,
            premature_and_wrong_seal_native_tests_passed=True,
            templates_dependent=True,
            learned_circuit=False,
            efficiency_claim=False,
            general_Question_B_solved=False,
            private_article_included=False,
        ),
    )
    design = read(FRESH / "design.json")
    paths = [p for p in FRESH.rglob("*") if p.is_file()] + [
        ROOT / n for n in design["source_hashes"]
    ]
    assert all(p.suffix != ".npz" and ".context" not in p.relative_to(ROOT).parts for p in paths)
    save(
        FRESH / "forecast-manifest.json",
        dict(
            stage="forecasts-sealed-before-native",
            scientific_freeze=design["source_commit"],
            files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
            scope="Immutable inputs/scientific source/forecasts/numerical qualification. Presentation and posthoc delivery code may be finalized later; no native measurements.",
        ),
    )


def old(run, folder):
    assert read(run / "audit.json")["status"] == "passed" and not OLD.exists()
    OLD.mkdir()
    arrays = records(run, OLD, lambda rel: True)
    save(
        OLD / "raw-array-inventory.json",
        dict(
            arrays=arrays,
            omitted=True,
            reference="source2-origin-282",
            lossless_aliases=True,
            regenerate="research/README.md#carried-state-response-283284",
        ),
    )
    logs(
        OLD,
        folder,
        (
            "283-synthetic-producer-audit",
            "283-synthetic-producer-audit-v2",
            "283-producer",
            "283-audit",
        ),
        "log-provenance.json",
    )
    failed = folder / "283-synthetic-producer-audit/complete/sources/preflight283.py"
    save(
        OLD / "execution-notes.json",
        dict(
            initial_unsealed_CPU_assertion_failure=True,
            failure_cause="Test wrongly equated local-vector and downstream margin eligibility after a complete passing producer/numerical audit. Fixed only the assertion before the passing fixture and freeze.",
            failed_unsealed_preflight_sha256=sha(failed),
            source_or_bound_changes_after_freeze=False,
            checkpoint_producer_full_forwards=448,
            finite_C_maps=32,
            local_down_maps=64,
            staged_norm_jvps=32,
            staged_gate_up_matrix_products=64,
            failed_checkpoint_attempts=0,
            independent_research_team=False,
            fresh_confirmation=False,
            learned_circuit=False,
            efficiency_claim=False,
            general_Question_B_solved=False,
            private_article_included=False,
        ),
    )


def native(run, folder):
    assert read(run / "native-audit.json")["status"] == "passed"
    forecast_manifest = read(FRESH / "forecast-manifest.json")
    for n, h in forecast_manifest["files"].items():
        assert sha(ROOT / n) == h, n
    prefixes = ("native-", "native_")
    names = {
        "native.csv",
        "local.csv",
        "effect.csv",
        "incremental.csv",
        "incremental_effect.csv",
        "removal.csv",
        "paired.csv",
        "joint.csv",
        "ledger.csv",
    }
    arrays = records(
        run,
        FRESH,
        lambda rel: rel.parts[0] == "native" or rel.name.startswith(prefixes) or rel.name in names,
    )
    assert len(arrays) == 32
    save(
        FRESH / "native-array-inventory.json",
        dict(arrays=arrays, omitted=True, aliases_reference_sealed_forecasts=True),
    )
    logs(FRESH, folder, ("284-native", "284-native-audit"), "native-log-provenance.json")
    save(
        FRESH / "native-execution-notes.json",
        dict(
            forecast_full_forwards=320,
            native_control_full_forwards=128,
            independent_ordinary_True_audit_replay_forwards=32,
            forecasts_and_qualification_public_before_native=True,
            forecast_manifest_sha256=sha(FRESH / "forecast-manifest.json"),
            source_or_bound_changes_after_forecasts=False,
            templates_dependent=True,
            learned_circuit=False,
            efficiency_claim=False,
            general_Question_B_solved=False,
            private_article_included=False,
        ),
    )


def seal(dest, experiment):
    assert not (dest / "manifest.json").exists()
    d = read(dest / "design.json")
    for n, h in d["source_hashes"].items():
        assert sha(ROOT / n) == h
    report = f"docs/analysis/{experiment}-carried-state-response.md"
    paths = [p for p in dest.rglob("*") if p.is_file()] + [ROOT / n for n in d["source_hashes"]]
    paths += [
        ROOT / n
        for n in (
            report,
            "research/boolean_coupling/package_carried.py",
            "research/boolean_coupling/verify_carried_response.py",
            "research/boolean_coupling/describe_carried.py",
        )
    ]
    paths += list((ROOT / f"docs/figures/carried-response/{experiment}").iterdir())
    assert all(
        p.suffix != ".npz"
        and ".context" not in p.relative_to(ROOT).parts
        and "drafts" not in p.parts
        for p in paths
    )
    save(
        dest / "manifest.json",
        dict(
            stage="completed-old-retrospective" if experiment == 283 else "completed-fresh-literal",
            scientific_freeze=d["source_commit"],
            files={str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
            scope="Release integrity, all cases and retained failures; not independent model replication.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("forecast", "old", "native", "seal-old", "seal-native"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    a = p.parse_args()
    if a.stage == "forecast":
        forecast(a.run.resolve(), a.logs.resolve())
    elif a.stage == "old":
        old(a.run.resolve(), a.logs.resolve())
    elif a.stage == "native":
        native(a.run.resolve(), a.logs.resolve())
    else:
        seal(OLD if a.stage == "seal-old" else FRESH, 283 if a.stage == "seal-old" else 284)
