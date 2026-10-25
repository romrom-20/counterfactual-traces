"""Bind the cache-only continuation, then execute ORIGINAL native audit284 unchanged."""

import argparse
from pathlib import Path

from . import audit284
from .audit229 import sha
from .audit272 import read
from .cache272 import save_json
from .carried284 import ROOT, bindings


def main(run):
    assert not (run / "native-continuation-audit.json").exists()
    bindings(run)
    qualification, status, old = (
        read(run / (n + ".json"))
        for n in (
            "native-continuation-design",
            "native-continuation-status",
            "retained-native-stop/inventory",
        )
    )
    assert qualification["qualification"] == "284b-unused-MPS-cache-and-exact-partial-replay"
    for n, h in qualification["source_hashes"].items():
        assert sha(ROOT / n) == sha(run / "native-continuation-sources" / Path(n).name) == h
    assert qualification["design_sha256"] == sha(run / "design.json") and qualification[
        "forecast_seal_sha256"
    ] == sha(run / "forecast-seal.json")
    assert qualification["original_failure_sha256"] == sha(
        run / "retained-native-stop/failure-native.json"
    )
    assert (
        qualification["original_native_started_sha256"]
        == sha(run / "native-started.json")
        == sha(run / "retained-native-stop/native-started.json")
    )
    assert qualification["original_partial_inventory_sha256"] == sha(
        run / "retained-native-stop/inventory.json"
    )
    assert not any(
        qualification[k]
        for k in (
            "predictions_changed",
            "scientific_thresholds_changed",
            "numerical_validation_scope_changed",
        )
    )
    assert (
        qualification["per_case_reserve_MiB"] == 512 and qualification["combined_raw_cap_MiB"] == 96
    )
    assert (
        qualification["replay_every_completed_case_before_new_native"]
        and qualification["previously_measured_native_cases"] == 17
    )
    assert (
        status["status"] == "complete"
        and status["all_replay_arrays_and_metadata_exact"]
        and len(status["replays"]) == len(old["cases"]) == 17
    )
    for actual, prior in zip(status["replays"], old["cases"], strict=True):
        assert (
            actual["exact"]
            and actual["key"] == prior["key"]
            and actual["arrays_sha256"] == prior["arrays_sha256"]
            and actual["metadata_sha256"] == prior["metadata_sha256"]
        )
        key = actual["key"]
        for suffix, field in ((".npz", "arrays_sha256"), (".json", "metadata_sha256")):
            assert (
                sha(run / "replay" / (key + suffix))
                == sha(run / "native" / (key + suffix))
                == prior[field]
            )
    assert old["full_forwards"] == status["original_partial_full_forwards"] == 68
    assert (
        status["replay_full_forwards"] == 68
        and status["new_unmeasured_full_forwards"] == 60
        and status["total_native_producer_full_forwards"] == 196
    )
    assert (
        status["unique_native_cases"] == 32 and status["actual_native_True_producer_forwards"] == 49
    )
    assert status["cache_events_sha256"] == sha(run / "native-cache-events.csv")
    # The original numerical/readout/endpoint implementation is not changed.
    audit284.native(run)
    assert read(run / "native-audit.json")["status"] == "passed"
    save_json(
        run / "native-continuation-audit.json",
        dict(
            status="passed",
            qualification=qualification["qualification"],
            source_sha256=sha(Path(__file__)),
            qualification_sha256=sha(run / "native-continuation-design.json"),
            continuation_status_sha256=sha(run / "native-continuation-status.json"),
            original_native_audit_sha256=sha(run / "native-audit.json"),
            original_native_audit_source_sha256=sha(ROOT / "research/boolean_coupling/audit284.py"),
            scientific_predictions_and_bounds_unchanged=True,
            all17_replay_arrays_and_metadata_exact=True,
            fresh_literals_in_dependent_templates=True,
            private_article_included=False,
            boundary="Original native producer storage stop retained; only unused MPS cache/runtime scheduling changed. Original native audit unchanged. No new numerical scope, learned circuit, efficiency or generalQB claim.",
        ),
    )
    print(
        "284b cache-only continuation qualified:17 exact array+metadata replays, ORIGINAL native audit passed; predictions unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    main(p.parse_args().run.resolve())
