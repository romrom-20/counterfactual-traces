"""Posthoc published-record inventory and current-batch completion check."""

import json

from .audit229 import sha
from .cache272 import ROOT, save_json

DEST = ROOT / "results/causal-prefix-budget-272"


def read(path):
    return json.loads(path.read_text())


def main():
    target = DEST / "research-review.json"
    assert not target.exists()
    inputs, records = {}, []
    for path in sorted((ROOT / "results").rglob("design.json")):
        if path.is_relative_to(DEST):
            continue
        parts = path.relative_to(ROOT).parts
        if any(p in ("sources", "preflight") or p.startswith("frozen-") for p in parts):
            continue
        design = read(path)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        observed = []
        for item in sorted(path.parent.glob("*.json")):
            if not ("audit" in item.name or "status" in item.name or "failure" in item.name):
                continue
            value = read(item)
            inputs[str(item.relative_to(ROOT))] = sha(item)
            flags = {k: v for k, v in value.items() if k.startswith("primary")
                     and isinstance(v, (bool, str, int, float))}
            observed.append(dict(file=str(item.relative_to(ROOT)), status=value.get("status"),
                                 primary_fields=flags))
        records.append(dict(design=str(path.relative_to(ROOT)), experiment=design.get("experiment"),
            declared_case_count=len(design["cases"]) if isinstance(design.get("cases"), list) else None,
            retrospective=design.get("retrospective"), source_commit=design.get("source_commit"),
            observed_records=observed))
    latest = []
    for name, cases in (
        ("results/signed-amplitude-forecast-269/pre-native/forecast-status.json", 32),
        ("results/signed-amplitude-forecast-269/native-status.json", 32),
        ("results/backbone-feedback-270-271/backbone-curvature-feedback-270/status.json", 4),
        ("results/backbone-feedback-270-271/amplitude-query-mediation-271/status.json", 4)):
        path = ROOT / name
        status = read(path)
        assert status["status"] == "complete" and len(status["cases"]) == cases
        inputs[name] = sha(path)
        latest.append(dict(file=name, status=status["status"], case_records=cases))
    save_json(target, dict(posthoc=True, model_calls=0, source_sha256=sha(__file__),
        inputs=inputs, design_artifacts=records, current_batch=latest,
        current_269_271_batches_complete=True,
        interpretation="Artifact inventory, not a count of independent experiments or endorsement of every conclusion. An absent status/cases field reflects heterogeneous schemas; consult the sealed reports/qualifications. Preserved failures are not pending runs.",
        assessment="Recent policy-contrast diagnostics constrain approximations but are not cumulative proof of the native learned generator. Keep native target, budget and circuit claims separate."))
    print("Reviewed", len(records), "published design artifacts; latest269/270/271 batches complete; zero model calls.")


if __name__ == "__main__":
    main()
