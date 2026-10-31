"""Separately frozen posthoc CPU64 review; original298 failures stay failed."""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from .audit298 import cpu_final, cpu_head, margin, scientific
from .cache272 import save_json
from .messages298 import GROUPS, ROOT, record, reference_weights
from .publish285 import read, sha

REVIEW_SOURCES = (
    "research/boolean_coupling/review298.py",
    "docs/experiments/298-posthoc-numerical-review.md",
)


def main(run):
    assert not (run / "numerical-review.json").exists()
    design = read(run / "design.json")
    for name, digest in design["source_hashes"].items():
        assert sha(ROOT / name) == digest
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    import hashlib

    hashes = {n: sha(ROOT / n) for n in REVIEW_SOURCES}
    for name, digest in hashes.items():
        assert (
            hashlib.sha256(
                subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
            ).hexdigest()
            == digest
        )
    weights = reference_weights(design["checkpoint"])
    checks = []
    rows = []
    operations = joint_decisions = 0
    status = read(run / "status.json")
    for case, entry in zip(design["cases"], status["cases"], strict=True):
        p = run / "cases" / (case["key"] + ".npz")
        assert sha(p) == entry["arrays_sha256"]
        with np.load(p) as archive:
            actual = dict(archive)
        cpu = dict(actual)
        names = ["native_reference", "frozen_query_reference"] + [
            g + "_" + o for g in GROUPS for o in ["add", "remove"]
        ]
        for name in names:
            final = cpu_final(weights, actual[name + "_middle"])
            cpu[name + "_final"], cpu[name + "_scores"] = final, cpu_head(weights, final)
        ae, _, aj = scientific(case, actual)
        ce, _, cj = scientific(case, cpu)
        for a, c in zip(ae, ce, strict=True):
            record(
                checks,
                case["key"] + a["group"] + a["operation"] + "/classification",
                float(a["passes"] != c["passes"]),
                0.0,
            )
            operations += 1
        for a, c in zip(aj, cj, strict=True):
            record(
                checks, case["key"] + a["group"] + "/joint", float(a["passes"] != c["passes"]), 0.0
            )
            record(
                checks,
                case["key"] + a["group"] + "/eligibility",
                float(a["eligible_margin"] != c["eligible_margin"]),
                0.0,
            )
            joint_decisions += 1
        for g in GROUPS:
            for operation in ["add", "remove"]:
                name = g + "_" + operation
                baseline = "frozen_query_reference" if operation == "add" else "native_reference"
                sign = 1 if operation == "add" else -1
                ap = sign * (
                    actual[name + "_final"].astype(float)
                    - actual[baseline + "_final"].astype(float)
                )
                cp = sign * (cpu[name + "_final"] - cpu[baseline + "_final"])
                target = actual["native_reference_final"].astype(float) - actual[
                    "frozen_query_reference_final"
                ].astype(float)
                for scope, index in [("receiving", slice(None)), ("verdict", -1)]:
                    norm = float(np.linalg.norm(target[index]))
                    delta = float(np.linalg.norm(ap[index] - cp[index]))
                    error = delta / norm if norm else delta
                    bound = 1e-3 if norm else 1e-6
                    record(checks, case["key"] + name + "/" + scope, error, 0.0, bound)
                    rows.append(
                        dict(
                            key=case["key"],
                            group=g,
                            operation=operation,
                            scope=scope,
                            target_norm=norm,
                            scaled_contrast_error=error,
                            bound=bound,
                        )
                    )
                am = sign * (margin(actual, name) - margin(actual, baseline))
                cm = sign * (margin(cpu, name) - margin(cpu, baseline))
                record(checks, case["key"] + name + "/margin", am, cm, 1e-3)
    failed = [r for r in checks if not r["passes"]]
    save_json(
        run / "numerical-review.json",
        dict(
            experiment=298,
            posthoc=True,
            passed=not failed,
            original_diagnostics_unchanged=True,
            source_commit=commit,
            source_hashes=hashes,
            design_sha256=sha(run / "design.json"),
            audit_sha256=sha(run / "audit.json"),
            required_checks=len(checks),
            failed_checks=len(failed),
            operation_decisions=operations,
            joint_decisions=joint_decisions,
            checks=checks,
            contrasts=rows,
            max_scaled_contrast_error=max(r["scaled_contrast_error"] for r in rows),
            full_transformer_forwards=0,
            new_terminal_interventions=0,
            boundary="Promoted finite arithmetic stability for conditional comparisons; no original raw pass or independent model replication.",
        ),
    )
    print(
        dict(
            passed=not failed,
            checks=len(checks),
            failed=len(failed),
            operation_decisions=operations,
            joint_decisions=joint_decisions,
            max_contrast_error=max(r["scaled_contrast_error"] for r in rows),
        ),
        flush=True,
    )
    assert not failed, "Posthoc contrast review failed; preserve qualifications"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
