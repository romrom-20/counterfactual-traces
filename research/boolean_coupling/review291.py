"""Posthoc promoted-array rounding and gradient accounting; frozen decisions unchanged."""

import argparse
import itertools
from pathlib import Path

import numpy as np

from .publish285 import read, sha, write
from .verify288 import rows


def main(run, out):
    assert not out.exists()
    audit, status = read(run / "audit.json"), read(run / "status.json")
    assert audit["status"] == "passed"
    old = {(r["key"], r["arm"]): r for r in rows(run / "receiving_vectors.csv")}
    ledger = {(r["key"], r["term"]): r for r in rows(run / "ledger.csv")}
    records, interactions, gradient_accounting = [], [], []
    corners = tuple(f"{j:03b}" for j in range(8))
    arms = (*corners, "R0", "R1", "R2")
    for entry in status["cases"]:
        key = entry["key"]
        p = run / "cases" / (key + ".npz")
        assert sha(p) == entry["arrays_sha256"]
        with np.load(p) as data:
            a = {k: data[k + "_finite_offset"].astype(np.float64)[1:] for k in arms}
            raw = {k: data[k + "_finite_offset"][1:] for k in arms}
            gs = np.stack([data[k + "_gradient"].astype(np.float64) for k in corners])
        gram = gs @ gs.T
        target_g = gs[7] - gs[0]
        gn = float(np.linalg.norm(target_g))
        terms = {}
        for term in corners[1:]:
            subset = [j for j, bit in enumerate(term) if bit == "1"]
            coeff = np.zeros(8)
            for bits in itertools.product((0, 1), repeat=len(subset)):
                c = ["0"] * 3
                for j, bit in zip(subset, bits, strict=True):
                    c[j] = str(bit)
                coeff[int("".join(c), 2)] += (-1) ** (len(subset) - sum(bits))
            terms[term] = coeff @ gs
        main = sum(v for k, v in terms.items() if k.count("1") == 1)
        second = sum(v for k, v in terms.items() if k.count("1") <= 2)
        assert np.max(abs(sum(terms.values()) - target_g)) < 1e-12
        gradient_accounting.append(
            dict(
                key=key,
                corner_gram=gram.tolist(),
                target_norm=gn,
                main_relative=float(np.linalg.norm(main - target_g)) / gn,
                through_pairs_relative=float(np.linalg.norm(second - target_g)) / gn,
                terms={
                    k: dict(
                        norm_ratio=float(np.linalg.norm(v)) / gn,
                        target_projection=float(v @ target_g) / (gn * gn),
                    )
                    for k, v in terms.items()
                },
            )
        )
        target = a["111"] - a["000"]
        norm = float(np.linalg.norm(target))
        for arm in arms:
            pred = a[arm] - a["000"]
            raw_target = raw["111"] - raw["000"]
            raw_pred = raw[arm] - raw["000"]
            relative = float(np.linalg.norm(pred - target)) / max(norm, 1e-30)
            cosine = float(np.sum(pred * target)) / max(float(np.linalg.norm(pred)) * norm, 1e-30)
            passed = norm >= 0.001 and relative <= 0.25 and cosine >= 0.95
            before = old[key, arm]
            records.append(
                dict(
                    key=key,
                    arm=arm,
                    relative=relative,
                    original_F32_dot=float(np.sum(raw_pred * raw_target)),
                    original_F32_target_norm=float(np.linalg.norm(raw_target)),
                    cosine=cosine,
                    relative_change=abs(relative - float(before["relative"])),
                    cosine_change=abs(cosine - float(before["cosine"])),
                    frozen_pass=before["passes"] == "True",
                    promoted_pass=passed,
                )
            )
        total = np.zeros_like(target)
        for term in corners[1:]:
            subset = [j for j, bit in enumerate(term) if bit == "1"]
            rv = np.zeros_like(target)
            for bits in itertools.product((0, 1), repeat=len(subset)):
                c = ["0"] * 3
                for j, bit in zip(subset, bits, strict=True):
                    c[j] = str(bit)
                rv += (-1) ** (len(subset) - sum(bits)) * a["".join(c)]
            total += rv
            before = ledger[key, term]
            projection = float(np.sum(rv * target)) / max(norm, 1e-30)
            interactions.append(
                dict(
                    key=key,
                    term=term,
                    norm_change=abs(float(np.linalg.norm(rv)) - float(before["receiving_norm"])),
                    projection_change=abs(
                        projection - float(before["receiving_signed_target_projection"])
                    ),
                )
            )
        assert np.max(abs(total - target)) < 1e-12
    write(
        out,
        dict(
            experiment=291,
            posthoc=True,
            model_calls=0,
            script_sha256=sha(Path(__file__)),
            audit_sha256=sha(run / "audit.json"),
            raw_archives={e["key"]: e["arrays_sha256"] for e in status["cases"]},
            vector_comparisons=len(records),
            interaction_comparisons=len(interactions),
            classification_changes=sum(r["frozen_pass"] != r["promoted_pass"] for r in records),
            maximum_relative_change=max(r["relative_change"] for r in records),
            maximum_cosine_change=max(r["cosine_change"] for r in records),
            maximum_ledger_norm_change=max(r["norm_change"] for r in interactions),
            maximum_ledger_projection_change=max(r["projection_change"] for r in interactions),
            gradient_accounting=gradient_accounting,
            records=records,
            interactions=interactions,
            boundary="Promoted Float64 subtraction and factorial sums of saved Float32 endpoints; frozen Float32 decisions remain original. Gradient Gram matrices permit reproducible signed factorial accounting. Not new model or derivative validation or a unique causal attribution.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.out.resolve())
