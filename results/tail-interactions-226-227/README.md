# Continuation release: 226b and 227b

This release adds the two completed retrospective follow-ups, their original failed attempts, frozen scientific sources, protocols, case CSVs, numerical audits, logs, runtime records, and figures. It reuses the byte-identical public inputs and model helpers from 225b.

The result reports are [226b](../../docs/analysis/226b-finite-tail-interaction-result.md) and [227b](../../docs/analysis/227b-mlp24-mixed-primitives-result.md). The [reproduction guide](../../research/README.md) runs the complete chain from public prompts and the pinned checkpoint.

## What is sealed

`manifest.json` records the published continuation files. `original-local-manifest.json` is the original 56-file snapshot made after the experiments, before this repository update. Its `local_only` and `not_pushed` fields describe that historical snapshot, not the current repository's state.

`publication-revision.json` records the original and current SHA256 for four presentation files: the two reports and two result guides. Their changes remove stale "not pushed" wording and point to this release record. Scientific sources, protocols, numerical artifacts, failure records, logs and figures remain byte-identical. The 225b release is unchanged.

The new reproduction wrapper is orchestration outside the frozen computation. It verifies source seals and executes the existing runners and auditors unchanged. Dry-run verification needs only Python's standard library.

```sh
python -S research/boolean_coupling/reproduce_tail.py --verify-only --history --plan
```

Verification does not load weights, rerun the omitted raw arrays, or count as model replication. Raw arrays stay local with sizes and hashes inventoried. Article drafts are absent from this release.

## Failures remain failures

Original 226 failed its first reference replay before interventions because of a boundary indexing bug. Original 227 stopped at a raw coordinate bridge in its second case. Their original sources and failure records remain alongside the corrected/narrowed versions.

In 227b,31/32 raw final-coordinate bridges to226b's differently placed intervention still fail 1e-4. The .005 two-score bridge passes. Two raw ideal-write comparisons also fail 1e-4. Local arithmetic closure is separately reported and does not certify the failed final bridge. None of the follow-ups recertifies the original whole-block derivative operator.

Question B remains open. All 32 contexts were previously observed and dependent; no general False-only predictor or learned circuit is established.
