# 276: old-context first source MLP test

The shifted-sensitivity rule passes32/32 joint native + paired-contrast predictions.
The stronger registered gain explanation FAILS8/32. These are separate decisions.
This uses old275 native source-attention operands and is retrospective.

Read the [result and limitations](../../docs/analysis/276-277-first-source-mlp.md),
[frozen protocol](../../docs/experiments/276-first-source-mlp-gain.md), and
[reproduction instructions](../../research/README.md#first-source-mlp-276277).

- `design.json`, `status.json`, `runtime.json`: exact inputs, frozen source hashes,
  checkpoint and computation counts.
- `native.csv`, `paired.csv`, `combined.csv`: every original and added endpoint,
  including all main/mixed/random/self controls.
- `local-gain.csv`, `gain.csv`, `explanation.csv`: separate local, downstream and
  joint causal-account outcomes; material-margin failures remain failures.
- `derivatives.csv`, `raw-primitives.csv`, `heads.csv`, `audit.json`: qualification,
  original raw bounds, finite heads and independent CPU64 derivative checks.
- `raw-array-inventory.json`, `cases/*.json`: local-array hashes and provenance;
  activation arrays/checkpoint weights are omitted and reproducible.
- `manifest.json`: scientific source and all-case publication integrity.

No fresh confirmation in276, learned circuit, efficiency or general Question B
solution. Earlier numerical failures are not recertified. Article drafts excluded.
