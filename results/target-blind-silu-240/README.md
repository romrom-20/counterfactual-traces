# Target-blind SiLU restoration: experiment 240

A False/forecast-derived fixed 1% mask recovers MLP16's omitted whole-sequence SiLU contrast in **32/32 old contexts**. Direct activation restoration lowers the original generator-contrast error **45.2% → 2.94%**. Verdict-only recovery of the smaller omitted term passes **16/32**.

The finite behavioral rule does not establish a stable consequential account: only **3 native-input cases** and **0 forecast-input cases** exceed the frozen .10-nat guard. Full finite continuation from the forecast input still has **25.2% native displacement error and 1.09 nats margin error**. No native-state prediction, fresh confirmation or learned circuit is claimed.

[Report](../../docs/analysis/240-target-blind-silu-result.md) · [Frozen protocol](../../docs/experiments/240-target-blind-silu-tail.md)

## Evidence

- `design.json`, `sources/`: pre-execution source hashes and exact source copies; freeze commit `3affecb`.
- `mask-seal.json`, `masks/*.json`: all 32 target-blind mask/input seals, completed before the intervention phase opens native receiving inputs.
- `status.json`, `cases/*.json`: every direct activation intervention, finite suffix replay and native-reference check.
- `audit.json`: all 1,568 independent numerical comparisons and 384 unrelaxed Float64 bridges; all pass.
- `local.csv`: both whole and verdict missing-tail recovery, including failures.
- `final.csv`: all selected and three row-count-matched random controls, both incoming states, all materiality decisions.
- `native-errors.csv`: absolute errors relative to original native True, distinct from the same-input local reference.
- `masks.csv`: selected budgets and row counts, plus changes to the original239 generator contrast.
- `input-recheck.json`: post-run 128 raw-prerequisite hash comparisons against prior release inventories, all pass. This integrity check does not change predictions or qualify a new derivative.
- `logs/`: exact execution output for every phase.
- `local-array-inventory.json`: all 65 raw activation/weight arrays by size and SHA256; arrays remain local, not in Git.

## Reproduction

Install the pinned environment from the [research guide](../../research/README.md). Model work requires Apple Silicon/MPS and the pinned cached checkpoint.

```sh
python -S research/boolean_coupling/reproduce_silu.py --verify-only --history --plan
python research/boolean_coupling/reproduce_silu.py --run --offline \
  --input-239 runs/existing-239 --out runs/my-silu-test
```

Omit `--input-239` to regenerate all fixed prerequisites. Outputs must be new. Unexpected failures stop and retain their evidence. Model-free verification checks archived file integrity; it does not reproduce omitted arrays or establish numerical replication. New raw outputs are assessed with the frozen numerical criteria, not required to match archived compressed-file bytes.

240 adds 32 ordinary False forwards, 32 source-only blocks, 32 mask-preparation MLP calls, 384 intervention MLP calls, 4,608 suffix blocks and 384 finite heads. No new ordinary True forward. Native receiving inputs remain supplied oracle diagnostics; source rows have their declared pre-observation causal provenance. The mask is blind to native error within old data, not independent of the preceding investigation.

The private article draft is excluded. All older failures and qualified retrospective conclusions remain unchanged.
