# Experiment 288 evidence

[Explanation](../../docs/analysis/288-source-receiver-tangent-routing.md) ·
[Frozen protocol](../../docs/experiments/288-source-receiver-tangent-routing.md) ·
[Regeneration commands](../../research/README.md#source-and-receiving-response-maps-288)

All 32 old contexts are retained. Both primary sufficient accounts and all three
secondary accounts fail 0/32 joint cases. Matched finite functions pass the
4,384-record required numerical gate. Seven original 1e-4 finite-coordinate
diagnostics remain failed under the separately declared cached replay bounds.
The previous 287 failed numerical gate remains unchanged.

- `design.json`, `status.json`, `runtime.json`: frozen inputs, source hashes,
  complete cohort and work counts.
- `audit.json`, `checks.csv`, `derivatives.csv`: saved-array gate and every check.
- `intermediate_diagnostics.csv`, `native_coordinate_diagnostics.csv`: passing
  and failed narrow coordinate/trace diagnostics, with original bounds.
- `hypotheses.csv`: every account, vector sufficient statistics, signed finite
  predictions, materiality and case decisions.
- `effects.csv`, `ledger.csv`: all eight finite functions and all seven signed
  three-factor interaction terms per case.
- `native_scalar.csv`: secondary scalar calibrations against the old T-minus-M
  target. These use M's margin baseline and do not claim native-vector fidelity.
- `cases/*.json`, `scores.json`: raw metadata seals and actual finite candidate
  logits, including the two FD steps. No gradient vectors or weights.
- `omitted-array-inventory.json`: original archive hashes, sizes, shapes/dtypes.
- `cpu-qualification/`: final tiny unequal-length tests, all-coordinate checks
  and three rejected tamper-fixture outcomes.
- `execution-notes.json`, `release-manifest.json`: costs, retained qualifications,
  scientific source seals and exact export inventory.

Run `python -S research/boolean_coupling/verify288.py` from the repository root.
It verifies records and sufficient-statistic arithmetic, not omitted vectors or
independent checkpoint replication. General Question B remains open.
