# Receiving nonlinear-response evidence290

[Explanation](../../docs/analysis/290-receiving-nonlinearity.md) ·
[Protocol](../../docs/experiments/290-receiving-nonlinearity.md) ·
[Reproduction](../../research/README.md#attention-and-receiving-response-289290)

GatedMLP-only01 passes 32/32; normalizer-only10 passes 0/32. All-E11 is a
32/32 calibration, not a discovery. Only 16 remaining finite contrasts reach
materiality. All cases remain. This is conditional on edited source/attention/
head anchors, not native-response forecasting or a learned circuit.

All 9,280 required records pass. Eight narrow1e-4 finite-coordinate failures
stay FAILED under the separately declared cached replay bound. Original prior
failures and source seals remain unchanged.

Files follow the same layout as289: design/status/runtime/audit, complete
checks/derivative/primitive CSVs, original narrow diagnostics, all hypotheses,
receiving vectors, old native scalar calibration and signed interaction ledger.
Case metadata and actual candidate logits are public; norm weights, parent and
gradient vectors are omitted with hashes/sizes/shapes and reproduction commands.
CPU qualification includes256 source-coordinate FDs and three rejected tamper
fixtures. Public CI checks records, not independent checkpoint replication.

Run `python -S research/boolean_coupling/verify290.py` from the repository root.
