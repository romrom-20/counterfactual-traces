# State-dependent source readout: 286 and numerical qualifier 286b

Read the [explanation](../../docs/analysis/286-state-dependent-readout.md) first.
All 32 cases are **old 285 contexts**, not new native forecasts.

The False-state gradient predicts 26/32 scalar native margins and identifies
none of the six large donor errors. The gradient at the False-derived M forecast
predicts 32/32 and identifies all six. Direct source-component interventions
separate those sensitivities. This is a scalar conditional diagnostic under
substantial finite suffix computation; no learned circuit or full Question B
solution is claimed.

- [Design and immutable source map](design.json), [runtime](runtime.json),
  [completion and array hashes](status.json), [original saved-array audit](audit.json).
- [Scalar predictions and all denominators](predictions.csv),
  [direct interventions and controls](interventions.csv),
  [finite two-candidate scores](scores.json), [source geometry](geometry.csv).
- [Projected derivative checks](projected_derivatives.csv),
  [required checks](checks.csv), [CPU64 heads](heads.csv),
  [192 original failed coordinate diagnostics](raw_coordinates.csv).
- [Descriptive source-adjoint traces](source_sensitivity_trace.csv) are not a
  causal localization of which block changes the readout.
- [Per-case metadata](cases/) bind actual False activation input/output hashes,
  prefix cache seals, native/forecast references and complete derivative records.
- [Full-shape numerical qualifier](full-shape-qualifier/audit.json) compares all
  source-row adjoint coordinates at every block; its original 1e-4 bounds pass.
  It does not erase the original cached-coordinate failures or independently
  validate every library derivative.
- [CPU qualification](cpu-qualification/), [execution notes](execution-notes.json),
  [omitted array inventory](omitted-array-inventory.json),
  [qualifier inventory](full-shape-qualifier/omitted-array-inventory.json),
  [complete release manifest](release-manifest.json).

The 19/32 material scalar successes do not rescue 285's failed 15/32 joint local/
vector/margin repair claim. Parallel projections are mathematically constructed;
their small norms do not identify sparse learned features. Random directions
match the F-parallel norm only. All original grounding, numerical, budget and
novelty limitations remain.

```sh
python -S research/boolean_coupling/verify286.py
```

This command verifies published scalar records, endpoint formulas, original
failures, source freezes and inventories with the standard library. It does not
load omitted arrays or the checkpoint. Use the
[reproduction guide](../../research/README.md#state-dependent-readout-286) to
repeat the model experiment on these now-known inputs.
