# Local MLP mixing, state dependence and sparse-rule fidelity

Read the [result and interpretation](../../docs/analysis/265-266-local-mixing-and-state-feedback.md).
Four old dependent contexts;224 saved input points;28 new full forwards.
General Question B remains open. No new fresh predictor or learned circuit.

- `local-mixing-265/`: all same-input gate/up/product components, 224 exact
  old-MLP replay controls and3,824 checks. Cross-product-only primary fails:
  3/88 H and2/89 L whole inputs pass. All guard-excluded and secondary rows kept.
- `local-mixing-continuation-265/`: all28 forwards, per-layer inputs/writes and
  installation hashes,16 complete outcomes, finite-head margins and16,952
  audit checks. Local-interaction removal0/4 sufficiency passes under adaptive
  and common-mask variants; random0/4. All remain4/4 native-accuracy passes.
- `finite-silu-interaction-266/`: full-finite activation control at the same224
  inputs. Adaptive sparse mixed-write fidelity0/177; common-mask control175/177.
  All whole/verdict comparisons, same-input alignments, finite margins and1,120
  exact replay checks. This does not retroactively change258's forecast result.

The raw CPU64 failures remain failed at1e-4:120/3584,115/5712,141/2688 in the
respective stages. Matched full-shape publicFloat32 controls and explicit
rounding vectors qualify the stated scope. Older Jacobian disputes are open.

`raw-array-inventory.json` lists all omitted local tensors by hash and size.
`manifest.json` binds every published record, protocol/source and figure.
Scientific sources were frozen before their calculations; plotting/reporting
are posthoc. Large arrays, model weights, private writing and local continuity
notes remain excluded.

Use [reproduction commands](../../research/README.md#local-product-mixing-and-finite-activation-control)
for actual model/tensor work. `python -S research/boolean_coupling/verify_transmission.py`
verifies the public release with the standard library. It is not independent
model replication.
