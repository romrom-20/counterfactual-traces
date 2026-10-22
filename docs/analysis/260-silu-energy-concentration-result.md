# Sparse local capture includes concentration before cancellation

**Retrospective result:** the original ranked 1% mask contains at least 90% of
individual omitted-write energy in **20/24 whole-input witnesses**, meeting the
frozen 18/24 criterion. The median is **97.73%**. The same threshold holds in
only **16/24 verdict-row witnesses**, with a 94.69% median and minimum zero.
The strong whole-input result does not imply uniform verdict-local capture.

[Protocol](../experiments/260-silu-energy-concentration.md) ·
[All records and checks](../../results/silu-energy-concentration-260/README.md) ·
[Source](../../research/boolean_coupling/energy260.py)

## The ambiguity this tests

258 already shows a target-blind mask can produce an accurate final forecast
under the declared rich budget.259 shows curvature/gate-edit information helps
select its coordinates. But a sum of vectors can be small because large terms
cancel. Does sparse recovery merely exploit that sum, or are the omitted
individual write contributions concentrated in the selected coordinates?

At every saved C/S MLP input, let `c_j` be the omitted SiLU-tail times the current
up value, projected through its learned down column. Compute `sum_j ||c_j||²`
and its selected/unselected partition. Separately compute the summed vectors
G/V/U and `||G||² = ||V||² + ||U||² + 2<V,U>`. Whole-sequence positions occupy
distinct coordinates; cancellation is within each position's residual vector.
The mask, selected budget and inputs remain exactly those from258.

## Complete outcomes

All 24 whole witnesses are eligible. Twenty meet the 90% individual-energy
threshold; four miss it. The minimum selected fraction is **89.69%**. Median
summed-vector coherence `||G||² / sum_j ||c_j||²` is **1.0982**. Selected and
unselected summed vectors have a negative cross term in 14/24 witnesses; the
median cross term divided by `||G||²` is **−.0000378**. These are medians of
different quantities, not an additive attribution of behavior.

All 24 verdict witnesses are eligible. Sixteen meet 90%; the selected fraction
can be zero even though the final258 forecast passes. Median coherence is
**1.0326** and the cross term is negative in 4/24. Local capture and final native
prediction remain distinct requirements. C/S block1 inputs coincide; the24
witnesses are not independent examples.

This supports a concentration account for the whole receiving write before
cross-neuron cancellation, under these coordinates and input states. It does
not establish that cancellation is generally absent, that selected neurons
represent a learned feature, or that this measure is basis invariant. It does
not explain why the upstream learned computation generates these gate edits.

## Scope and numerical qualification

The diagnostic uses all four saved contexts at blocks1/14/28, C/S forecast
states only. It is frozen after258/259 outcomes but before these energy
statistics. There are **zero model forwards, new native measurements or new
global primitive interventions**.

All 192 checks pass, including original-mask identity, complete/selected
component bridges at the original 1e-4, vector closure 1e-10 and energy identities
1e-8. This inherits258b's qualified Float32-component convention. The original
raw whole-down and QK failures, and older derivative disputes, remain failures.
No new upstream prediction, speedup, learned circuit or complete answer to
Question B follows from this local result.
