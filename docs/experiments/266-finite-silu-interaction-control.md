# 266 — does the sparse rule preserve the local interaction of finite SiLU?

Retrospective control motivated AFTER265's local and continuation outcomes are
known. The pointwise local interaction is not sufficient for the global policy
contrast; removing it slightly strengthens that contrast. Moreover265 shows
substantial changes when masks differ across corners. Do not attribute those
changes to the native model without checking the fully finite activation.

At all224 old H/L incoming points, reconstruct the SAME L/P/D/H normalized
inputs and learned gate/up projections as265. Compare three products: adaptive
ranked1% SiLU, common-H-mask ranked1% SiLU, and actual finite SiLU at ALL neurons.
All learned down projections use the original full-sequence publicFloat32 shape.
Keep the same source/prefix, weights, input points and corner definitions.
This adds primitive tensor evaluations only: no complete forward, new native
trajectory, fresh forecast or complete-response repair.

Require exact old H/L gates/up/finitephi and exact local265 fixed/adaptive mixed
write replays0. Retain CPU64 down-primal comparisons at their original1e-4
threshold, including failures. Finite product/head calculations are explicit
component interventions, not a new whole-model reconstruction. Old derivative
failures are neither used nor recertified.

PRIMARY: Does the adaptive sparse mixed-write vector approximate the fully
finite mixed-write vector with relative error<=.25 and cosine>=.95 in>=90% of
eligible whole inputs, separately H/L? Eligibility: finite mixed-write norm>=.001.
Keep every input, all ineligible records and verdict scopes. The common-mask
comparison is a separately reported secondary, not a fallback primary success.

Also describe, without a newly invented success threshold, cosine between the
same-input direction effects W_D-W_L and W_H-W_P for all three conventions.
Report their complete vectors and both whole/verdict statistics at every block.
At block28 retain finite-head score margins on x+W_corner for every corner and
mode. Do not add or linearize final margins. Save full vectors locally with a
public inventory and all numeric records/decisions; no case selection.

Even local success does not make265 a native learned-circuit account: the
normalization-corner interventions are defined decompositions and the inputs
are still saved approximate trajectories. Failure would show that native
forecast accuracy does not guarantee fidelity of this derivative-like
counterfactual contrast. It would constrain interpretation of sparse-policy
interaction, not invalidate258's recorded global forecasting criterion.
