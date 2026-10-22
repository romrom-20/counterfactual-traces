# 265 — local MLP mixing versus changing computational state

Retrospective, SAME four old263 contexts. No fresh examples or novelty/learned
circuit claim. 264b shows that the final effects D-L and H-P differ; they come
from evolving own-state policies, not a same-input primitive factorial. Test
whether immediate local nonadditivity is sufficient to account for that global
interaction, or whether substantial dependence remains after it is removed.

## Frozen maps and numerical scope

Retain263's False cache, cubic receiving queries, ranked1% finite SiLU, finite
source/prefix/K/V/softmax/projections/products/head. At the SAME current MLP
input x, construct normalized inputs L, L+P, L+D, L+P+D, named L/P/D/H. Evaluate
all gate/up projections at the original full-sequence shape, publicFloat32.
For each corner, compute the unchanged258 quadratic/ranked SiLU rule.

Two mask conventions: adaptive uses each corner's own ranked mask; fixed uses
the H-corner mask for ALL four corners at this same input. Fixed does not mean
constant across layers or contexts. H itself is identical under both.

Compare exact reconstructed old H/L corner gates, up values, finite SiLU,
adaptive product and down write against saved263 nodes at EVERY layer/context.
Require zero error for these matched-full-shape publicFloat32 replays. Retain
all CPU64 projection primal comparisons with their original1e-4 passes/failures;
they are not required to pass and will not be relabeled. No whole-block
derivative is used. New receiving RMS tangents/cubic queries in the continuation
must retain independently checked smooth AD/FD and exact installed-input checks
at the263 bounds. Older derivative disputes remain unresolved.

## Stage1: same-input local primitive diagnosis

Use all28 saved MLP inputs in BOTH H (accurate forecast) and L (inaccurate
forecast) in all4 old contexts,224 input points. Save both complete mixed-write
vectors, their mask contrast and signed components, whole and verdict scopes.

For a fixed common mask, let phi_c/up_c denote Float32 used-SiLU/up at corner c.
The ideal product mixed difference has the exact decomposition

    gate_side = (phi_H-phi_P-phi_D+phi_L)*up_H
    gate_up_cross = (phi_P-phi_L)*(up_D-up_L)
                    + (phi_D-phi_L)*(up_P-up_L)
    up_nonadditivity = (phi_P+phi_D-phi_L)*(up_H-up_P-up_D+up_L).

Keep Float32 product-rounding and public-down-projection arithmetic differences
as separate measured vectors. Project the defined components with CPU64 learned
down weights; check component closure1e-8, product identity1e-9 and reconstruction
of actual public mixed writes with the explicit measured rounding terms1e-8.
Gate-side includes any measured gate-input arithmetic nonadditivity; do not call
it pure SiLU curvature or a feature. Publish raw gate/up nonadditivity too.

PRIMARY LOCAL hypothesis: gate_up_cross alone accounts for fixed-mask mixed
write within25% relative vector error and cosine>=.95 in>=90% eligible whole
inputs, separately H/L. Eligibility is mixed-write norm>=.001; retain and report
all ineligible records. All other components/scopes are secondary, not fallback
primary successes. This is a defined coordinate decomposition, not unique
mechanistic percentages of the native response.

## Stage2: direct product interventions in complete continuation

Seven full-forward arms F/Z/H/A/K/R/T,4 old contexts,28 forwards total.
F ordinaryFalse; H replay of263H; T ordinary nativeTrue; Z zero edit with A.
At each receiving MLP's OWN incoming state, calculate the four corner products:

- PRIMARY A replaces the H product with Float32(product_P+product_D-product_L),
  using each corner's adaptive mask. This removes pointwise mixed nonadditivity
  of the entire receiving MLP policy while preserving its changing input.
- K does the same with the common H mask, separating mask-choice dependence.
- R keeps the H product, then subtracts a seeded random receiving WRITE. Its
  per-row norm matches the actual public H-to-A down-write change at R's own
  current input. Seed SHA256(case:layer:row:local-mixing265). Report this as a
  matched-write control, not a matched-neuron-product control.

All operations before the edited source and at the source remain untouched.
No observed native receiving state is an intervention input. H/F/T trajectories
and scores must replay263 EXACTLY; Z/F exact. Final head remains finite.
Nonfinite or>100x False tail norm stops the affected A/K/R arm and counts as
failure; all other failures stop the study. Save every layer's receiving raw
input, write, tail and score, all installation hashes/norm controls. Keep raw
output below300MiB; never delete old evidence or overwrite any input.

Let r_add=r_P+r_D-r_L at the final layer, from the old263 actual policies. Its
margin is computed by the FINITE head on r_add, not by adding scalar margins.
Global target interaction is I=r_H-r_add. For each new arm N in A/K/R, test
whether r_H-r_N predicts I within25% error and cosine>=.95 at whole AND verdict,
I norms>=.001. Require |margin_N-margin_add| <= .005 + .25*|margin_H-margin_add|.
PRIMARY GLOBAL sufficiency requires A>=3/4 joint passes AND randomR<=1/4.
K is secondary, all arms/cases reported. Also report original native5%/.99/.10
accuracy and signed native changes for H/A/K/R without replacing this criterion.

This tests sufficiency of a defined pointwise product interaction for the global
policy contrast. Failure leaves changing inputs/attention/later MLP sensitivity
as an unresolved source of the contrast; it does not by itself identify which
one. Success still uses substantial finite work and is not a fresh predictor.
Do not interpret this as solving Question B, a unique learned circuit, or an
independent replication. All original261–264 failures remain intact.
