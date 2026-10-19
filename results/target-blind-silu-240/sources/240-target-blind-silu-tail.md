# 240: does a forecast-derived SiLU mask identify a consequential missing term?

Freeze code and this protocol before calculating masks or intervention outcomes.
Retain the32 old228 contexts. This is a retrospective primitive diagnostic at
MLP16 only, not another end-to-end correction sweep or fresh prediction.

239's remaining whole-sequence contrast error concentrates at the first
receiving row and in a small target-dependent set of SiLU terms. Test whether
that concentration can be recovered without observing the native error, and
whether the recovered terms matter to a finite final response.

## Selection before native outcomes

From the ordinary False computation retain MLP16's public gate g0, activation
phi0 and up value. Compute gates g1/up u1 at238's installed own4 MLP16 input.
This forecast is dependent on old-context site selection; the new mask must not
read native inputs, native errors, final margins or target-dependent readouts.

For dg=g1-g0 let m be the point of the closed interval[g0,g1] nearest zero,
p=sigmoid(m), and define

`B = p(1-p) [3+max(|g0|,|g1|)] |dg|³ /6`.

This bounds the scalar second-order SiLU Taylor remainder: SiLU'''(g) equals
`s(1-s)[3(1-2s)+g(1-6s+6s²)]`, and the bracket magnitude is bounded by3+|g|.
Rank all receiving neuron-position coordinates by
`B * |u1| * ||W_down[:,neuron]||` descending, flat index tie-break.
Select ceil(1% of all coordinates), matching239's descriptive budget. No fitted
readout or native gate is used. Three controls sample the same number of
coordinates *in each receiving row* uniformly, with SHA-derived seeds.
Seal every mask, input hash and source hash across all32 cases before opening
native receiving inputs or executing intervention outcomes. Do not tune the
budget after seeing results.

## Direct activation intervention and finite continuation

Two supplied incoming states: native MLP16 input (oracle diagnostic), and238's
installed own4 MLP16 input. Prefix/source rows come from the declared causal
source-only computation; receiving inputs are installed exactly as saved.
At each state execute the public MLP, with finite normalization and up/product.
Compare six activation arms: full SiLU, False-anchored quadratic SiLU,
quadratic with exact public SiLU restored on the sealed mask, and three matched
random-mask restorations. Replace the actual activation before multiplication,
not an allegedly equivalent terminal residual. Prefix/source activations stay
unchanged. Continue blocks17–28 and final norm/head finitely.

The quadratic is `phi0 + phi'_0 dg + .5 phi''_0 dg²`, using public Float32 phi0
and CPU64 analytic derivatives at the public False gate, cast only on installation.
This small anchor-rounding difference from239 is recorded explicitly.

Primary local test: restoring the mask recovers the native-minus-own missing
SiLU-tail contrast, with relative remaining error<=.25 and recovery cosine>=.95
on the WHOLE receiving sequence, target norm>=.001. Require24/32 and6/8 in each
schema/order stratum, and publish verdict scope separately. Also report how the
restoration changes239's original full generator-contrast error; do not replace
the original239 outcomes.

For each supplied input, the reference final state/margin is that same input's
full-SiLU finite continuation. This isolates a local model discrepancy; the own
reference is not a successful native forecast. Materiality requires the quadratic
reference gap>=.10 nats AND verdict-vector gap>=.001. A final restoration passes
only if material, both gaps fall by>=50%, and both are strictly smaller than all
three matched random controls. Report counts out of all32 (and eligible counts)
for native/own separately; a stable behavioral account needs24/32 and6/8 in
each stratum for each state. All ineligible cases remain in the denominator.
Report whole-vector gaps and absolute errors versus original native True too.

## Numerical qualification and limitations

False and source/own replay at1e-4, native full final replay at1e-4, scores .005.
Independently reconstruct masks, bound coefficients, installed activation/product
edits and local outputs; Float32 reconstruction at1e-4. Preserve raw Float64
bridges at the same bound separately. Check analytic first/second SiLU
coefficients against independent CPU64 autograd at1e-10. Recompute all final
two-token scores from saved final residuals at.005. No new block derivative is
used or recertified. A failure stops and is retained without tolerance changes.
Publish all32 outcomes, counts, hashes and any failures. These are conditional
primitive interventions, not learned features/circuits, and cannot establish
prediction of the unknown native receiving input or solve Question B globally.
