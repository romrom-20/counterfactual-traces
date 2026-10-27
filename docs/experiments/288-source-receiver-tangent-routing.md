# 288: source transmission versus receiving response

## Status and scope

Freeze this protocol and all enumerated code before checkpoint execution.
This is a retrospective causal diagnostic on all 32 already measured 285–287
contexts, not a fresh prediction test. Four layouts have eight cases each;
schema pairs are dependent. No native True gradient or new True observation is
needed. Preserve 287's failed original numerical gate and its narrower posthoc
287b qualification. This experiment cannot recertify either one.

287 changed the residual after block 1. Its source-only initial state (MF) almost
recreates the sensitivity at the fully edited forecast, while later computation
is always allowed to adapt. That does not distinguish a changed source tangent
map from a changed response to messages. Nor does it separate the final head.

## Finite functions, not backward overrides

Call the ordinary False trajectory F and the source-only edited trajectory E
(MF in 287). At this boundary E has the forecast M source row and False rows
strictly after it. The causal prefix stays False. Both trajectories execute
the remaining 27 blocks with finite attention, normalization, MLPs and head.

At layer l write the causal block update as
`s' = a_l(s)` and `r' = b_l(s,r)`, where s is the source row and r all later rows.
For small displacements the derivative has the triangular form

```
ds' = A_l ds
dr' = B_l ds + C_l dr.
```

Each arm has three binary factors, source/receiving/head, in that order.
0 selects F anchors; 1 selects E anchors. With incoming displacement z, define
an actual finite block function by subtracting its anchor output:

```
z'_s = block(x_l^source + z)_s - x_(l+1)^source_s
z'_r = block(x_l^receiving + z)_r - x_(l+1)^receiving_r.
```

The same z enters both branches; their chosen output rows are concatenated.
Source causality is not imposed by detaching a value. It follows from the
native attention mask and is checked. The score is lifted to E's primal:
`m_E + head(x_final^head + z_final) - head(x_final^head)`.
Thus every arm preserves E's zero-displacement trajectory and score. In
conceptual residual coordinates that trajectory is `x_l^E + z_l`. Store z
separately to avoid repeatedly rounding that recentering addition. At zero,
ordinary library differentiation gives the selected A, (B,C), and head maps.
The eight arms are finite modified networks; finite differences apply directly.
They are broad architectural interventions, not learned circuits.

The receiving factor includes B: the derivative of source key/value construction
and its coupling into later rows, as well as C's receiving recurrence. A result
favoring this factor cannot distinguish a changed message encoder from changed
receiving queries or internal receiving MLPs. The source factor A refers narrowly
to source-residual-to-source-residual recurrence. Keep that distinction explicit.

Matched 000 and 111 recover F and E derivative/finite-displacement controls.
Each uses full-vocabulary model logits; candidate scores are false/true tokens
895/830. Head anchoring includes final RMSNorm, preventing it from silently
being attributed to receiving blocks.

## Inputs and tests

Use the original small `d = Float64(C_source)-Float64(M_source)` from 286/287,
including its saved Float32 residual-addition rounding. It is not the large
Boolean edit. Recompute one ordinary False observation and the known first-block
forecast with the False gate installed before SiLU, as in 286/287. No True
trajectory is an input to these tests.

For every arm save the source margin gradient g, an independently composed
blockwise triangular-VJP gradient, full source adjoint traces, its finite d
effect, actual head inputs/scores and finite differences along d plus two
seeded source coordinates at steps .05 and .025. This is two compositions of
the same library primitives, not independent primitive AD. Qualified projections
and boundary gradients do not amount to a full-Jacobian certification.

Primary target: `V=g111-g000`, `D=E111-E000`, where Earm is its finite d effect.
Two prespecified accounts:

- SOURCE: `g100-g000`, finite `E100-E000`.
- RECEIVING_AND_HEAD: `g011-g000`, finite `E011-E000`.

Case success requires norm(V)>=.001, vector relative error<=.25, cosine>=.95,
and signed finite-contrast error<=.005+.25*abs(D). Population support requires
at least 24/32 and at least 6/8 in every layout. Separately report abs(D)>=.02
materiality without dropping small cases from denominators. These repeat 287's
diagnostic bounds; no p-value or independent-population inference is intended.

Secondary accounts are receiving blocks alone (010), head alone (001),
SOURCE+RECEIVING_AND_HEAD, and the full eight-corner signed/Mobius interaction
ledger. Do not turn interaction fractions into learned causal shares. All eight
arms and all failures are reported. Old native donor-margin correction is a
secondary scalar calibration using the original M margin baseline, never the E
primal margin. It cannot certify a native residual forecast.

## Numerical gate before scientific interpretation

- Exact False and forecast source replay, gate/phi witnesses and immutable prefix.
- Every arm's zero offset is exactly zero; zero head scores match its anchor.
- Tape versus triangular gradient at the actual source intervention boundary:
  max coordinate error <=1e-4. All later-trace discrepancies are reported against
  the same bound as diagnostics, not silently recertified as passed.
- Matched 000/111 boundary gradients replay BOTH old cached and old full-shape
  FF/MF gradients at <=1e-4. Original raw full/cached failures remain failures.
- Matched cached primal replay <=1e-4. Finite matched native d displacements
  <=.002 coordinate and <=.001 score; report the original 1e-4 coordinate
  diagnostic separately. Same finite replay bounds as 286/287, no old bound edit.
- All 8*3*2 projected AD–FD checks <=.001+.02*abs(AD); step-halving differences
  <=.002+.03*abs(AD). Reconstruct from retained finite candidate scores.
- Each finite head's CPU Float64 reconstruction <=.005 logit coordinate.
- Noncausal source-output adjoints into later rows are exactly zero.

Any required gate failure saves a failed audit and all numerical records, and
prevents scientific tables. No automatic posthoc exception. The tiny unequal-
length CPU preflight checks all arms against full-tape and triangular algebra
and demonstrates that altered bounds, FD values and required checks are rejected.

## Cost, storage and interpretation

No new whole-model True call. All False, first-block, anchor, modified forward,
head, backward and triangular VJP calls are counted. Causal-prefix cached shapes
are intentional; matched full-shape comparisons use the saved 287 control.
This is not a speedup. Initial free space must be >=640MiB and per-case >=384MiB;
case arrays are capped at 128MiB total. No old experiment's guard is changed.

Even a clean split explains only how this particular small correction acquires
its changed sensitivity, conditional on two measured/forecast anchor trajectories.
It does not explain the learned semantic source direction, identify primitive
neurons, produce a compact False-only native predictor, establish novelty, or
solve general Question B. A fresh limited-budget test would require a separate
frozen rule after this diagnosis.
