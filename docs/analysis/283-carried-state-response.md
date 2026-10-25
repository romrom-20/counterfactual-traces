# Predicting attention's modulation of the carried correction

**The frozen carried-state response rule passes its joint test on all 32 old
contexts, including all 16 dependent schema pairs.** Evaluating the response of
the missing-product map at the carried state predicts a smaller attention-induced
change that the carry-only rule missed. Gate-only, up-only and matched random
increments fail the new conditional local-increment test. Native accuracy alone
does not discriminate these explanations: all of those controls already pass the
original native-response criterion.

This is a qualified **retrospective** result about a specified approximation's
correction at its own incoming state. It is not a complete account of the native
source generator, a learned circuit, an efficiency result or a solution to general
Question B. The earlier carry-only primary remains failed. [Frozen protocol](../experiments/283-carried-state-response.md)
· [Evidence and checks](../../results/carried-state-response-283/README.md).

## The question left by the previous test

[Experiment 282](282-source2-gate-origin.md) found that carry from the first source
block generates most of the second source MLP's missing correction. Adding that
carry correction restored the native-response test, but failed every paired
schema-write contrast. Attention's standalone correction around the False state
was tiny; its interaction with the carried state was appreciable. Adding the two
standalone corrections did not recover that interaction.

The new hypothesis is more specific than “attention matters”: predict the
missing-product map's response to the attention step **at the carried operating
point**, and install its gate and up contributions separately. This tests a local
generated update. It does not explain why the fully native computation chooses
its complete displacement.

## The fixed rule and its information budget

F is native False. B uses False-coefficient cubic SiLU at source MLP1 and MLP2.
C1 retains the cubic only at source MLP1; T is fully native True. Every other
source activation, all receiving computation, attention, normalization,
projections, products and final readout remain finite.

At source MLP2, let xF be the native False post-attention midpoint and c the
post-block1 source-state change produced by B. The additional finite corner is
`xC = Float32(xF+c)`. Let xU be B's actual MLP2 midpoint and `v = xU-xC`.
This step is available from B's own computation; it includes the attention-write
change and the specified local rounding. No full-current source2 activation or
native True trajectory supplies the candidate correction.

At a corner x, R(x) is the difference between finite and False-cubic **Float32
neuron products**, subtracted in Float64. R is an approximation discrepancy, not
the complete native MLP write. Write `eC = finiteSiLU(gC)-cubic(gC)` in Float64.
The fixed response is

```text
dz = normWeight * (v/s - xC*mean(xC*v)/s^3)
dg = Wgate*dz; du = Wup*dz
gateDelta = (SiLU'(gC)-P3'(gC))*dg*uC
upDelta   = eC*du
P correction = R(xC)+gateDelta+upDelta
```

Here s includes RMSNorm's epsilon. P3 is the smooth cubic defined by the observed
False coefficients. The formula is the ordinary product rule after a normalized
input change: the gate-remainder changes, and the existing remainder is
reweighted by the changed up branch.

The derivative is explicitly that of an **anchored smooth surrogate**. Its
normalizer is Float64; gate/up maps are anchored at observed gC/uC, and its
remainder is anchored at observed eC. It is not the Jacobian of rounded public
Float32 execution or of the complete block. R(xC) retains its observed product
rounding and is not replaced with `eC*uC`. Actual intervention products are
`Float32(Bproduct+correction)`.

All interventions use B's same actual midpoint, gate and up values. K installs
R(xC); P installs both response terms; H installs only the gate term on top of K;
J only the up term. G0–G2 retain K and add seeded random output-write increments
matched to the actual local P-K norm. FZ is the exact False-zero control. Only
after these candidates are run do C1 and the full-current correction become
available. U installs that full correction and must equal C1 exactly. N removes
the predicted gate-plus-up increment from full calibration. N is retrospective,
not a forecast. T is evaluated last.

The budget includes one extra finite corner per case and its staged derivative,
in addition to finite computation through all 26 later source blocks and all
receivers. This is not a compact predictor or a demonstrated speedup over the
[causal-cache baseline](272-research-review-and-causal-baseline.md).

## What discriminates the rule

The primary requires five criteria jointly: the original native residual/verdict
and finite-margin bounds; the full C1-B local write; its actual continued
receiving/verdict/margin effect; the paired descriptive-minus-generic C1-B local
write contrast; and the new **C1-K local increment**, predicted by P-K. The local
increment is R(xU)-R(xC), including the small standalone attention remainder;
it is not 282's mixed-only target. Vector targets retain the .001 norm guard,
25% relative-error and .95 cosine bounds. The native vector bounds are 5%/.99
and the native margin bound is .10 nat. Full-effect margin eligibility remains
.02 nat. The joint threshold is at least 24/32 and 6/8 in each schema/order stratum.

- P passes all five criteria **32/32**, with **8/8 in every stratum**. Its paired
  local contrasts pass **16/16**.
- H, J and every random increment pass the original native criterion **32/32**,
  but the conditional C1-K local-increment criterion **0/32**. Their joint counts
  are all zero. H nevertheless passes the paired contrast 16/16; that contrast
  alone does not establish the need for both terms.
- K passes native, full local and full continued-effect criteria 32/32, but the
  paired contrast 0/16 and conditional increment 0/32. Its joint count remains zero.
- C1 and full U pass jointly 32/32 and are exactly equal. N passes the original
  native test 32/32 but the conditional increment and joint test 0/32. Baseline B
  passes the native test 16/32 and the joint test 0/32.
- T passes its native criterion 32/32 but the conditional increment 0/32. It has
  a finite first source activation, so its MLP2 input differs from C1's. **T is
  not the positive control for this conditional C1-K target.** C1/U are.

![Carried-state response, every case and the paired target](../figures/carried-response/283/carried-response.png)

P's median relative error on the new local increment is **12.29%**, cosine
**.99251**, compared with 48.48% for gate-only H and 78.96% for up-only J. Random
increments have about 140% error. On the complete missing local write, P's
median error is 1.54%; on its paired schema contrast, 3.36%, compared with K's
38.37%. This is the discriminating gain: a response evaluated at the carried
state predicts the otherwise missed local modulation.

A separate **posthoc saved-forecast control** limits how specifically to interpret
that success. A donor's P-K write from the other literal, with schema, assignment,
order and target held fixed, passes the old local-increment tolerance **32/32
without rescaling**: median error 17.62%, cosine .98438. Matching its norm to the
recipient's own forecast also passes 32/32. Conversely, switching schema while
holding the other fields fixed passes 0/32, with or without norm matching.
[Every donor comparison and provenance](../../results/carried-state-response-283/transport-posthoc/summary.json).

The analogous retrospective comparison on 284's fresh cases passes 30/32 without
rescaling, versus the own rule's 32/32 and finer precision. Much of the direction
therefore transfers across literals within the tested matched backgrounds; random
directions are not enough to establish case-specific necessity. This neither
falsifies the frozen primary nor identifies a learned feature. These are saved
vector comparisons, with zero model forwards and no donor installed. The 128
donor rows reuse 32 recipients in 16 dependent pairs; they are not independent
examples or a new fresh prediction test.
[Reproduce the saved-array diagnostic](../../research/README.md#posthoc-carried-write-portability)
without overwriting published records or running new model forwards.

Native prediction remains bounded and conditional on the rich finite budget.
P's median native receiving-vector error is .230%, verdict-vector error .177%,
and actual finite-margin error .00281 nat. K already has .452% receiving error
and .00433-nat margin error and passes the same native criterion in every case.
Therefore native success by itself would not identify the added response rule.

## Finite interventions and the behavioral boundary

The ledger measures the signed vector effects of actually installing H-K and
J-K, their nonadditive difference from P-K, and the unpredicted C1-P remainder.
Every case closes within 1e-10. Median projections onto the conditional target
are approximately .691 gate and .288 up at the local write, .696/.279 at all
receivers, and .790/.193 at the verdict row. Installation/nonadditivity and
prediction remainders remain recorded. Separate medians need not sum to one;
these projections are not causal fractions or identified learned features.

The stronger actual-increment and removal accounts **do not pass**. No case has
a C1-K margin effect reaching the frozen .02-nat guard; the largest absolute
effect is about .00411 nat. Eligibility is **0/32, including full C1/U calibration**.
Consequently every strict incremental continued-effect account and the removal
account is 0/32. We retain those failures rather than relax the guard. The test
supports a small vector correction and its propagation, not its universal
necessity for the native verdict or a material margin effect under this rule.

Paired whole-token receiving contrasts remain unavailable because the schemas
have different sequence lengths. The paired primary concerns source-write
coordinates only; no token alignment has been invented.

## Numerical qualification

The original frozen 283 audit passes **5,408 required checks**. It independently
checks 64 False anchors and 192 derivative-coefficient vectors, and all 32 new
staged derivative anchors using Double automatic differentiation and central
differences at 1e-4 and 5e-5. The coordinate bound stays 1e-4. It also verifies
32 public corner maps, 480 public down maps and 448 finite full-vocabulary heads,
with exact public replay requirements and the separately declared CPU64 head bound.

**Nine of 224 raw comparisons still fail their original 1e-4 bound, all on up
projections.** They remain failed diagnostics under the predeclared public
Float32 and dimension-derived rounding-envelope qualification. This does not
independently prove the GPU reduction, relax required norm/gate/SiLU checks,
recertify older failed raw-logit or whole-block-Jacobian checks, or establish a
Jacobian of rounded execution. Separate audit code within this investigation
is not an independent research team.

Before freeze, the first unsealed synthetic CPU run exposed an audit assertion
that conflated local-vector eligibility with downstream-margin eligibility.
Only that assertion was corrected; the failed test was retained. The complete
unequal-token CPU producer and separate audit then passed before any real
checkpoint calculation. Original source and evidence seals remain immutable.

## Reproduction and the unresolved question

The producer completed 448 full forwards in 102.21 seconds, plus 32 finite corner
maps, 64 local down maps, 32 normalizer JVPs and 64 CPU64 learned gate/up
matrix-vector products. The separate audit took 17.11 seconds; audit and fixture
work is additional. New raw arrays occupy 31,051,716 bytes, with checked lossless
aliases to earlier identical arrays. No old evidence was deleted or recompressed.
The source/protocol freeze is `463ece70233d28e2f3bc6d2ae6d08e4e62880817`.
[Reproduction](../../research/README.md#carried-state-response-283284) regenerates
the model arrays; the evidence package preserves every arm, endpoint, failed
diagnostic and denominator.

Fresh prediction with this exact rule was sealed before observing its targets
and is reported separately in [284](284-carried-state-response.md). This result
uses one small checkpoint and 32 old dependent contexts. Historical singleton
grounding controls failed, so it is not a competent-verifier shutdown claim.
It explains how attention modulates a defined cubic correction at a carried
state; it does not yet explain the complete native learned computation, identify
semantic features or establish novelty priority.
