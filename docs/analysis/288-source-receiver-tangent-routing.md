# The source seeds the change; its local response does not explain it alone

**Question B remains open.** The previous initial-state factorial found that a
changed source can seed a changed downstream sensitivity. This test separates
the subsequent response maps. Neither source recurrence alone, receiving and
head response alone, nor their additive combination reproduces the defined
sensitivity shift in any of the **32 retained contexts**. Final-head response
alone also fails. These are broad computational interventions, not a learned
circuit or a new native-response predictor.

[Frozen protocol](../experiments/288-source-receiver-tangent-routing.md) ·
[Scalar evidence and array inventories](../../results/tangent-routing-288/README.md) ·
[Reproduction](../../research/README.md#source-and-receiving-response-maps-288)

![All accounts and signed interactions, with every context retained](../figures/tangent-routing-288/tangent-routing.png)

## The question left by 287

The source is the edited Boolean row, at the residual after block 1. Receivers
are strictly later rows. F is the ordinary False computation. E is 287's MF
state: the donor forecast's source row, with every other incoming row still
False. All later source and receiving computation is allowed to adapt.

287 found E approximately recreates the full forecast's source-gradient shift.
It did not distinguish changes in how a source perturbation propagates through
later source blocks from changes in how its messages affect receiving rows.
It also did not separate the final RMSNorm/readout from the receiving blocks.

Here the intervention is the **small** original C-minus-M source correction,
not the large Boolean edit. Both F and E anchors are generated from False,
the known edit, weights and the retained forecast. No new True trajectory or
True gradient is computed. The cohort is already observed: four layouts of
eight cases, with sixteen dependent schema pairs. This is retrospective.

## Hold the edited primal fixed, change the response maps

Causality gives each block the form

```
source'    = a(source)
receivers' = b(source, receivers).
```

Its derivative is triangular: source perturbations advance through A; receiving
perturbations receive B times the source perturbation plus C times their own.
There is no path from a later receiving row back into the earlier source row.

The test crosses three factors: source update A, receiving update (B,C), and
final head H. Each uses the F or E anchor, labeled 0 or 1. The eight arms are
000 through 111, source first and head last. **The receiving factor includes
source key/value encoding and coupling into later rows through B.** It is not
an isolation of receiving queries or receiving-token MLPs.

Each arm is a real finite function. Apply a displacement at the selected anchor,
subtract that anchor's block output, and express the resulting displacement
around E's trajectory. Do this separately for source and receiving output rows.
The head is recentered likewise. Every zero-displacement arm has exactly E's
primal trajectory and margin; the response to an added displacement differs.
There is no straight-through backward replacement. Finite differences test the
actual modified functions. Matched 000 and 111 recover the F and E controls.

Source-only 100 asks whether changing A suffices while (B,C,H) remain at F.
Receiving-and-head 011 asks the complementary question. Receiving blocks 010,
head 001 and the additive combination are secondary accounts. The primary
target is the complete gradient difference g111−g000 and its signed finite
correction-effect contrast. The joint case criterion requires 25% vector error,
.95 cosine and the frozen signed finite-effect bound; a population account
needs 24/32 and at least 6/8 in every layout.

## What failed, and how

Both primaries fail **0/32**. All three secondary sufficient accounts fail
**0/32** too. This is not just a few finite-margin misses: **every account fails
the vector criterion in every case**.

Median gradient-shift relative errors are 101.9% for source recurrence, 82.7%
for receiving plus head, 79.4% for receiving blocks, 97.5% for the head, and
126.4% for the additive account. Finite-contrast-only passes are respectively
5, 8, 13, 7 and 4 of 32. Twenty-three full finite contrasts reach the separate
.02-nat materiality guard; all nine smaller cases remain in the denominator.

The source/receiving-plus-head mixed vector is the negative of the additive
account's error. Its median norm is therefore 1.264 times the target shift's
norm. **That is an interaction/error ratio, not 126% of a mechanism explained.**
The seven signed three-factor interaction terms and every finite effect are
retained in the ledger. Norms alone cannot show their cancellation or identify
learned features.

Scalar accuracy can obscure this failure. Arm 011 passes the loose original
.10-nat native scalar-error criterion 32/32, but identifies only one of the six
large old donor errors and has median scalar error .0431 nat. Full 111 identifies
all six, with median error .00429 nat. The 110 corner, keeping both block-map
families edited but the head False, also identifies all six and has .00573-nat
median scalar error. The complete eight-arm calibration is in `native_scalar.csv`;
these secondary scalar comparisons are not native-vector successes. They use
the **original M margin baseline**, not E's absolute primal margin.

## Why a mixed term is possible

There is a simple architectural accounting, without naming a learned feature.
Let P(l) be the product of source A maps before layer l. Let q(l+1) be the
receiving adjoint propagated from the head through later C maps. Then the
source margin gradient is a sum over transmission sites:

```
g = sum_l P(l)^T B(l)^T q(l+1).
```

The source factor changes P. Receiving and head factors change B and q. Changing
both produces a product of their changes, which an additive account omits.
No matrices P, B or C were explicitly materialized or independently certified
here; this identity explains the derivative structure of the implemented test.

The algebra is standard. The empirical constraint is that neglecting the mixed
term fails badly in these contexts, even though the small correction itself is
locally well approximated at the appropriate edited anchor. Thus initial-source
sufficiency does not imply source-local-response sufficiency. The current data
also reject a final-normalization-only sufficient account.

This does **not** explain which primitive creates the consequential source
direction, isolate B from C, or identify source keys/values versus receiving
queries as the learned cause. It does not imply all three factors are necessary
for every scalar. A future causal test should target a specific primitive in
this conditional composition, with finite removal/control interventions; another
layer-error plot or a norm peak alone would not identify it.

## Numerical qualifications and actual work

The frozen saved-array audit passes **4,384 required checks**: 2,080 replay,
boundary-adjoint, source/prefix and head checks, plus 2,304 projected derivative
and step-halving records. Every arm's source gradient agrees across full-tape
and separately composed triangular block VJPs; all 6,912 intermediate source
trace diagnostics also agree. These use the same library primitives. They are
not independent primitive AD or a full-Jacobian validation.

Matched F/E gradients agree with both saved cached and full-shape 287 gradients
at the original 1e-4 boundary tolerance. Finite differences directly evaluate
each modified function along d and two seeded coordinates at .05/.025. Maximum
saved projected-check error is .0004353. CPU Float64 head reconstruction uses
the actual finite head inputs and the declared .005 score bound.

**Seven of 128 narrow 1e-4 finite-coordinate diagnostics fail**, maximum
.00013733. The separately declared .002 finite-coordinate replay and .001 score
bounds pass. Those seven stay failed. 287's original failed full-trace gate and
all 160 original raw failures also remain failed; this experiment does not
recertify their broader numerical scopes.

The final tiny CPU qualification passes 548 required records, checks all 512
source coordinates across eight arms/four unequal-length examples by finite
differences, and rejects three coherent tamper fixtures. An earlier **unfrozen**
extra check incorrectly assumed double model weights made all primitives
Float64; Qwen's internal Float32 RMSNorm/softmax casts invalidate that assumption.
Its failed attempt is retained. No frozen checkpoint tolerance was relaxed.

Checkpoint producer: **453.71 seconds**. Saved-array audit: **6.17 seconds**.
The 32 case archives total **107,164,829 bytes**, below the 128MiB cap. All
640MiB initial and 384MiB per-case reserve guards pass. Work includes 32 full
False observations, 32 first blocks, 64 anchor suffixes, 64 native calibration
suffixes, 3,584 modified forwards (145,152 cached block calls), 256 tape reverse
passes, 10,368 triangular block recomputations/VJPs and 256 triangular head
VJPs. Full-vocabulary head forwards total 4,000, including ordinary False,
anchor, calibration, modified and triangular-control calls. There is no speedup.

All 78 scientific sources were committed at
`153c3e1b95b9540197a46d704c4ca5a2e5acc764` before checkpoint execution and remain
unchanged. CI verifies source seals, published arithmetic, counts, failures and
figure provenance. Weights and activation arrays are omitted; their inventories
and regeneration commands are public. CI does not independently replicate the
checkpoint or validate the omitted gradient vectors.
