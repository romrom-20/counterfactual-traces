# A scalar fit does not identify the changed receiving computation

**None of six partial receiving-MLP coefficient accounts meets the frozen
population criterion.** Five pass 0/32 cases; the two-up-term account passes
1/32, in a case below the finite-effect materiality guard. Every partial account
fails the separate receiving-vector test in all 32 cases. Three shuffled
coefficient controls also fail throughout. Full restoration is calibration,
with 32/32 expected passes.

One partial account nevertheless passes **32/32 native scalar calibrations**,
flags all six old large margin errors with zero false positives, and has
0.00638-nat median scalar error. Its gradient contrast has 65.4% median relative
error; its receiving-displacement contrast has 69.3% median error. A scalar
prediction can pass while the tested internal response remains wrong.

This rejects the proposed simpler coefficient accounts. It does not identify a
learned circuit or establish why the edited receiving state was generated.
General Question B remains open.

[Frozen protocol](../experiments/291-receiving-coefficients.md) ·
[All cases, controls and checks](../../results/receiving-coefficients-291/README.md) ·
[Reproduce](../../research/README.md#receiving-coefficients-291)

![Every coefficient condition, gradient comparison, receiving vector and scalar calibration](../figures/receiving-coefficients-291/receiving-coefficients.png)

## The contrast is smaller than the native Boolean response

290 found that restoring receiving gated-MLP response maps, with edited source,
attention and readout supplied, accounted for the remaining gradient/finite-effect
gap within its conditional bounds. Normalizer response alone did not.

291 separates that MLP response into three coefficients. F denotes ordinary
False; E is the **source-only MF forecast**, not native True or full MM. The
boundary is after block 1. E installs the predicted source row and keeps False
incoming receiving rows, followed by finite adapted computation. The small
source correction d=C_source-M_source includes original Float32 residual
rounding. It is not the large Boolean edit.

All 32 contexts were previously observed in 285: four layouts with eight cases
each and 16 dependent schema pairs. There are no new ordinary True forwards or
True gradients. All comparisons are retrospective.

Source recurrence, attention response maps and final readout remain E.
Receiving input/post-attention norm response maps remain F. Every condition has
the same zero-offset E trajectory. The modified functions are finite; their
derivatives are checked against actual finite changes, without backward
replacement.

291's reference is **the MLP-only response change**, full corner 111 minus 000.
Those endpoints reproduce 290's 01 and 00. It is not 290's entire all-E-minus-00
target. Calibration against both saved endpoints passes the declared bounds.

## Separate three state-dependent multipliers

For normalized input z, gate g=Wg z, up value u=Wu z and phi=SiLU, the smooth
receiving MLP derivative is

    dMLP = Wdown [u*phi'(g)*(Wg dz) + phi(g)*(Wu dz)].

Three bits independently restore:

- **G**, gate response: the operating point for SiLU's slope and finite gate increment.
- **U**, up multiplier: the up value multiplying that gate response.
- **P**, up response coefficient: the gate activation multiplying the up increment.

The local derivative in condition abc is
<code>Wdown[diag(u_b*phi'(g_a))*Wg + diag(phi_c)*Wu]</code>.
The finite product retains its gate-up interaction. Direct coefficient
interventions allow these parents to be crossed; mixed conditions are not
claimed to be native learned states. Exact functions are in the protocol.

000 is baseline and 111 full calibration. Singles 100/010/001 and pairs
110/101/011 are the six preregistered sufficient-account hypotheses. R0–R2
shuffle the two compound coefficient changes among neuron coordinates while
preserving their row norms and joint coordinate pairing. These controls test
alignment with projection columns, not on-manifoldness or every higher-order
finite effect.

All receiving MLPs are intervened on together. These factors do not isolate a
particular learned neuron, layer or cross-layer path.

## What passed, and which tempting account failed

The frozen joint test requires gradient-target norm >=.001, relative error <=.25,
cosine >=.95 and signed finite-effect error <=.005+.25*abs(target). Population
success requires >=24/32 and >=6/8 in each layout. All cases remain in every
denominator. Only 16 remaining finite effects are material, at abs >=.02 nat.

- G, U and P individually: **0/32** joint passes.
- G+U and G+P: **0/32**.
- U+P: **1/32**, XML only; **0 material successes**.
- Each shuffled coefficient control: **0/32**.
- Full 111 calibration: **32/32**, including all 16 material cases.

The conditional receiving-vector criterion is also .25 relative error/.95
cosine with the .001 guard. Every partial restoration and shuffled control
passes **0/32**. Full 111 is its exact reference. These are correction-vector
contrasts, not the old native 5%/.99 displacement criterion.

P alone (001) is especially instructive. Its native scalar prediction uses
the **original M margin baseline** and target mT-mM. All 32 errors are within
.1 nat; all six targets above .1 nat are flagged, with zero false positives.
Median error is .006382 nat, maximum .06101. Yet its gradient contrast has
39.6%–87.9% relative error, and every receiving vector fails. The scalar
comparison constrains one projection of the response; it does not determine
the response vector.

This does not erase 290's successful composed family result. It rejects the
attempt to compress that result into one changed coefficient family or pair.
Restoration/removal comparisons are conditional on the supplied computation;
failure of the tested sufficient accounts is not a universal necessity proof.

## Posthoc accounting explains why adding isolated effects is inadequate

A separately labeled, zero-model analysis reconstructs all seven Boolean
factorial gradient terms from the saved corners. The sum of the three isolated
effects has **167.1% median relative error** against the full contrast. Retaining
all pair interactions still leaves **76.8%** median error: the three-way term
is appreciable.

Per-case signed terms close algebraically. They can reinforce or cancel;
across-case medians do not sum. The gate/up mixed term has median signed target
projection -.535 and median norm 1.58 times the target norm. This is
conditional factorial accounting across all intervened layers. It cannot
distinguish one local gate-up product from subsequent cross-layer composition,
and it is not evidence for two opposing learned mechanisms.

Published 8-by-8 gradient Gram matrices allow these signed calculations and
additive errors to be reconstructed without publishing gradient vectors.
The original decision rules remain unchanged.

The frozen receiving-vector endpoint subtractions and factorial accumulation
use Float32 arrays. A separate promoted-Float64 check changes **0/352**
receiving-vector classifications. Maximum relative-statistic change is
3.33e-11; maximum factorial projection change is 8.53e-8. This bounds that
accounting contrast, not every derivative or numerical error.

## Numerical scope and cost

The original audit passes **8,544 required records**: 2,112 replay/general,
3,168 source projected FD/halving and 3,264 primitive formula/local FD records.
All 9,504 intermediate reverse-trace diagnostics pass.

**Two of 96 original narrow 1e-4 finite-coordinate diagnostics fail**:
both baseline endpoint replays, maximum 0.000106812. The separately declared
.002 coordinate/.001 score bounds pass. These two failures and all historical
287/288/290 failures remain failed.

Primitive probes use one predetermined layer per case, actual cached operand
shapes, one verdict-row cotangent and all intermediate gate/up coordinates.
CPU64 reconstructs SiLU primals, down cotangents and VJPs. Other rows have exact
zero adjoints. Gate/up coordinate finite differences use .05/.025 and halving.
Maximum primitive error is 2.98e-5; whole-function projected maximum is .000610352.
Full-tape and separately composed blockwise reverse gradients also pass.
This is scoped qualification using the same library primitives, not every
layer/cotangent, an independent full Jacobian or rounded-program derivatives.

Before checkpoint execution, four unequal-length tiny CPU cases passed 1,068
required checks, 704 all-source-coordinate FD checks and coherent bound/FD/
primitive tamper rejection. All 94 scientific sources were frozen at
<code>293335e</code> before checkpoint execution. No scientific source changed afterward.

The study took 964.35 s for production and 6.62 s for the original audit.
32 archives total 69,388,833 bytes. The 512/384 MiB headroom guards and 192 MiB
case-archive cap pass; no prior evidence was deleted or recompressed.

The complete budget includes 32 False observations, 32 first-block calls,
64 anchor suffixes, 32 finite calibration suffixes, 32 all-E reference reverse
passes, 4,928 modified forwards (266,112 cached block calls), 352 full-tape
reverse passes, and 9,504 blockwise VJPs (19,008 recomputed block calls).
Additional primitive/projection work is recorded in status.json. No speedup
or compact prediction claim follows.

## What this leaves for Question B

The combined receiving response remains a valid conditional account within
290's scope. 291 shows that matching a native scalar error is not enough to
explain it and that isolated coefficient contributions do not add correctly.
More coefficient partitions alone will not explain the origin of the edited
state.

A useful next hypothesis must predict the **two compound response coefficients
together**, from False information and a declared forecast state update, and
then face fresh native residual and margin criteria. Supplying E gate/up
parents, or reproducing a derivative after supplying its coefficients, cannot
by itself meet that test. No such new hypothesis or fresh result is claimed here.
