# The correction changes more than its local MLP operating point needs to

**A controlled finite test distinguishes two kinds of state dependence.** On
four old contexts, allowing the normalization correction vectors to change
preserves the signed final interaction even when their local MLP operating
point is frozen. Freezing those vectors while allowing the operating point to
change fails. This is a useful constraint on the explanation of the defined
normalization-policy interaction. It is not a learned circuit or a complete
answer to Question B.

![Every case, including frozen and random controls](../figures/fields/field-feedback.png)

[Frozen protocol](../experiments/267-correction-fields-and-operating-points.md) ·
[All records and failures](../../results/correction-field-feedback-267/README.md) ·
[Reproduce](../../research/README.md#finite-correction-field-factorial-267)

## The ambiguity this resolves

265 found that removing the immediate mixed MLP product effect did not remove
the final interaction between two normalization corrections. The contrast
survived their passage through changing computational states. 266 also found
that a sparse adaptive SiLU mask could misrepresent the local contrast, despite
accurate whole forecasts. We therefore use **finite SiLU at every neuron** here.

There were at least two explanations left. A changed incoming state could
generate a different correction vector. Or the same correction could encounter
a different MLP activation baseline and produce a different write. A global
contrast alone cannot distinguish them. This experiment crosses those two
dependencies during complete continuation, keeping the common backbone and
attention adaptive in every condition.

The target is explicitly **a contrast between defined policies**, not the
native True-minus-False response itself. We test native-response accuracy
separately to check that the central policies remain behaviorally relevant.

## What is held fixed

At each receiving MLP input x, split the False-anchored RMSNorm update into its
tangent z(x), a correction p(x) along the normalized False pattern, and a
correction d(x) along the weighted displacement. The frozen 263 formula computes
these from the current input, False input/normalized anchor, norm weights and
epsilon. It uses no native True receiving state.

Let M(z) be the actual finite full-shape MLP: gate and up projections, SiLU,
gate–up multiplication and down projection. Run a tangent-only L forecast
first, and cache its layerwise z*, p*, d*. This extra forecast is part of the
budget; the cache is an internally produced approximate trajectory.

The four p correction fields are:

```
11: M(z(x)+p(x)) - M(z(x))     both arguments adapt
10: M(z*  +p(x)) - M(z*)      correction generation adapts
01: M(z(x)+p*)   - M(z(x))    local operating point adapts
00: M(z*  +p*)   - M(z*)      both arguments frozen
```

Apply the same crossing to d. MLP weights never change. A frozen local operating
point does **not** freeze attention queries, the common M(z(x)) backbone, or
the finite nonlinear response to the correction itself.

For each family, run P with its p correction, D with its d correction, and A
with their write-additive sum. Add actual Float32 write operands in Float64,
then round the installed write once to Float32. This intervention differs
explicitly from265's sum before down projection. All families omit the
pointwise mixed p/d correction but retain evolving inputs and attention.

The signed final interaction is `I_s = rA_s - rP_s - rD_s + rL`. Compare each
crossed family to I11. Require at most 25% vector error and cosine at least .95
for both all receiving rows and the verdict row, plus a finite margin check.
The margin is read at a common anchor rA11 using endpoints rA11-I_s, rounded
once and passed through the public finite final norm/full-vocabulary head.
It is not a sum of scalar margins. The study criterion requires at least 3/4
joint successes. Every family and case was declared before execution.

## Results

- **Generator adapts, operating point frozen (10): 4/4 joint passes.** Median
  whole-interaction error 10.61%, cosine .99491; verdict error 7.93%, cosine .99737.
  Median common-anchor margin discrepancy .004965 nats. Per-case whole errors
  are 14.10%,13.92%,7.30%,7.04%, in generic0/generic1/descriptive0/descriptive1 order.
- **Generator frozen, operating point adapts (01): 0/4.** Median whole error
  93.34%, verdict error 93.36%, margin discrepancy .11250 nats. Whole errors are
  99.25%,100.06%,87.44%,87.02%. Keeping the local response adaptive is insufficient.
- **Both frozen (00): 0/4.** Median whole error 97.16%, margin discrepancy .08612 nats.
- **Matched random frozen writes: 0/4.** Median whole error 100.24%, margin
  discrepancy .10658 nats. Every random trajectory completed; each p/d write was
  separately norm-matched by layer and row. These are write controls, not
  neuron-product controls. Random-versus-original frozen fields does not isolate
  a unique trained direction, since neither preserves the target interaction.

The target whole-interaction norms are 5.31,5.54,9.93,10.36; verdict norms are
3.82,4.17,9.12,9.57. All exceed the frozen .001 guard. There are no instability
guards, omitted cases or outcome-dependent exclusions. The11 self-comparison
passes 4/4 by construction and is not an additional discovery.

Native calibration passes as preregistered: the actual finite-MLP/cubic-query
reference H and adaptive write-additive A11 each pass 4/4 under the original
5% whole/verdict error, .99 cosine and.10-nat margin criterion. Their median
whole errors are .983% and 1.094%. H exactly replays258's finite-SiLU C reference.
The frozen-operating-point A10 also passes 4/4 native cases, at 3.200% median
whole error and .05545-nat margin error. A01 passes 2/4; A00 and random Ar pass 0/4.
The L baseline passes0/4, with 31.72% median whole error. All 17 evaluated
policy outcomes are published per case; accuracy does not replace the
interaction test.

## What changed in the explanation

For these interventions, changing normalization **correction generation** is
sufficient to preserve the signed global interaction within the declared
accuracy, even when the correction's MLP operating point stays at the earlier
L forecast. Changing that operating point with fixed correction vectors is
not sufficient. Thus we can remove one proposed state dependency without
destroying the measured effect. The large global contrast need not be attributed
to an immediate product interaction or to a continually changing local MLP
baseline for reading these corrections.

This is conditional sufficiency, not a percentage allocation of the native
mechanism. The common backbone, query feedback and subsequent layers still
adapt. A10 retains finite learned MLP evaluations at z* plus the current
correction; it is not a linear fixed readout. The frozen L cache itself comes
from a substantial forecast. The interventions do not show that changing
correction generation alone explains native displacement without those
remaining operations, or that this is the only sufficient account.

The next unresolved distinction is inside that generator. Its p coefficient
depends on the False/current radial overlap and displacement energy, while d
also follows the current displacement direction. Which of those state changes
must adapt? Why does the learned upstream computation produce those particular
values and directions? A frozen test of those dependencies, followed by new
contexts, would be stronger than another output-accuracy sweep. This study
does not yet answer them or establish a smaller prediction budget.

## Numerical scope and the failed initial attempt

Scientific source/protocol freeze:`d2e3e408d1ebcd682e18289c27b8f5022813d5ff`,
before any267 study calculations. Four old first-per-schema/order 263 contexts,
all indigo; not four independent tasks or fresh confirmation. Pinned
Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306,
Torch 2.14.0 / Transformers 4.57.6 / NumPy 2.5.3, MPS Float32/eager, CPU64 coefficients.

The retained execution has 80 full forwards and 4,816 additional public MLP
evaluations, with 112 L caches and 2,240 policy points. Its producer wall time is
71.19 seconds in the recorded local environment, including loading and saving;
this is not a speedup comparison. The separate audit adds primitive work, not
another model rollout. There are 57,552 passing declared checks, maximum
absolute error 1.6055e-7. It checks 4,480 smooth RMS derivatives against CPU64 AD,
8,960 finite differences, independent cubic jets, exact full-shape public
normalization/MLP node seals, write installations, receiving residual additions,
prefix/source isolation and all 80 actual finite-head scores. F/Z and original
258 F/C/T controls replay exactly.

**371/13,440 raw CPU64 gate/up/down comparisons still fail their original 1e-4
bound.** All remain failures. The separate CPU64 finite-head bridge passes 80/80
at .005; the public matched-shape head replays exactly. This is a qualified
public Float32 component scope plus independently checked smooth RMS formulas,
not an independent whole-model real-arithmetic or Jacobian certificate. All
older failed derivative checks remain unresolved.

The first attempt stopped at the 512 MiB free-space guard after 20 forwards, before
persisting its first case arrays. No numerical result from it is claimed. Its
design/runtime/source seals and failure are retained. Identical tracked checkout
files were then converted to APFS copy-on-write clones with byte/hash checks;
no experiment arrays were deleted or altered. The unchanged frozen experiment
ran into a new directory. Actual work includes those 20 unretained forwards in
addition to the 80 retained forwards. Raw new case arrays total 122,608,513 bytes;
all case and signed-contrast arrays are inventoried and remain local.

The result is one useful mechanistic constraint toward Question B. General
prediction, learned upstream displacement generation, cross-template/checkpoint
scope, circuit identification and compute savings remain open. CI checks the
record and source integrity; it is not independent scientific replication.
