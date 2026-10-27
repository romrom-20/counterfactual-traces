# Which initial states generate the changed source sensitivity?

**Qualified retrospective result:** changing the source row alone after block 1
recreates the measured sensitivity shift in 32/32 old contexts; changing only the
other incoming rows passes 0/32. Every later receiving computation remains finite
and adaptive. This locates a sufficient INITIAL change for this particular
gradient/finite-effect contrast. It does not make receiver adaptation dispensable.

**Numerical boundary:** the original 287 audit FAILED 1/3,072 required records.
A separately frozen, posthoc 287b qualification passes 128 boundary-adjoint and
1,152 projected-derivative checks at the original bounds. It supports the boundary
gradients actually used here; it does NOT pass the original full-trace gate.
All 160 original cached-coordinate diagnostics also remain failed. Interpret the
following under that narrower, explicitly changed numerical scope.

![All 32 cases, sensitivity and full native fidelity](../figures/source-receiver-readout-287/source-receiver-readout.png)

## The question and information budget

[286](286-state-dependent-readout.md) found that the ordinary False-state margin
gradient misses all six large donor margin errors, while the edited forecast's
gradient predicts them. But both its source and other incoming suffix states
changed. That comparison did not identify which initial change generated the
sensitivity difference.

287 crosses those states at the residual AFTER block 1. F denotes the ordinary
False state, M the False-derived, magnitude-matched donor forecast from 285.
The edited Boolean's row is the source; rows STRICTLY AFTER it are receivers.
The source is excluded from the receiver factor. The four corners name source
first: FF, MF, FM, MM. All share the unchanged False causal prefix. In MF only
the source row changes; in FM only the other rows change. M is regenerated using
the original known edit/forecast write, with a verified False-gate substitution
BEFORE SiLU. No native True gradient is used.

All 27 subsequent source and receiving blocks, attention/query adaptation and the
full-vocabulary final head execute finitely. The "readout gradient" differentiates
that entire continuation. It is not just the final unembedding. One ordinary False
observation supplies both prefix KV and full prefix inputs for numerical checking.

The cohort is all 32 already observed 285/286 contexts, four layouts of 8 and 16
dependent schema pairs. This is retrospective, not a new fresh prediction. Sources
and scientific thresholds were frozen before these new checkpoint calls.

## The two planned explanations

Let gAB be the source-row gradient of the final candidate margin at corner AB.
The target is V=gMM−gFF. The coequal hypotheses are:

- Source initial-state account S=gMF−gFF.
- Receiving initial-state account R=gFM−gFF.

The small source correction d is the original Float64(C_source)−Float64(M_source),
including Float32 residual-addition rounding. At each corner install that SAME d
and run the finite continuation. Define EAB=m(AB+d)−m(AB). The signed target
effect shift is D=EMM−EFF. S predicts EMF−EFF; R predicts EFM−EFF.

A case passes jointly when norm(V)≥.001, gradient-vector error≤25%, cosine≥.95
and signed finite-shift error≤.005+.25abs(D). A population claim requires 24/32
and 6/8 in EVERY layout. These frozen scientific thresholds are unchanged in 287b.
The separate .02-nat materiality guard applies to D, not to every native error or
the earlier 286 projection contrast.

## What the factorial says

The source account passes 32/32, 8/8 in every layout. Median gradient-shift error
is 2.69%, maximum 7.15%; median cosine is .999676. The finite-effect-shift error is
.000803 nat at the median and .003902 nat at maximum.

The receiving account passes 0/32. None passes its vector criterion and one passes
the finite criterion. Median gradient-shift error is 100.17%; median finite error
is .02557 nat. This is a failure of the INITIAL-receiver-only account, under these
states and tolerance, rather than a claim that receivers do not matter later.

The secondary additive S+R passes 30/32. All 32 vector criteria pass; two dialogue
finite-effect comparisons fail. Its exact missing term is
gMM−gMF−gFM+gFF, with a corresponding finite four-corner interaction. The mixed
vector has 3.86% median norm relative to V; the finite mixed effect reaches .01020 nat.
These are defined factorial coordinates, not competing learned mechanisms or
fractions of causation. The secondary does not rescue or replace either primary.

Twenty-one D effects reach .02 nat. Source-only passes all 21 eligible cases;
the other 11 remain ineligible in all 32-denominator material counts. Receiving-only
passes 0 material cases; additive passes 21. The basic gradient criterion permits
small finite effects and must not be relabeled as 32 material behavioral effects.

Every case's vector, finite, joint and material classification is identical when
using the full-shape gradients instead of cached gradients. Both paths use the
same finite intervention scores. Their agreement is a sensitivity analysis,
not an independent research team's replication.

## Scalar importance and complete native prediction diverge

The secondary scalar comparison asks whether gAB·d predicts the OLD native
margin error mT−mM. FF and FM pass26/32 and miss all six large errors. MF and MM
pass32/32 and identify all six. MF's median scalar prediction error is.00429 nat,
maximum.01105 nat; MM's is.00358/.01408 nat. These are scalar estimates, not native
residual predictions.

Directly continuing each corner+d gives a stricter result. FF, MF and FM each
pass0/32 complete native-response criteria; MM passes 32/32. Median receiving
vector error is 8.80% for MF versus.259% for MM, against the frozen5% bound.
MF's median verdict-position error is 1.82%, but the complete receiving target
still matters. Do not discard it because one scalar or position is close.

Thus the source state can seed the changed downstream sensitivity without
reproducing the complete native trajectory from otherwise False boundary rows.
The test separates those two targets. It does not rescue 285's failed stronger
repair claim, remove the need to predict receiving updates, or provide a smaller
native generator. MM+d is the old C forecast up to its retained replay bounds.

## Numerical qualification, including the failed gate

The original independent saved-array audit verifies all 3,072 record. It fails
one: FM source adjoints in case b2c7c95919cf, before block 16, differ by
1.41113997e-4 against 1e-4. The relevant boundary in that case differs by
7.00355e-6. Original audit status stays FAILED and it writes no scientific tables.
Its 1,152 cached projected-derivative records pass. Raw cached/full final-coordinate
diagnostics fail 160/160 at1e-4, maximum .000831605; predeclared .002 coordinate,
1e-4 relative and.001 score replay bounds pass. No old failure is recertified.

The separately named287b protocol was written after this failure and before
reading scientific outcome tables. It narrows the claim to the post-block 1
source gradient actually used by the factorial, without changing the1e-4 bound.
All 128 cached/full boundary-coordinate comparisons pass, maximum 9.04873e-6.
All 1152 full-shape projected checks pass against original saved plus/minus
finite scores at steps .05/.025, using the original .001+.02abs(AD) and halving
.002+.03abs(AD) bounds. Every other original required check remains required.

The full-shape reverse calculation uses the same library primitives and is one
scalar adjoint, not a full Jacobian or independent primitive derivative. Autograd
treats Float32 rounding as identity. Full-shape projected checks use the saved
cached finite differences; the qualified cached/full finite replay comparison is
also part of the evidence. This is not a claim about the derivative of the
bitwise rounded program. The intermediate trace before block 16 remains outside
the fully passing numerical scope. No primitive-level learned attribution follows.

Source-trajectory hashes match EXACTLY when only later receiving rows change,
with and without d, as causal masking requires. False inputs, first-source gate,
activation, write and original forecasts replay; prefix hashes stay fixed. CPU
full-tape/cached/blockwise reverse tests cover all four corners and unequal-length
sentinels. Boundary, projected-derivative and other-required-check tampering is
rejected by the new array-only qualification. These qualify the defined paths;
they do not validate every older derivative implementation.

## Execution and reproducibility

The original 287 source freeze is ff01b72138b1eeee03a044f24abc18b1b6a38cb2,70 files.
The posthoc 287b freeze is 98f58d5,73 files including those 70 unchanged. Neither is
edited to accommodate results. The public manifest also binds posthoc plotting,
export and arithmetic-verification helpers separately.

The completed producer takes 292.65 s on this Mac: 32 ordinary False forwards,
32 regenerated first blocks,1,792 cached suffix forwards,128 cached backwards,
3,456 full-shape block forwards,3,456 reverse block recomputations/VJPs and 128
head VJPs. New case arrays occupy58,271,747 bytes, under the 64 MiB cap. The
original audit takes 6.84 s and fails as described;287b is saved-array arithmetic,
with no new model calls. All later finite/reverse work counts against any
efficiency interpretation. There is no efficiency claim.

An initial checkpoint attempt stops at the unchanged512 MiB disk guard after
model initialization and BEFORE any case or False forward. It is preserved, as
are CPU development failures. The completed retry uses a new output directory
with the same70 source hashes and all original guards. Storage was recovered by
APFS sharing of byte-identical, metadata-verified files in inactive checkouts;
scientific arrays, weights and private drafts were not deleted or recompressed.
Execution notes bind the retained failure/housekeeping records without publishing
private local paths or writing.

[Scalar evidence and omitted-array inventory](../../results/source-receiver-readout-287/README.md),
[original protocol](../experiments/287-source-receiver-readout.md),
[posthoc qualification](../experiments/287b-boundary-qualification.md) and
[reproduction commands](../../research/README.md#source-and-receiving-initial-states-287)
provide the full record. The dependency-free verifier checks source seals,
candidate-score arithmetic, vector sufficient statistics, every denominator,
failures and figure provenance. It does not verify omitted arrays or replicate
the checkpoint; the saved-array audit and producer are supplied for that work.

## What this changes for Question B

We can now locate a sufficient initial change for the previously observed
sensitivity shift. Initial source state is enough when later source and receiving
computation remain adaptive; changed incoming receiver rows alone are not. The
whole native residual still needs more. This is a quantitative causal statement
about the defined boundary interventions, not an identified learned circuit.

The next unresolved mechanism is how that source change alters the derivative
of the subsequent computation: changed source-local transmission, changed
receiver response to the resulting messages, or both. A useful follow-up would
hold the edited forward trajectory fixed while separately substituting the
source-local and receiving tangent maps, with finite-difference qualification of
the actual modified functions. Simply plotting large adjoint norms or running
another source correction sweep would not answer that question. No such new
experiment is reported here. General Question B remains open.
