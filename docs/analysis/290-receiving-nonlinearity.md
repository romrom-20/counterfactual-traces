# The changed receiving gated MLP response fills the remaining gap

**Receiving gated MLP response passes 32/32; receiving normalization alone
passes 0/32.** This is conditional on supplying the edited source recurrence,
attention interface and head. It explains the defined remaining sensitivity
discrepancy and its finite correction effect in these retained contexts.
General Question B remains open.

[Frozen protocol](../experiments/290-receiving-nonlinearity.md) ·
[All evidence](../../results/receiving-nonlinearity-290/README.md) ·
[Failed attention account](289-attention-interface.md) ·
[Reproduce](../../research/README.md#attention-and-receiving-response-289290)

![Normalization-only and gated-MLP-only responses, with all-E calibration separate](../figures/receiving-nonlinearity-290/receiving-nonlinearity.png)

## The experiment that selects the family

289 restored source K/V normalization, QK/softmax routing and probability–value
mixing while leaving other receiving responses False. Every variant failed.
The full attention interface still missed 69.9% of the remaining gradient
contrast at the median. Attention transmission alone was therefore insufficient
under that explicit intervention.

290 holds that edited attention interface fixed. It crosses two receiving
response families: N, input/post-attention RMSNorm; and M, the gated MLP after
normalization. The source's K/V input normalizer is already edited in every arm.
The M family includes learned gate/up projections, SiLU, gate–up multiplication
and the down projection. It does not isolate SiLU from multiplication.

All four arms preserve exactly the same E baseline trajectory. E is the
source-only MF forecast, not full MM or native True. Primitive responses are
actual finite functions recentered at False primals with False or E parents.
Their gradients are checked against their own finite differences. The same
small C-minus-M source correction is installed in each arm.

Arm 00 replays289's failed111. Arm 01 updates gated MLP response alone. Arm 10
updates normalizer response alone. Arm 11 restores all local E response maps:
it is a calibration by construction, not the scientific discovery. Targets
are the remaining gradient, finite margin and receiving-correction difference
between E and arm00. Both single-family accounts were frozen as primary tests.

## What the positive result covers

Gated-MLP-only01 passes 32/32 joint gradient/finite-effect cases, eight in each
layout. Median gradient relative error is 8.97%, maximum19.22%; median cosine
.9970. Median finite-effect error is.00212 nat, maximum.00800 nat. All 32 pass the
separate receiving-correction vector criterion, with14.7 %median relative error
and.9893 median cosine. The frozen conditional vector tolerance is25%, not the
stricter5% native-response criterion.

Normalizer-only10 passes 0/32, including 0/32vector cases. Its median gradient
relative error is 101.4%, finite-effect error.0248 nat. Receiving correction vectors
also fail0/32. The all-E11 calibration passes 32/32 with gradient relative error
about2.6e-6. That calibration verifies the intervention construction.

Only 16 of the32 remaining finite contrasts reach the frozen.02 nat materiality
guard. All16 are recovered by01; the other16 remain in the full denominator
and do not become material successes. All 32 cases are old, with sixteen
dependent schema pairs; this is not a fresh prediction test.

Secondary native donor scalar calibration passes 32/32 with01. It identifies all
six large old donor errors and has.00568 nat median error, maximum.01310 nat.
Arm 00 identifies none of those six. These predictions use the ORIGINAL M margin
baseline, not E's absolute margin. They do not establish complete native vectors.

## A concrete computation, and the unresolved primitive split

For a normalized receiving input z, write g=Wg z, u=Wu z and phi=SiLU(g).
The gated MLP is `Wd(phi * u)`. Its response to a small normalized input step is

```
dMLP = Wd [ u * SiLU'(g) * (Wg dz) + SiLU(g) * (Wu dz) ].
```

The learned linear maps are unchanged across anchors. The state-dependent
coefficients are `u*SiLU'(g)` and `SiLU(g)`. Supplying the edited gated response
changes these coefficients, thereby changing how a small source correction is
transformed through receiving states and toward the verdict. The finite
restorations show that this family accounts for the tested remaining response;
the derivative identity itself is ordinary gated-network algebra.

The test has not separated activation-slope curvature, changes in up values,
and their multiplicative contribution. It changes every later receiving MLP
together and does not locate individual layers or neurons. A next primitive
test should split those coefficient changes, measure signed receiving-vector
effects and validate them by finite removal/installation controls before
proposing a smaller forecast rule.

**This does not make normalization unimportant.** Every arm already has the
edited baseline, and the E MLP parents were formed through normalization in that
forecast. N0 retains False normalizer response derivatives; it does not erase
normalization from how E's gate/up values were generated. The result is about
conditional perturbation response, not the complete origin of the native edit.

## Numerical qualification and limits

All **9,280 required records pass**:1,216 source/primal/replay/head checks,
1,152 projected FD/halving records and 6,912 receiving RMSNorm formula checks.
Full-tape and blockwise boundary gradients agree at 1e-4; all3,456 intermediate
diagnostics pass. Receiving input/post RMSNorm VJPs at both anchors are checked
across all coordinates for one verdict-row cotangent against explicit Float64
formulas. Maximum norm-record error is5.30e-6. The gated MLP's composed
derivatives are qualified by the actual modified-function FDs and two reverse
compositions, not independent AD of every SiLU/product coordinate.

Eight of 96 narrow1e-4 finite-coordinate diagnostics remain FAILED, maximum
.00012207. The separately declared.002 cached finite-coordinate and.001 score
bounds pass. Do not erase these failures or reinterpret the old287/288 failures
as passed. The final tiny CPU qualification passes360 required records,
all 256 source-coordinate FDs, four unequal lengths/four arms and three tamper
rejections. The original unfrozen stale-field analysis failure is retained.

All 89 scientific sources were committed at
`462b0ba36266b7493bfcde01f025ee83af423b1b` before checkpoint execution. Producer
331.82s; saved-array audit6.37s.32 case archives total81,154,957 bytes. All
384/256 MiB reserve guards and96 MiB cap pass. Work includes32 fullF captures,
32 first blocks,64 anchor suffixes,32 finiteE calibrations,32 E reference
forward/backwards,1,792 interface forwards with96,768 cached block calls,
128 tape backwards,3,456 blockwise VJPs,51,840 encoding norms,51,840 receiving
norm re-evaluations,25,920 MLP reanchor calls, and3,456 norm-probe VJPs. All
heads remain full-vocabulary. This is substantial diagnostic computation.

Source recurrence, attention and head E anchors are supplied by finite
False-derived forecast replay. This is a conditional family-level account of
small-correction sensitivity, not a compact False-only native predictor or an
identified learned circuit. Historical grounding failures, the single small
checkpoint, dependent synthetic templates and lack of a fresh independent
replication still constrain the claim. CI verifies records/arithmetic/seals,
not omitted vectors or independent checkpoint execution. No priority claim.
