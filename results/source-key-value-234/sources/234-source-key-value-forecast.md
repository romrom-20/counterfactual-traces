# 234: can source key/value mixing forecast the missing local transport?

Frozen 19 October 2026 before these forecasts and native targets. Hypothesis
selected from the explicitly retrospective 233 factorial on already-observed
232 contexts. This fresh test validates that selected local computation; it
does not change the earlier registered results or establish novelty priority.

## Mechanistic hypothesis and exact rule

Keep the old False-context finite RMSNorm/probability–value-product rule FNV.
Correct its receiving tangent with **changed source keys and values**, while
retaining the **False softmax Jacobian and False receiver-value weights**.

Let p,V,K be False attention operands; qdot,kdot,vdot are derivatives in the
known receiving direction d. Source-only edits dK,dV are nonzero at the Boolean
row only. H(p)u=p*(u−sum(p*u)); output projection O has no bias on increments.

```
l = scale * (qdot K^T + q kdot^T)
b = scale * qdot dK^T
key   = O[H(p)b V]
value = O[H(p)l dV]
mixed = O[H(p)b dV]
C = FNV + key + value + mixed
A = FNV + key + value                  # direct mixed-product removal
K = FNV + key
V = FNV + value
R0/R1/R2 = FNV + row_norm_matched_random(key+value+mixed)
```

The source normalized K/V edits are computed from the known edited first-block
source row. The candidate never uses the source-context softmax Jacobian or its
probability weights for transporting receiver-value changes. The finite public
source-only primal remains in FNV, so **this does not eliminate the finite
source-only attention call**, all source-probability computation, or exact
edited block 1. Qualification includes extra graphs and counts them.

The mixed term is a direct probability-tangent/value product. Its removal is
an intervention on the specified local forecast computation, not a native
learned-circuit ablation. Removing it need not make every case inaccurate.

## Inputs, controls and frozen decisions

32 new pairs in two families: jade pavilion/copper rotunda Mon/Thu and beige
observatory/ginger foundry Tue/Fri, crossed schema, order, assignment and target.
64 full token hashes absent the bounded catalog through 232; still dependent
templates in one checkpoint. Zero new native True measurements before all 256
forecasts are qualified, globally sealed, pushed and remotely verified.

Primary target: native residual displacement **after attention 2 across all
receiving rows**. Native norm>=.001, C relative error<=.025, cosine>=.995, and
strictly lower relative error than FNV, A, K, V and all three random controls.
Require >=24/32 and >=6/8 in each schema/order stratum. Report raw accuracy and
every comparison for all arms. The primary could fail even if C is accurate.

Secondary: C preserves verdict-position accuracy under the same norm, relative
and cosine bounds, >=24/32 and >=6/8 per stratum. It need not improve every
control there and cannot rescue a failed primary. Publish the signed mixed
vector and whether its removal improves/worsens errors; do not infer learned
features from signed regimes. No final residual or verdict-margin forecast.

## Qualification, information budget and preservation

Per pair: one ordinary False, one exact known-edit block 1, one finite public
source-only attention-2 call, one False smooth anchor, two finite/affine graphs
0000/1001. Derive source K/V from one normalized source row. No finite native
both-input attention-2 target or ordinary native True before global seal.
Native True observation subsequently checks the first block exactly.

The used False receiving JVP is checked by AD/FD (h=1e-5, bound1e-4). Each
distinct affine-node derivative is independently qualified as in 230, analytic/
AD1e-9 and FD1e-4. Separate NumPy reconstruction checks the FNV rule, source
K/V increments, all three signed terms, all eight predictions and row controls.
Independent Torch hybrid functions from233 qualify 0000/1000/0010/1010 by
AD/FD before sealing; both their anchors and the finite full-F derivative
bridge are retained. These are CPU64 comparisons, not recertification of old
Float32 block derivatives. Public/smooth primals retain1e-4; analytic equations
retain1e-9; known prefix and native first-block replay require exact agreement.

Expected forecast work: 32 ordinary False, 32 edited blocks1, 32 public source
attention calls, 32 smooth False anchors, 64 intervention graphs, 32 receiving
JVP/FD comparisons (64 endpoints), and 32 source-row K/V projections. The
separate Torch auditor adds64 anchor reconstructions,128 hybrid AD/FD and32
full-F AD/FD comparisons (320 endpoints); native functions require96 further
finite attention graph evaluations. Distinct affine-node qualification and FD
endpoints are counted separately (expected192 comparisons/384 endpoints).
After sealing,32 ordinary True targets. No shortcut or compactness claim.

Preserve every case, forecast, raw-array hash, control, failed check and phase
log. Stop on failure rather than changing tolerances, predictions or the rule.
End-to-end Question B, source-generation compression, checkpoints/templates,
learned circuits and final behavioral consequences remain open even on a pass.
