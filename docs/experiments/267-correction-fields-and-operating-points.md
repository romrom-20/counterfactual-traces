# 267 — Separate correction generation from the MLP operating point

Freeze this protocol and its producer/helper/auditor before any new study calculation.
This is a retrospective causal diagnostic on the **same four old first-per-schema/order
263 contexts**, all indigo. They are dependent synthetic templates, not fresh confirmation.
Question B remains open. No browser, literature lookup or disputed whole-block Jacobian.

## Why this test

265's pointwise mixed-product removal did not remove the global policy interaction.
266 showed that an adaptive sparse SiLU mask can badly misrepresent local mixed writes
despite accurate final forecasts. Here **all neurons use finite SiLU**. We test changing
correction vectors separately from changing local MLP operating points, while retaining
the nonlinear adaptive backbone and attention. This is not another accuracy-only sweep.

Let `M(z)` be the actual Float32 full-shape gate/up, finite SiLU, product and down
projection. At a receiving MLP input x, let z(x) be its False-anchored RMS tangent,
including the observed normalized False anchor. Define p(x), d(x) by frozen263's `plans`:

```
s² = mean(xF²) + eps; h = x-xF
a = mean(xF*h)/s²; b = mean(h²)/s²; rho = (1+2a+b)^(-1/2)
z = zF + gamma*(h/s - xF*mean(xF*h)/s³)
p = (rho-1+a)*gamma*xF/s
d = (rho-1)*gamma*h/s
```

First run the known-edit L forecast using M(z(x)). At every layer cache its z*, p*, d*
and finite writes. These are **L forecast states**, never native True inputs.
The extra L rollout and multiple primitive evaluations are part of this rich budget.
This is not a cheap predictor or a learned-circuit claim.

Cross generator adaptation (first digit) and local MLP operating-point adaptation
(second digit) for each correction, with weights fixed throughout:

```
c11_p(x) = M(z(x)+p(x)) - M(z(x))
c10_p(x) = M(z*  +p(x)) - M(z*)
c01_p(x) = M(z(x)+p*)   - M(z(x))
c00_p(x) = M(z*  +p*)   - M(z*)
```

Likewise d. The common M(z(x)) backbone stays adaptive in every arm. A frozen
"operating point" is not a clamp of attention queries or of learned weights.
For every family s run P_s=M(z)+c_s_p, D_s=M(z)+c_s_d and
A_s=M(z)+c_s_p+c_s_d. Combine actual Float32 write operands in Float64 and round
the installed write once to Float32. These are **write-additive** interventions,
not identical to265's addition before down projection. A_s has no pointwise mixed
p/d correction, but preserves changing states and common nonlinear computation.

Use matched full-sequence primitive shapes; never apply sparse ranking. Receiver
queries use the previously qualified False-anchored cubic rule. Source/prefix, K/V,
softmax, projections, products, head and all nonreceiving computation remain finite.
H uses actual finite MLP RMSNorm/SiLU with cubic queries: it must replay258 C exactly.
F and T must replay258 F/T exactly. No frozen True trajectory supplies a forecast.

## Arms, controls and guards

Order per case: F,Z,L,H,P11,D11,A11,P10,D10,A10,P01,D01,A01,P00,D00,A00,Pr,Dr,Ar,T:
**80 complete forwards planned**, including repeated references. Z is A11 with no edit,
and must equal F exactly; its zero fields do not use the edited L cache.
For r, freeze seeded random WRITE vectors per case/layer/row/tag, each matched to
the norm of c00_p/c00_d (not neuron-product norm). Run all Pr/Dr/Ar controls.

Abort reference failures. Record intervention nonfinite or >100×False tail-norm
guards as failures, keep partial output and retain denominator4. No outcome-based
case exclusion or second primary. Require all random-family trajectories complete
before interpreting a failure of random preservation as directional specificity.

## Frozen primary and interpretation

At the final receiving state define I_s=rA_s-rP_s-rD_s+rL (signed whole vector).
I11 is the target. Test every predeclared family10,01,00,r with the same criterion:
whole and verdict norms of target >=.001, relative error <=25%, cosine >=.95,
and the finite common-anchor margin bound below. Preservation requires >=3/4 cases.
Report all four cases and all contrasts; no selected winner or causal percentages.

Use anchor rA11 and endpoints Float32(rA11-I_s). Read every endpoint with the
finite public RMSNorm and **full-vocabulary** head. Compare its margin to
the endpoint for I11; require error <=.005+.25*abs(m(rA11)-m(rA11-I11)). This avoids
adding scalar margins or comparing effects at different readout anchors.
Report native5%/.99/.10nat whole+verdict criteria for every complete arm separately.

Calibration: H and A11 must each meet the native joint criterion in all4 cases
before relating the factorial to the native-response explanation. If they fail,
retain the factorial as a property of the defined policies and explicitly limit
the mechanistic connection; do not adjust calibration post hoc.

If00 preserves I11, the common nonlinear backbone can retain it under constant
fields. If10 alone preserves it, adapting generator vectors suffices under this
clamp; if01 alone preserves it, adapting local response suffices. If both preserve
and00 fails, there are alternative sufficient accounts. If neither preserves and
11 is qualified, joint adaptation matters under these clamps. Random preservation
prevents trained-direction specificity. These are intervention-specific statements,
not an additive causal decomposition, learned features or a unique learned circuit.
Secondary: report I11-I01, I11-I10 and I11-I10-I01+I00 as signed vectors, and their
norms/projections, without renaming them primary successes.

## Numerical qualification and preservation

Pinned258 checkpoint/runtime, MPSFloat32/eager, CPUFloat64 coefficients. Bind all
source/dependency/protocol hashes, external Qwen implementation, input seals and
checkpoint. Qualify smooth RMS derivatives separately with CPU64 AD1e-9,
central differences1e-3/1e-4 at1e-3/1e-5 and independent cubic jets1e-10, as261b.
Do not recertify the unresolved whole-block Jacobian checks.

Replay public RMSNorm and finite gate/up/SiLU/product/down at original full shapes;
verify node digests, installed writes and receiving block residual additions exactly.
P11/D11 must equal their direct finite corner writes exactly. Frozen field hashes
must regenerate from L; random per-row norm relative error <=1e-5. Prefix is exactF,
edited source exactT; F/Z and258 F/C/T trajectories/scores exact. Public finite head
must exactly replay every actual arm's saved logits before common-anchor decisions.
Any required numerical failure stops attribution and remains visible.

Record separate CPU64 raw gate/up/down checks at the **original1e-4** bound;
failures remain failures, not rescued by matched-public arithmetic. Record CPU64
finite-head comparisons at.005 separately. Neither is a whole-model certificate.

Save raw x/qx for each arm/layer, False normalized anchors, final full receiving
tail and scores. All other primitive/trajectory data have exact digests and can be
regenerated from these arrays/weights. Save signed final contrasts. Input arrays,
protocol and scientific sources are immutable. New output directory required;
failures preserved; cap267 raw output260MiB and keep512MiB free. Never delete old
evidence to fit a run. Publication scripts/figures are explicitly post hoc.
