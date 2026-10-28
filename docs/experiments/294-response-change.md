# 294: response change versus endpoint sensitivity at the same inputs

Freeze code, protocol and criteria before new checkpoint-weight calculations.
This is a conditional, retrospective diagnostic on all eight qualified293c
contexts, using only P probes at blocks 1/15/28, source and verdict rows. It has
48 points, not 48 independent tasks. No new model forward or native True call.
General Question B remains open. The private article stays local.

## Question

293's False and adaptive 32-direction response ranges poorly captured its finite
nonlinear MLP write even at identical P inputs. Here distinguish instantaneous
response from the change in that response along the edit. For recorded local
operators define

    J(v) = Wd[a*(Wg v) + b*(Wu v)]
    aF = saved False u*SiLU' coefficient; bF = saved False activation
    aE = saved P u*surrogate_slope coefficient; bE = saved P activation
    D = JE-JF, with coefficients (aE-aF,bE-bF).

Use the recorded Float32 coefficients promoted to Float64 before subtraction,
and pinned checkpoint weights converted to Float32 then Float64. This defines a
linear operator exactly in Float64; it is not a whole-model Jacobian or a claim
that the derivative of the rounded program exists. Earlier derivative failures
remain failed. The ideal smooth path-integral identity motivates the question;
D is an endpoint difference, NOT that integrated operator or a complete generator.

## Fixed comparison

At each SAME saved P point and same target n from293, construct four bases:
FALSE_RANGE from JF, ENDPOINT_RANGE from JE, CHANGE_RANGE from D, RANDOM_RANGE
from orth(Omega). Omega is the actual saved293 Gaussian input. Rank 32 and two
power iterations are unchanged. CPUFloat64 reduced QR and dense operator
applications are used for every learned basis, so the comparison is numerically
matched. Source/verdict are processed separately; missing intermediate rows are
not fabricated or treated as measured. This is not an optimal SVD.

For learned M, set Q0=orth(M Omega), then twice
Q=orth(M orth(M^T Q)). Project n orthogonally and retain its signed target
projection, norm/dot/error sufficient statistics, and the full unselected vector.
Do not use n or the final readout to choose the orientation. n is used only for
projection and scoring, as in293. It is a supplied P-state conditional target;
it is not a newly forecast native write or an information saving.

## Frozen decisions

All 48 points remain in every denominator; norm<.001 means failure.

H1 PRIMARY: CHANGE_RANGE relative vector error must be <=half the BETTER of
FALSE_RANGE and ENDPOINT_RANGE errors at the same point. Require>=36/48 overall,
AND>=18/24 separately at source and verdict. Thus a favorable pooled median or
improvement against only the weaker endpoint is insufficient.

H2 separate complete-vector account: CHANGE_RANGE relative error<=.25 and
cosine>=.95, conditional target norm>=.001, with the same 36/48 and18/24 rule. H2 cannot
rescue H1. Report all four ranges, all layers, and both H1/H2 results. Random
projection is a dimension-matched context control, not a norm/manifold control.
No changed rank, extra power iteration, seed search or omitted difficult point.

## Numerical qualification

Before production, tiny unequal-shape fixtures must independently verify the
operator against explicit dense JE-JF, complete power iterations, signed
projection/orthogonal closure, and reject a sign-corrupted operator witness and
an orthonormal but wrong final range. Scientific fixture performance is not
required. Production source hashes and input/checkpoint/library identities bind
all calculations; no original scientific source may change.

Required production checks at ALL 48 points:
- independently reconstruct J/JT applications and the complete final range with
  TorchDouble; elementwise max error<=1e-8 for operator applications and
  final saved-basis distance to independent range<=1e-6;
- verify D(Omega)=JE(Omega)-JF(Omega) and transpose difference within1e-8;
- all basis orthogonality, signed projection/orthogonal closure and Pythagorean
  identities within1e-10 absolute +1e-11 relative;
- new CPU64 False/current ranges must match their ORIGINAL293 saved FP32/MPS
  ranges in span distance<=1e-3. Record all comparisons. This is span agreement,
  not bitwise replay or retroactive validation of other kernel shapes.

No SiLU, activation-gradient or disputed whole-block derivative is newly
attributed. Use explicit local linear operators whose coefficients already
have scoped293 qualifications. New TorchDouble checks are arithmetic/range
checks, not independent replication of the original learned trajectory.

Initial free-space>=512 MiB, per-layer>=384MiB, new archive total<=192MiB.
Read checkpoint tensors directly; no full model is instantiated. Freeze and
verify tiny qualification before checkpoint use. Preserve all stops/failures.
A passing local result would need a separately frozen own-state causal
intervention and fresh prediction before any stronger claim. No such later
experiment is part of294.
