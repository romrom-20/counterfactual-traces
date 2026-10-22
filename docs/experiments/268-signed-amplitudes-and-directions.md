# 268 — Are adaptive signed amplitudes enough on fixed correction axes?

Freeze source, helper, auditor and this protocol before new study tensors.
Four OLD first-per-schema/order263 contexts, all indigo; no fresh confirmation.
Question B remains open. No browser/literature lookup or whole-block Jacobian.

267 retained the final policy interaction when correction generation adapted
at a frozen L-forecast MLP operating point. We now distinguish signed amplitude
from direction **inside that generator**. Freezing raw displacement would also
freeze its length, so that is not the directional control used here.

Let h=x-xF, s²=mean(xF²)+eps, v=gamma*h/s, n=||v||₂, u=v/n (u=0 when n=0).
Let a=mean(xF*h)/s², b=mean(h²)/s², rho=(1+2a+b)^(-1/2).
The263 normalization corrections have the finite identities:

```
p = alpha*f0,       alpha = rho-1+a,   f0 = gamma*xF/s
d = k*u,           k = (rho-1)*n
```

k and alpha are SIGNED amplitudes; a sign change belongs to this factor, not
an arbitrary flip of the unit vector. p's axis f0 is always fixed. d's unit
axis u can change. At each layer cache L's z*,p*,d*,alpha*,k*,u* from the same
False-plus-known-edit tangent forecast as267. Never use native True inputs.

All correction responses use the frozen local point z*:
cP=M(z*+p)-M(z*), cD=M(z*+d)-M(z*). All neurons use finite SiLU. The common
M(z(x)) backbone and attention still adapt; query normalization is the qualified
cubic rule, source/KV/softmax/projections/products/head are finite.

Crossed families (these digits mean AMPLITUDE/DIRECTION, unlike267's digits):

```
11 both adapt:        p(x),       d(x) [exact old26710 arithmetic]
10 amplitudes adapt:  p(x),       k(x)*u*
01 direction adapts:  p*,         k*  *u(x)
00 both frozen:       p*,         d* [exact old26700 arithmetic]
r  random fixed axis: p(x),       k(x)*u_random
```

Random unit axes are seeded per case/layer/row, held fixed across its complete
continuations, with D correction INPUT norms matched at each own current input.
This is not a matched MLP-write/product control. If an axis is undefined while
its required amplitude is >1e-12, record a guard failure; never choose an
outcome-dependent substitute. All zero cases and guards remain in denominator4.

Run P,D,A per family, where A adds both finite correction WRITES to the common
backbone. Float64 sums of Float32 operands, one final Float32 write rounding.
80 full forwards: F,Z,L,H, P/D/A×11/10/01/00/r, T for each old context. Z verifies
the zero-edit identity. H must replay258C; F/T replay258F/T. New11 must exactly
replay26710 (x/qx/final/scores and per-layer receiving hashes);00 replays26700.
Identical P policies in11/10/r must replay each other exactly.

## Frozen primary and interpretation

Target I11 = finalA11-finalP11-finalD11+finalL (thus old267 I10).
Each family must match whole and verdict I11 within25% relative error and.95
cosine, both targetnorms>=.001, plus the same finite common-anchor margin rule:
anchorA11, endpointsFloat32(anchor-I_s), public finite final norm/fullvocabhead,
error<=.005+.25*abs(mAnchor-mTargetEndpoint). Preservation needs>=3/4.

Primary hypothesis:10 preserves in>=3/4 and01 in<=1/4. Report every family,
including possible joint or redundant sufficiency; do not promote a secondary.
Random preservation prevents claiming the particular frozen D axis is necessary
for this policy interaction under the live P correction. Its failure is a
direction control only under the specified input-amplitude matching.

Native-response calibration requires H and11's A to each meet the original
whole/verdict5%/.99/.10nat rule in4/4. Score all policy arms against native
separately. If10 preserves the interaction but fails native accuracy, state that
boundary. No inference of a learned feature/circuit or general cheap predictor.

## Numerical scope and preservation

Pinned258 checkpoint/runtime, MPSFloat32/eager, CPU64 scalar/geometry work.
Bind scientific/dependency/protocol/checkpoint/Qwen source/input hashes. Preserve
all old failures. Independently qualify the smooth RMS formulas with261b's
CPU64 AD1e-9, cubic jets1e-10, centralFD1e-3/1e-4 at1e-3/1e-5.
Check p=alpha*f0 and d=k*u within1e-10; unit norms1e-12; crossed D input norms
within1e-10 CPU64 and1e-5 relative afterFloat32 rounding for norms>=1e-8.
Actual public finite norm/MLP nodes, installed writes, receiving residual
additions, prefix/source isolation and actual finite head replay must be exact
at original full-sequence shapes. Do not relabel raw CPU64 gate/up/down1e-4 or
finite-head.005 failures. No older whole-block derivative is recertified.

New output directory only; immutable originals. Save receiving x/qx, False
normalized anchors, final tails/scores, all other node/trajectory hashes,
signed contrasts and every failed guard. Reserve512MiB free and cap raw260MiB;
check headroom before starting each case and retain metadata before post-write
storage assertions. Never delete old evidence to make space. All reports,
release/plot scripts and figures will be labelled post hoc.

If the primary and native accuracy succeed, a separately frozen prospective
new-literal test can follow. Its rule, prediction budget, primary/control arms
and source seals must be fixed before any new native True trajectory is read.
