# 256 — one-pass polynomial receiver-query forecast

255 is a retrospective algebra diagnostic: the quadratic term misses about
25.5% of the local native RMSNorm remainder; adding its cubic term leaves
about 5.37%. This motivates a fixed **cubic primary C** on 32 new literal pairs.
All cases, code and criteria are frozen before any model forward. No native
True receiving measurements or preceding approximate trajectory enter forecasts.

## Rule and information budget

One ordinary False pass caches each receiving residual x0 and actual normalized
query input z0, with known model weights gamma and eps. At each forecast layer,
let d be its own incoming residual minus x0, s²=mean(x0²)+eps,
a=mean(x0*d)/s² and b=mean(d²)/s². Let J0*d be the smooth RMSNorm tangent.

`Q2 = gamma/s * [(1.5*a²-.5*b)*x0-a*d]`

`Q3 = gamma/s * [(1.5*a*b-2.5*a³)*x0+(1.5*a²-.5*b)*d]`

- **C, primary:** install z0+J0*d+Q2+Q3 before receiver q_proj.
- **Q, secondary:** z0+J0*d+Q2.
- **L, baseline:** z0+J0*d.
- **R, directional control:** tangent plus seeded Gaussian vectors matched to
  Q2's norm per row/layer at R's own input. Seed SHA256 of
  casekey:layer:local-query-quadratic-control. No projected-query or manifold match.

Each arm evolves in one forward pass; no complete seed/correction trajectories,
exact finite query remainder, gain fitting or selected sites. CPU64 coefficients,
Float32 installation. Receiver K/V, softmax, MLPs, source/prefix computation and
readout remain finite at their own states. The source is computed from the known
edit, not supplied. The ordinary shared normalization executes for K/V; its
current normalized query input is not used by the polynomial query forecast.
This remains a rich finite-component budget, not a complete learned mechanism.

## Frozen criteria and controls

Joint whole-receiving AND verdict relative displacement errors <=.05, cosines
>=.99, native displacement norms >=.001, absolute finite margin error <=.10 nats.
Success requires >=24/32 and >=6/8 each schema/order stratum. Apply the identical
rule to secondaries, never promote them after outcomes. Fresh literals must have
64 hashes absent from all published catalogs/case fixtures; dependent templates
and one pinned checkpoint remain the scope. Preserve nonfinite or tail-norm
>100 times False failure; no silent retries or threshold revision.

F ordinary False and Z zero-edit tangent must replay every tail/score exactly.
Global forecasts, separate pre-audit and public remote seal precede native T
and current-QK replay O. Source trajectories and first-block tails must match T
exactly after measurement. Final readout stays finite.

## Numerical qualification

Save installed query inputs at every arm/layer. A separate auditor reconstructs
all local inputs and random controls. Compare linear/quadratic/cubic coefficients
to a separate generic polynomial-jet recurrence within 1e-10. Witnesses are first
case per schema/order stratum, blocks 1/14/28, all eight arms. At modified inputs,
qualify the smooth tangent by centered differences (1e-3 step bound 1e-3; 1e-4
step bound 1e-5), and quadratic coefficient by second centered differences
(1e-3 step, 1e-5 bound). This does not recertify old whole-block derivatives.

Reconstruct public-weight query projection, rotary, scaled QK, softmax and PV.
Composed probabilities/outputs must agree within 1e-4, finite heads within .005.
Raw projection/rotary/QK absolute coordinate comparisons retain original 1e-4
bounds and failures are separately recorded, not relabeled passing. No claim of
independent full-forward reconstruction; no learned features from Taylor terms.

## Computation and interpretation

192 forecast full forwards F/Z/L/Q/C/R, then 64 native/control T/O. 4,480 CPU64
tangent maps, 2,688 quadratic maps, 896 cubic maps, 1,792 extra full QK/softmax/PV
products each in Z/O, 96 witness QK products; separate numerical audits additional.
The primary algorithm uses one False pass plus ONE forecast pass, versus 254's
False+seed+two-correction passes. Dense K/V/MLP/source work remains; no useful
speedup over ordinary edited inference is claimed. Success would remove the
need for complete correction trajectories in this query-path account, not solve
general Question B, learned-circuit identification or task/checkpoint generality.
