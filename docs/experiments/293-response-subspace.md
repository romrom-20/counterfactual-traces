# 293: can a small response subspace carry the nonlinear MLP write?

RETROSPECTIVE diagnostic on eight already observed292 contexts: the two schemas
with order0/target0 in each of four layouts. No new native True observation.
Freeze these source bytes before checkpoint use. Preserve292 and all old failures.
General Question B remains open. No learned features or speedup are claimed.

## Question and distinction from earlier tests

292's near-exact activation surrogate still does almost the whole dense forward.
Repeating normalization/activation/product ablations would add little. This test
restricts the RESIDUAL DIRECTIONS of each nonlinear MLP write to32 dimensions,
at every affected source/receiver row and every block, while leaving the
False-linear contribution unrestricted. Does adapting those directions to the
candidate's own state work where a fixed False-derived subspace does not?

The bases use the learned normalized-input MLP response operator

    J(v) = Wd[(u*slope)*(Wg v) + phi*(Wu v)].

For fixed basis F, u/phi/slope are observed False values/smooth False SiLU slope.
For adaptive basis E, they are the candidate's own up value, installed rational
activation and coherent292 surrogate slope. These are defined smooth local
operators, not whole-model or bitwise Float32 Jacobians.

## Basis and finite update

Use k32 (tiny CPU fixtures k=min(32,hidden//2)). A seeded Gaussian input basis
Omega has entries N(0,1/hidden); seeds are SHA256(casekey/layer/row/293). The
same Omega is used for False/adaptive bases and isotropic controls. Form
Q0=orth(J Omega), then twice Q=orth(J orth(J^T Q)). orth is reduced NumPyFloat64
QR; reject rank loss below1e-8 relative diagonal scale. J/J^T run explicitly
through Float32 public weights on the model device, without autograd. Final
basis is Float64 and must be orthonormal within1e-10. This is a randomized
response range, NOT an exact SVD, optimal rank32 space or learned semantic basis.

At each own predicted MLP input, use292 P's anchored rational activation and
actual finite gate/up projections, normalization and attention. Define

    dp_linear = uF*d1F*(g-gF) + phiF*(u-uF)
    l = public full-sequence-shape Wd(Float32(dp_linear))
    n = current public rational MLP write - observed False write - l.

n is a finite-operation remainder including explicit projection/cast rounding;
it is not asserted to be pure ideal-real curvature. The installed write is
False_write + l + chosen_remainder, cast once toFloat32. Prefix rows unchanged.

Five conditions, each from its OWN evolving state:
- P: complete rational MLP write, unchanged292 control; no write replacement.
- L: chosen_remainder=0 (False-linear-product reference).
- F: chosen_remainder=Q_F Q_F^T n; Q_F cached from False only.
- E PRIMARY: chosen_remainder=Q_current Q_current^T n.
- R: project n onto orth(Omega), then rescale to the norm of Q_current Q_current^T n
  at R's OWN input. Zero target norm gives0; a zero random projection uses its
  first basis axis. This matches retained correction norm, not native accuracy,
  manifold validity or any learned feature. Adaptive-basis work counts for R.

All candidate SiLU calls receive saved False gates BEFORE execution; no current
finite native SiLU is used. All dense operations and full heads still execute.
This tests a dimension restriction, not a computation saving. No edited P
trajectory/basis/controller is supplied to any other candidate.

At P's same inputs additionally record signed complete/projection/orthogonal
remainder vectors for F/E, source/receiving/verdict scopes at all28 blocks.
These are conditional local diagnostics. Then the actual E/F/R/L interventions
above test their effects through evolving states and the finite final head.

## Frozen decisions

Original native rule: receiving strictly after source AND verdict displacements
norm>=.001, relative<=.05, cosine>=.99; finite margin error<=.10nat. Population
requires>=6/8 and>=1/2 in every layout. P must pass8/8 calibration.

Pair the schemas within each layout. Test their first-source-state displacement
contrast and final verdict-displacement contrast against saved native292,
norm>=.001, relative<=.10, cosine>=.995, paired margin-effect error<=.10nat.
Ineligible pairs fail. Paired population>=3/4; P must pass4/4 calibration.

The sufficient restricted account requires BOTH native and paired populations.
E is primary; alternatives cannot rescue it. The stronger adaptation-specific
hypothesis additionally requires F and R each<=2/8 native passes. Report every
case/pair/control and local norm guard; never call a large signed projection a
complete vector explanation. Do not tune k, power count or seeds after outcomes.

## Qualification, storage and scope

Eight F/P full arrays/scores must exactly replay292 before scientific scoring.
All five False-zero runs must replay292F exactly. Every installed write and
prefix is exact. Basis orthogonality1e-10; component closure1e-10. At the same probes, independently reconstruct the
complete power iteration in CPU64 and require saved-basis distance to its range
<=1e-3. QR batches and individual row QR calls are counted separately. At fixed
blocks1/15/28, source/verdict rows, independently reconstruct J/J^T applications
with CPUFloat64 original weights: absolute1e-4 +1e-5*maxabs(reference), actual
operand shapes retained. Independently qualify one seeded direction against
TorchDouble smooth local JVP and central differences1e-4/5e-5 at1e-6 absolute
+1e-5 relative. These are scoped primitives, not older disputed derivatives.
CPU64 final head bound .005; P/F/zero/head replays and all checks must pass.

Tiny unequal-length complete pipeline with a random CPU model must pass first;
scientific performance not required. Coherently tampered basis and changed
bounds must be rejected by the separate auditor. Production source commit and
107-source hashes, installed library/checkpoint hashes and every raw array are
sealed. Preserve stops. Initial512MiB/percase384MiB, archive cap256MiB.
No new native True runs, no browser/literature lookup. Raw arrays/private draft
stay local; publish code/protocol/scalars/figures/qualifications after audit.
