# 280: does the first-source curvature rule compose?

Status before execution: retrospective, fixed primary. The 32 contexts are the
already measured 279 literals (16 dependent background pairs). No fresh or
independent generalization claim. Freeze this protocol and all scientific sources
before checkpoint execution; never overwrite previous experiments or failures.

## Question and information

Does one uniform cubic source activation rule, anchored only at observed False
gates, generate its OWN 28-layer source trajectory and support the original native
receiving response? Known edit and weights, full False computation/prefix trace
are observed. Native True gates, source states, trajectories and scores are not
inputs to C/Q/L/C1. These are old examples, so execution order alone is not a
prospective seal. This study is not a learned circuit or complete Question B claim.

Source causality means no later receiving row can influence the source. Execute
full-shaped forwards, changing ONLY the source activation at the specified layers.
Receiving computation, input RMSNorm, attention/softmax, gate/up/down projections,
products and final vocabulary readout are finite. This is not a work reduction.

## Fixed arms

F/T: unhooked-value native False/True controls (observation hooks only).
C PRIMARY: all 28 source activations use the False-coefficient cubic from 279.
Q/L: same at every layer with quadratic/linear activation, fixed controls.
C1: cubic at first source MLP only, all later source activations finite; must replay
279 C's source1/final/scores. FZ: C with False input, all layers exactly False.

Before SiLU in approximate source sites, replace ONLY that row's actual gate by
its observed False gate. Its actual SiLU primal must match observed False exactly.
Then replace the activation row by the polynomial evaluated using the originally
captured CURRENT gate and False coefficients. Thus no current finite source
activation at approximated sites is secretly supplied. Other rows remain native.
C uses zero current finite source SiLU sites; C1 uses 27. Padded full forwards
still perform source False and all receiving activations. All earlier prefixes
must replay exactly, including source K/V prefix hashes. No whole-block Jacobian.

## Primary endpoints, with every case counted

Native target T-F, excluding source in final receiving vectors: whole receiving
and verdict row each norm>=.001, relative error<=.05, cosine>=.99, finite actual
margin error<=.10nat. Source target is the concatenation of post-block T-F source
states over all28 layers: norm>=.001, relative error<=.10, cosine>=.995. Emitted
source V target is concatenated T-F values over28layers/2heads: same .10/.995 and
material norm. K is recorded and reported, not an added primary requirement.
Joint C needs ALL these, >=24/32 and >=6/8 in each schema/order stratum. Failures
and ineligible targets count in32. Q/L/C1 cannot rescue a failed C primary.
Do not replace original native response with a looser source-only success.

## New local error versus inherited input

After every base forecast, evaluate local finite source activations at the
captured C gates/up, and local cubic activation at captured native T gates/up.
These are explicit retrospective diagnostic computations, NOT prospective inputs.
For layer l let c_l/t_l be forecast/native post-block source. Let x_l be the
finite MLP output plus the actual C attention midpoint, rounded as public Float32.
Then in Float64 accounting exactly:

    t_l - c_l = (t_l - x_l) + (x_l - c_l).

The second vector is the new local activation approximation discrepancy; the
first is the effect of inherited input under the finite block (source prefixes
are unchanged). It is NOT a linear Jacobian error transport law. Record signed
projections, cosine, norms and cross-terms. Verify closure<=1e-10. Compare the
native-input local cubic write discrepancy with C-input discrepancy. Do not turn
large projections into causal shares or count medians as additive quantities.

## One targeted causal correction; not a correction sweep

Using only C's own-input local finite-write diagnostics, select the layer with
largest missing-write Euclidean norm (ties first index). R makes ONLY that source
activation finite, all other source activations cubic on its OWN new state.
G0-G2 keep C but add independently seeded random output-write directions at that
same layer, norm-matched to the selected missing write before Float32 rounding.
Selection has no native True or final-response information but explicitly spends
28 local finite activation/product/down evaluations; this is a retrospective
necessity/sufficiency diagnostic, not a cheaper prediction.

R must match C's incoming state/gate/up at that layer, remove its local discrepancy
exactly, and preserve all earlier source states. All controls must do so too.
For each, record original native and trajectory outcomes. Separately test whether
R-C reconstructs missing T-C receiving/verdict vectors at .25/.95/material .001
and actual margin effect>=.02 with error<=.005+.25*abs(target), while >=2/3 random
controls fail. Require24/32 and6/8 strata for that stronger account; insufficient
material effects remain failures. It cannot rescue the primary composition test.
No claim of necessity just from better precision or random failures alone.

## Numerical scope, storage and cost

Independent TorchDouble autograd and two finite-difference steps qualify first,
second,third derivatives at all896 False layer anchors (2688 coefficient vectors),
required<=1e-4. All polynomial installs, actual False primals, unchanged prefixes,
public Float32 product and full-shaped down projection replays must be exact.
Final full-vocabulary public heads must replay exactly; CPU64 bridges<=.005.
All CPU64 raw phi/product/write diagnostics retain original1e-4 bounds and failed
rows. Those raw intermediate diagnostics are NOT all required, and no old
whole-block derivative/raw-logit failure is recertified. State all counts.

352 full forwards (11 per case), 1792 extra local diagnostic product/down maps
(C finite and T cubic), of which896 also evaluate finite SiLU (C), plus explicit
numerical audits. Additional
public replays in the audit are counted. Only source primitive rows/source states,
final receiving states/scores and prefix hashes stored, not whole intermediate
receiving tensors. Raw cap1200MiB; require1536MiB initial and768MiB per-case free
space, model/process headroom additional. Never delete old evidence to continue.
If a primitive qualification fails, preserve failure and do not make a new
attribution claim. An independent code path within one investigation is not an
independent team or whole-model implementation.

A stable result can motivate a SEPARATELY frozen fresh test, whose entire forecast
and numerical qualification must be sealed before ordinary True measurement.
