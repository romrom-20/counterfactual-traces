# 292: a coherent False-anchored gate transition through the native trajectory

Freeze this protocol, code, cases and history catalog before checkpoint use.
Run and pass the complete unequal-length CPU pipeline and coherent negative
fixtures first. Then predict all 32 fresh cases, independently qualify them,
seal and publish the forecasts globally, and only then measure native True.
Preserve all old failures and do not call general Question B solved.

## Hypothesis and relation to earlier work

291 rejects all six partial receiving coefficient accounts. A scalar calibration
can pass despite wrong vectors. 280's uniform cubic source forecast fails from
MLP2 onward, and 258 already forecasts receivers using quadratic plus selected
finite activations with finite source computation. Repeating those tests would
not resolve the new question.

The new hypothesis is that a False-value-and-slope anchored rational gate
transition, retaining joint gate/up changes, predicts the entire affected source
and receiving trajectory without evaluating finite edited SiLU anywhere.
It is a fixed analytic update, not a learned feature or a novelty claim for
rational approximation. It still spends finite attention, normalization,
projections, multiplication and readout; no cost advantage is assumed.

## Information and fixed update

Use one ordinary full False observation. Capture affected gate gF, up uF,
actual Float32 activation phiF and product pF at every MLP. The edited input
token and weights are known. All forecast states come from the candidate's
own preceding computation. No native True source state, receiving state,
native edited coefficient or earlier accurate forecast is supplied.

For m doublings define the bounded tanh core:

    x = g / 2^(m+1)
    t = clip(x*(15+x^2)/(15+6*x^2), -1, 1)
    repeat m times: t = 2*t/(1+t^2)
    R_m(g) = g*(1+t)/2

P (PRIMARY) uses m=2, fixed before any checkpoint forecast. Let d1F be the
smooth native SiLU derivative at gF, computed only from False. At a current
predicted gate g=gF+dg:

    lambdaF = d1F - R_m'(gF)
    phi_hat = phiF + R_m(g)-R_m(gF) + lambdaF*dg
    slope_hat = R_m'(g) + lambdaF
    a_hat = current_up * slope_hat
    b_hat = phi_hat

This matches the observed False value and smooth False slope. The value and
slope are predictions of the same function. Current gate/up projections are
finite at the candidate's own normalized residual. Cast phi_hat to Float32,
multiply by the current up value and use the actual public down projection.
Every affected source/receiving MLP, including block 1, uses the rule.
Only prefix rows strictly before the edit retain their independent finite
activation. No selected finite edited exceptions, controllers or true gradients.

The internal tanh core is bounded. Anchoring adds a small constant/linear
correction, so P is not claimed to preserve the exact global SiLU range.
Its coefficient derivatives describe an anchored smooth surrogate before the
installation cast, not a whole-model Float32 Jacobian.

## Frozen controls

All controls evolve from their own states with the same known edit:

- D: same anchored rational rule, m=0 (coarse transition).
- C: original False-anchored cubic SiLU at every affected MLP, with the current
  up branch and finite joint product. No finite edited activation.
- B: C clipped pointwise to max(min(g,0),-.278465) <= phi <= max(g,0).
  This tests whether basic range constraints alone recover the response.
  B's unclipped derivative is not claimed as its effective derivative.
- A: P's phi rule, but replace each affected product by the observed pF plus
  uF*(phi_hat-phiF) + phiF*(current_up-uF), omitting the joint change.
  The normal predicted product is computed then discarded; count that work.

For each arm, replace affected gates by observed gF BEFORE the actual public
SiLU executes, check its output against phiF, then install the prediction.
Thus no current edited SiLU primal is evaluated, even if its write is discarded.
Record this witness at every layer and preserve prefix identity.

Each candidate also runs a full False-zero control; every block state and
primitive node must replay observed False exactly. Ordinary native T is allowed
only after all forecasts and their qualification are globally sealed/pushed.
Full T is calibration, not a forecast.

## Fresh cases and seals

32 new complete False/True input pairs: rhodonite conservatory and carnelian
planetarium facts, Tuesday/Saturday, in the four table/user-rule/dialogue/XML
prefix layouts. Each layout crosses schema, fact order and question target.
All 64 full-token hashes must be absent from the available historical catalog
and unique in this batch. These remain dependent synthetic templates in the
same checkpoint, not 32 independent tasks or a competence demonstration.

All source bytes, cases, history, installed model-library hashes, checkpoint
revision and runtime are recorded. A write-once pre-native audit and seal bind
every forecast archive/metadata file globally. Production native entry checks
the exact pushed branch head and published seal before the first True call.
Synthetic CPU receipts cannot authorize a checkpoint native run. A changed
bound, algorithm or resource policy requires a separate named qualification.

## Frozen scientific decisions

Evaluate all 32 cases, including ineligible effects. Population pass is at least
24/32 and at least 6/8 in EACH layout. P is primary; controls cannot rescue it.

H1 native prediction: receiving displacement strictly after source AND final
verdict displacement each require native norm >=.001, relative error <=.05 and
cosine >=.99. Actual finite true-minus-false margin error <=.10 nat.

H2 trajectory: additionally require .10 relative/.995 cosine/.001 norm for
the stacked all-28 source-state trajectory, all receiving-state trajectory,
and source/verdict MLP-write trajectory. H2 also requires H1. These concern
native trajectories, not approximation remainders.

H3 joint-product effect: compare forecast P-A with the unknown native T-A
receiving/verdict displacement. Require .25 relative/.95 cosine/.001 norm for
both, actual finite margin-effect error <=.005+.25*abs(target). Require abs(T-A
margin effect)>=.02 and A to have either native receiving error >=.10 or margin
error >=.20. Ineligible cases remain failures. Report exact T calibration and
all control outcomes separately; a passed H1 cannot rescue a failed H3.

Post-seal local coefficient diagnostics at six fixed source/verdict probes
(blocks 1/15/28 for 28 layers) compare the two predicted compound coefficients
with finite SiLU at P's SAME predicted inputs. All metrics and .001 eligibility
are retained. These are local prediction diagnostics, not native True parents
or a learned circuit. Do not promote them over H1/H2.

## Numerical qualification and actual work

Actual model paths use full sequence shapes, with no whole-block derivatives.
All prefix, False-primal substitution, zero-edit and installation replays are
exact; product witness bound1e-4; independent CPU64 finite head bound .005.
Pre-native audit replays F/P/P-zero without current edited SiLU and reconstructs
every saved gate/product witness independently. At False gates, check native
SiLU derivatives by Torch Double AD. At False and P gates, independently check
the rational anchored function/slope with Torch Double AD (1e-8) and finite
differences 1e-4/5e-5 (2e-6), without evaluating native edited SiLU.
These qualify specified primitives, not older disputed whole-model Jacobians.
Every historical failed check remains failed.

Before checkpoint use, four unequal-length CPU cases must pass the full
forecast/qualification/seal/native/audit pipeline. Bounds, failed qualification,
a corrupted LAST archive, wrong receipt and coherently tampered primitive
witness must all be rejected before unauthorized native measurement. Scientific
performance is not required on random-model fixtures.

Budget: per case one ordinary False, five full forecasts, five False-zero
controls; numerical qualification adds F/P/P-zero replays; after seal one
native True and one independent unhooked True replay. All dense projections,
full attention, norms and heads execute. Host Float64 rational/Taylor arithmetic
and device copies also count; the method is not a demonstrated speedup.

New guards: initial free space512MiB, per-case384MiB, combined forecast/native
case archives256MiB. Stop and preserve any failure; do not delete/recompress old
scientific files. Publish source, complete cases/scalar evidence, seals, cost,
figures and omissions. Raw arrays/weights and the private article stay local.
