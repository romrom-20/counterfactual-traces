# 254 — forecast query curvature from a previous forecast

253's smooth False RMSNorm tangent passes numerical checks but misses the
complete native displacement. Test a fixed two-iteration defect-correction
rule. This is a prospective predictor of the missing query-path term, not a
native-state repair or a search over correction sites/gains.

Use32 new literal pairs/64 hashes absent from the catalog including253. Same
dependent templates, pinned Qwen checkpoint, matched lengths, decoder and
criteria. Freeze code and input fixtures before any forward. All source/prefix,
receiver keys/values, MLPs and readout remain finite at own evolving states.
Only receiver-query normalization is replaced.

## Fixed algorithm

Cache False x0,z0,gamma,eps at all28 layers. D0 is253's tangent-query rule:
`z0 + J_RMS(x0)*(x_current-x0)` (CPU64 coefficients, Float32 installation).

For a completed seed trajectory s, compute its query-normalization remainder
at each receiving input using only that forecast and False:

`E(s) = f(x_s)-f(x0)-J_RMS(x0)*(x_s-x0)`,

where `f(x)=gamma*x/sqrt(mean(x*x)+eps)`. For block1 x_s=x0; later inputs are
the seed's saved previous-block output, not native True. This is finite RMSNorm
geometry, not two competing learned mechanisms. Do not infer a semantic feature.

Run **P1** with tangent queries at its own evolving states plus frozen E(D0).
After P1 is complete, compute E(P1). Run **P2, primary** with tangent queries
at its own states plus frozen E(P1). Do not update either correction using the
current pass's states. Exactly two iterations, no gain fitting or case selection.

**R** is a matched directional control: tangent queries plus deterministic
Gaussian vectors matched to E(D0)'s norm separately for every receiving row
and layer. Seed SHA256(casekey:layer:query-curvature-control). This control
matches normalized-query-input norms, not projected query effects or manifold
validity. D0/P1/R are secondaries, never replacements for P2.

Primary P2: whole-receiver AND verdict displacement relative errors<=.05,
cosines>=.99, native norms>=.001, absolute margin error<=.10nats;24/32 overall
and6/8 each schema/order stratum. Report identical secondary criteria and
per-case error changes versus D0/R. A successful scalar margin alone is not
success. Stop/preserve nonfinite or forecast tail norm>100times False.

## Qualification and order

F ordinaryFalse; Z zero-edit tangent/zero-remainder control exact at all28 tails
and scores. After all global forecasts, separate pre-audit and public remote
forecast seal, measure native T and current-QK O, which must replay exactly.
All forecast source rows and first-block tails must match native True exactly.

Independently reconstruct every E(D0),E(P1) and random control from the saved
seed trajectories and False coefficients. Verify exact Float32 query-input
installation, same CPU64 smooth tangent finite differences as253, and the full
public-weight query projection/rotary/QK/softmax/PV composed probabilities and
outputs within1e−4. Heads .005. Raw coordinate checks retain original1e−4
thresholds with failures recorded, not relabeled as passed. No whole-block
Jacobian or claim of full independent forward equivalence.

Witnesses: first case per(schema,reverse) tuple, blocks1/14/28, all8 arms
F/Z/D0/P1/P2/R/T/O. All case/arm/layer outputs and source hashes retained.
Fresh literals still share templates and one model. No independent-team,
learned-circuit, novelty-priority or general verification-competence claim.

## Budget and interpretation

192 forecast-phase full forwards(F/Z/D0/P1/P2/R);64 measurement/control
forwards(T/O);4,480 CPU64 tangent maps;1,792 seeded curvature-map evaluations;
1,792 extra full-shape QK/softmax/PV products EACH (Z/O);96 witness QK.
Original attention calls also execute. Numerical audits are additional work.
This is MORE work than an ordinary native edited forward, not a speedup or
the compact single-state account sought by Question B. A success establishes
predictability of this defined nonlinear query correction under the stated
rich component budget, not the unexplained learned generator of every state.
Publish all outcomes/code/receipts; arrays and article drafts stay local.
