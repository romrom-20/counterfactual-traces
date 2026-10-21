# 253 — query normalization curvature versus displacement feedback

252 supports a quantitative role for receiver query feedback, but its best
query-live/key-False forecast still has12.4% whole error. Test whether the
query updates require finite RMSNorm curvature or can be predicted by a
False-anchored tangent as the forecast residual evolves. Also separate this
query-generator approximation from omitting receiver-key feedback.

Freeze32 new literal pairs before model forwards. Same dependent templates,
one pinned Qwen checkpoint, matched False/True lengths and decoder.64 hashes
must be absent from the published catalogs including251/252. No learned
features, novelty priority, cheap predictor or generalization claim follows.

## Declared query predictor

For a receiving row, cache False residual x0 and actual Float32 normalized
query input z0. Let s0=sqrt(mean(x0²)+eps), gamma the learned RMSNorm weight,
and d=x_forecast−x0. The tangent query-input rule is

`z_hat = z0 + gamma * (d/s0 - x0*mean(x0*d)/s0**3)`.

Compute coefficients/increments in CPU64, cast installed z_hat toFloat32,
then use the ordinary q_proj (including bias) and rotary map. Only receiver
query-projector inputs change. Keys, values and MLPs retain their own finite
normalization unless receiver keys are explicitly clamped below. The source
and causal prefix remain fully finite; no native True source input is supplied.
This is an anchored smooth RMSNorm tangent, not an autodiff certificate for
the cast-containing Float32 Qwen implementation or any whole-block Jacobian.

- **A, primary:** tangent receiver queries, live receiver keys.
- **B, secondary:** tangent receiver queries, False receiver keys.
- **C, secondary:** finite receiver queries, False receiver keys (252 Q rule).
- **T, measured only after forecast seal:** finite queries and keys, native True.

All values, source computation, MLPs, residual updates, softmax and final full
vocabulary head stay finite at each arm's own state. No oracle receiving states.
The finite interaction T−A−C+B separates the two policy factors, but is not a
unique learned-circuit allocation because internal trajectories differ.

Primary A: whole-receiver AND verdict displacement errors<=.05, cosines>=.99,
native norms>=.001 and absolute margin error<=.10nats;24/32 overall,6/8 each
schema/order stratum. B/C share the unchanged secondary criterion and never
replace A. No layer/neuron search or fitting. Nonfinite or tail norm>100times
False stops with failure retained.

## Controls, numerical qualification and measurement order

Ordinary F captures anchors. Zero control Z uses tangent query inputs and
full-shape receiving attention recomputation on the False input; all28 tails
and scores must replay F exactly. Source/prefix query inputs remain unchanged.
After global forecast seal and public remote receipt, run native T and O, a
current-QK recomputation control; O must replay T exactly. All A/B/C source
rows and all three first-block tails must match native T exactly.

Before any native measurement, a separate CPU64 auditor qualifies the defined
RMSNorm tangent using centered differences at1e−3 and1e−4 on the actual used
forecast directions (small-step error<=1e−5). Independently reconstruct the
installed query input exactly after Float32 casting. Use public checkpoint
query weights/bias, saved rotary factors, K/V and mask to reconstruct the
complete query→QK→softmax→PV path; its probabilities/outputs must agree within
1e−4. Check endpoint heads at .005. Raw query-projection, rotary and scaled-QK
coordinate differences are recorded separately with their original1e−4 bounds,
including failures; large-logit failures do not silently become passes. This
inherits251b's explicit composed-output boundary, not universal coordinate
agreement or full-forward numerical reconstruction.

Witnesses: exactly the first case of each(schema,reverse) tuple, blocks1/14/28,
all7arms F/Z/A/B/C/T/O. Every forecast case/arm/layer is retained. Cache
False r/z and receiver K, plus all tail trajectories. All native True outputs
remain unopened/unmeasured until all forecasts pass pre-audit and their seal
is publicly committed/remote verified. Reproduction uses explicit local proof
and is not fresh confirmation. Unexpected failures stop; no tolerance tuning.

## Budget

Forecast phase:160 full forwards(F/Z/A/B/C). Tangent inputs inZ/A/B add2,688
CPU64 row-map evaluations across blocks/cases. Z/B/C add2,688 extra full-shape
QK/softmax/PV products each; A uses its ordinary native attention with modified
queries. Measurement/control:64 full forwards(T/O) and896 extra full-shape
QK/softmax/PV products each. Total84 extra witness QK products. Original
attention calls also run; audits are additional work. No speedup claim.
Code, sources, all cases, budgets, failures and receipts are published; model
arrays and unpublished article drafts remain local.
