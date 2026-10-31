# 297: source attention allocation versus selection of other context

This is a native primitive-routing test after296, not another geometry or
surrogate-error sweep. Previous failures and source seals remain unchanged.
No learned feature, circuit, novelty priority or general Question B solution is
claimed. Freeze exact inputs, code, bounds and alternatives before checkpoint
execution; seal every forecast publicly before native True measurement.

## Gauge-invariant competing accounts

The observed False pass supplies post-RoPE receiving queries qF at all layers.
The known token edit initializes each candidate. At its OWN evolving state,
compute the current q, K and V. At each receiving query strictly after source s,
let p=softmax(q K^T/sqrt(d)+mask), b=softmax(qF K^T/sqrt(d)+mask). Both use CURRENT
keys. Define alpha=p_s and beta_j=p_j/(1-alpha) for non-source positions. Compute
beta by a separate source-excluded softmax to avoid dividing saturated mass.
The corresponding False-query quantities are alphaF,betaF.

- S, source-allocation route: alpha from current q, beta from qF.
- C, context-selection route: alpha from qF, beta from current q.
- B, frozen-query comparator: the full qF probability b.
- N, both routes: the full current probability p. Only the native phase executes
  a complete N trajectory (ordinary T and recomputed-attention calibration O).

Hybrid attention probabilities are alpha at s and (1-alpha)*beta elsewhere.
This is invariant to an arbitrary constant offset in either complete logit row.
An earlier unexecuted raw-column design was abandoned before freezing: it lacked
that invariance and could create artificial route effects from a common offset.
A tiny fixture must explicitly demonstrate that the adopted maps avoid it.

The source-vs-rest allocation alpha depends on the source logit AND the context
logsumexp. S therefore is not solely a source-key scoring intervention. C asks
which OTHER positions are selected at held source mass, including unchanged
causal prefix and other receiving positions. Do not call C exclusively unchanged
evidence. Source values and keys stay live in every arm. B can still change its
attention through current keys. Thus this separates two query-mediated probability
routes, not all source communication.

Source and earlier query/output rows are protected. Every norm, MLP, residual
and full-vocabulary head stays finite. Exact equal current/False queries bypass
the hierarchical rounding route and use p unchanged. This preserves zero edits
and first-block queries. The original attention, both full QK calculations and
conditional softmax operations still execute; no speedup is claimed. No native
True state, score or gradient enters any forecast candidate.

H1: S is sufficient (24/32 complete native passes and6/8 EACH encoding), while
C and B each have<=8/32. H2: C is sufficient under the analogous S/B restriction.
Neither alternative can rescue the other. If neither passes, reject both simple
sufficient accounts. No threshold, token-group or layer search follows by default.

Strict native success requires receiving AND verdict displacement norms>=.001,
relative errors<=.05, cosine>=.99, and actual finite-margin error<=.10nat.
All32 remain denominator. A separately reported conditional query-effect target
is T-B rather than T-F: relative<=.25, cosine>=.95, vector norm>=.001, finite
margin error<=.10nat AND |mT-mB|>=.02nat. It cannot rescue H1/H2. Report every
signed route projection and interaction T-S-C+B as well as full errors. This
interaction is between defined network policies, not a neuron feature. Different
candidate histories prevent unique additive attribution to one native circuit.

## Common-parent local diagnostic

Eight fixed cases (both schemas,order0,target0 for each encoding), blocks2/15/28,
save current/False queries, shared current K/V, mask, raw logits, complete and
conditional probabilities, and all four corner head outputs at identical parents.
Only T-parent witnesses enter the retrospective native local explanation.
First-block receiving queries must remain exactly False in every arm.

With cF=sum(betaF_j V_j) and Delta c=sum((beta_j-betaF_j)V_j), the exact real-valued
attention difference is:

    Delta y = Delta alpha*(V_s-cF) + (1-alphaF)*Delta c - Delta alpha*Delta c.

The first two terms are S-B and C-B; the last is the fixed-parent interaction.
This is ordinary gated-mixture algebra, not a new law. Apply saved attention
output weights in CPUFloat64 to the saved head-output differences. Measure their
signed projections onto N-B at every receiving position. Numerically bridge the
bilinear expression to the runtime corner interaction at1e-4. Also split the
fixed-parent N-B write by weighted-value contributions from prefix/source/later-
receiver key groups. That sum is algebra with fixed current values, not three
additional causal interventions. Zero-target rows stay recorded as ineligible.
No local projection licenses a final-margin claim without executed finite arms.

## Inputs, qualification, cost and stopping

32 fresh full inputs use four related evidence encodings, two schemas, two fact
orders and two targets, with fixed new literals. Freshness is against available
complete-input hashes. These are dependent synthetic templates in one pinned
Qwen2.5-1.5B-Instruct checkpoint; failed grounding controls remain failed.
Tokenization uses only the shared offline cache.

First qualify tiny unequal-length CPU Qwen fixtures through BOTH phases. Check
source/prefix protection, exact zero edits, logit-offset invariance, independent
hybrid maps, common-parent identities, CPU64 QK/softmax/PV, finite heads and source
trajectories. Coherent wrong-factor maps and source-state corruption must be
rejected. Exact installation/bank/control checks, hybrid-map error<=2e-7,
probability row-sum error<=1e-6, composed probability/head-output error<=1e-4,
finite head bridge<=.005. Original raw-logit1e-4 diagnostics are retained even
if they fail. No whole-block derivative or old derivative recertification is used.

Forecast F/S/C/B/ZS/ZC/ZB requires224 full forwards; native T/O requires64.
Each intervened attention adds two full QK, three full softmax operations
(False full distribution plus two source-excluded distributions), and one PV.
Witness F/T kernels add the two QK and three softmax when absent. All witness
kernels add three PV for the other corners; their probability maps already exist.
Expected forecast extras:10,800QK;16,200softmax;5,376PV;288witnessPV.
Native extras:1,840QK;2,760softmax;896PV;144witnessPV. CPU audits, output-map
reads/projections, both complete attention calculations and all loading count
as work. Nothing here is a compact counterfactual predictor.

Use the shared-cache launcher with8GiB reserve. Cap total forecast archives at
512MiB, native at256MiB, fixed output maps at64MiB; check space before each case.
Never prune results to pass a guard or repeat intentionally pruned runs.
Progress receipts have count-specific immutable names. Source/input/runtime and
all forecast-array hashes must match the public seal and remote receipt before
the first ordinary True call. Withhold scientific outcomes if required checks
fail; preserve failed attempts; do not tune after observing native outcomes.

A result can establish a conditional source-allocation/context-selection account
under these native interventions. Executing the current query does not explain
its learned generator. Further work requires a specific causal hypothesis from
this test, not another end-to-end correction sweep. Private article stays local.
