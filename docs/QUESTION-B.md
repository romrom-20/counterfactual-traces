# Question B

> Why does a Boolean input edit endogenously generate a particular state-dependent residual displacement, and can we predict the resulting behavior from the False computation alone?

More generally, can identifying the nonlinear operations that generate and propagate a transformer's displacement let us forecast its counterfactual response from one observed computational state?

The input edit and model weights are known. The desired explanation must specify an explicit update and computation budget, then forecast the native residual and verdict on fresh contexts before their True trajectories are measured. Replaying the whole counterfactual model is a useful reference, but is not the compact account this project seeks.

## What has narrowed the question

The starting observation is a conditional preference dependency: changing an explicitly irrelevant Boolean in a supplied completion prefix changes the later verdict's true-minus-false score margin. The preferred Boolean can remain unchanged while the margin moves substantially. This alone does not establish verifier competence or a deployment failure.

A source-only hypothesis failed all 16 fresh cases on its frozen criteria. Receiving state carried from earlier blocks matters. Earlier MLP writes and their nonlinear remainders affect the native receiving displacement and verdict, but restoring one selected final-neuron projection did not restore the behavior. Improvements in that diagnostic did not always improve the verdict.

Experiment 225b separates a local product interaction K from a baseline shift B at MLP23. K controls the tested sign reversal; the strict quantitative K-only account fails overall. The finite follow-up 226b localizes a consequential part of the remaining baseline-dependent mixing to MLP24. In 227b, a gate/up cross-product explains much of its margin effect but fails the combined vector/margin account; only the combined mixed term passes. Some later mixed responses partly counteract the discrepancy. The prospective follow-up 228 tests finite MLP22/23 updates against native alignment on 32 new pairs. Every case improves against controls, but no case meets either accuracy criterion. An exact first-block receiving state followed by a discrepancy at attention2 supplies an earlier local boundary. See the [failed whole-response result and numerical boundaries](analysis/228-false-only-alignment-result.md) and [research notes](RESEARCH-NOTES.md).

## What remains unresolved

1. **Generation:** which upstream operations create the particular gate/up displacement and alignment that make K's verdict effect negative here?
2. **Propagation:** can we extend the conditional MLP24 account to a complete vector prediction, including SiLU mixing and later compensation? The current primitive allocation is order-dependent.
3. **Prediction:** can a frozen, explicit False-only update forecast both native residual displacement and verdict on fresh examples?
4. **Scope:** does the account survive new templates and checkpoints? The local forecasts add fresh literals but still use dependent synthetic templates in one checkpoint.

These questions can overlap. A successful local intervention does not answer the prospective prediction problem. A further correction sweep that only repairs outputs retrospectively would not, by itself, resolve the missing generation/propagation account.

## Boundaries that stay in force

- Experiment 204 remains a qualified retrospective diagnostic, subject to its recorded checks. Invalid replacement-agent conclusions remain invalid.
- The original whole-block derivative disagreement and failed 224 coordinate bridge remain documented. Later local checks do not recertify them.
- Primitive calculations need numerical qualification before attribution. Learned-circuit claims need additional primitive-level causal evidence.
- Fresh forecasts must be sealed before native True-state measurement. Old-context diagnosis is explicitly retrospective.
- Grounding controls failed; competent-verifier shutdown, generality and novelty priority have not been demonstrated.

Study 229 now separates the first attention discrepancy at an exact native input. Receiver-only curvature accounts for the verdict-position error in 32/32 old contexts, while source–receiver interaction is substantial elsewhere in the sequence. The primitive source of the receiver remainder remains open. Its isolated finite-suffix margin effect is below the frozen materiality guard in every case. This is not a repair or causal explanation of the full 228 forecast failure.

230 rejects every single-primitive account of the receiver remainder. A normalization/product composite selected after that factorial becomes a prospective local rule in 231: it passes the attention-2 verdict-position criterion in 16/16 fresh literal pairs, while the whole-receiving criterion fails in all 16. It computes block 1 exactly as declared work and does not predict the final residual or margin. Generation of the first-block state and propagation through the remaining network remain unresolved.

232 then moves the expansion point to the constructed source-only context, while retaining the same finite normalization/product operations. It passes the complete attention-2 receiving-displacement criterion in 32/32 fresh pairs; median error falls from 5.90% to 1.33%, with verdict accuracy preserved. This is a successful local forecast under a budget that still computes the edited first block exactly. It does not forecast later propagation or final behavior.

A post-observation hybrid suggests that reanchoring the linear transport is sufficient for the observed local accuracy when the old nonlinear correction is retained; reanchoring only that correction is not. This is a retrospective distinction, not a validated new prediction rule or an identified learned circuit. The upstream generator, later errors, and their final behavioral effect remain the larger open questions.

233 separates the changed tangent into explicit attention derivative operands. Updating source K/V in the False probability background supplies a useful restricted account, while all single-operand accounts fail. The selected three-term key/value/mixed rule becomes a frozen fresh test in 234: 32/32 whole local passes at 1.52% median error. Removing its mixed product leaves 16/32 accurate. This explains more of the restricted local computation; block 1 generation, later propagation and final behavior remain open.

235/236 test whether the accumulated local results explain the failed full forecast. The complete write ledger closes exactly, but MLP22/23's largest errors are inherited from inaccurate inputs even though those MLPs are finite. Recomputing upstream MLP20/21 improves the real forecast but has 0/32 full-response accuracy passes. An exact native reset before MLP20 still has 0/32 with only attention 21–24 approximate. Thus both earlier carry and later approximation errors remain relevant; local influence and local prediction do not establish compositional closure.

Read the [latest primitive diagnostic and numerical limits](analysis/237-late-attention-result.md), [actual-continuation test](analysis/236-upstream-factorial-result.md), [complete global error diagnostic](analysis/235-global-error-result.md), [earlier local prediction](analysis/234-source-key-value-result.md), [failed original forecast](analysis/228-false-only-alignment-result.md), [reproduction guide](../research/README.md), and [preserved evidence](../evidence/HISTORY.md).

237b/237c further narrows the late response. Fixed-input effects shift from QK interaction nearattention21 to softmax near23; neither uniform selective policy predicts the final vector. Even fully finite late attention from the ordinary wrong incoming state fails. The attention calculations qualify separately, while one original Float64MLP bridge remains failed. This constrains the omissions but does not establish a compact explanation of the native generator, a learned circuit, or fresh end-to-end prediction.

238 traces all37 upstream local remainders and their transport to the wrong input beforeMLP20. Four selected MLP repairs help but do not close the final response, including with native remainder vectors supplied.239 quantitatively explains some local native/forecast generator contrasts using False coefficients; it fails a complete four-site account. The missing term has an exact SiLU-tail identity, with posthoc position/neuron concentration that is not yet a target-blind causal explanation. Predicting native receiving directions and the complete response remains unresolved. [Latest local account](analysis/239-generator-curvature-result.md) · [Upstream causal test](analysis/238-upstream-generators-result.md).

240 turns239's target-dependent SiLU-tail concentration into a target-blind conditional primitive test. A mask derived from False/forecast gates recovers the whole missing contrast in32/32 old cases, while verdict-only recovery passes16/32. Direct restoration greatly reduces the defined local approximation discrepancy, but the registered behavioral rule fails: only3 native-input cases and no forecast-input cases meet its .10-nat guard. Most importantly, full finite computation after the forecast's MLP16 input still leaves25.2% native displacement error and1.09nats. The unpredicted incoming state remains a bottleneck; explaining the local Taylor omission does not explain its generation. [Result](analysis/240-target-blind-silu-result.md).

241 explicitly transports all29 earlier local error vectors to the MLP16 input. A predeclared greedy four-site reconstruction passes0/32; the first sufficient prefix contains16 sites, exactly the contiguous setMLP8–attention16, a posthoc depth pattern. This rejects the tested few-additive-vectors account, not a compact state-dependent rule. Negative interactions partly cancel local error growth, while transport damps individual-vector energy and also weakens that cancellation. These complete vector/metric identities explain properties of the failed linearization; the primitives creating the native displacement remain unidentified. No new native measurement or correction sweep is run. [Report and limits](analysis/241-before-mlp16-vectors-result.md).

242b tests a uniform primitive-family rule on 32 fresh literal pairs, with all forecasts pushed before native True measurements. Every arm fails the full-response criterion. Finite normalization keeps forecasts bounded; finite SiLU lowers median whole error from 50.3% to 17.4%, but substantial source-token and receiving errors remain. The test generates its own source trajectory, a stronger approximation restriction than earlier exact-source component budgets. The next unresolved distinction is which part of the SiLU benefit originates at the edited source versus receiving rows. This does not yet identify a native mechanism. [Report](analysis/242b-uniform-primitive-families-result.md).

243 addresses the source/receiver ambiguity directly. Neither region alone explains the full finite-SiLU policy contrast; their interaction has a median norm of 35.8% of that contrast and is non-negligible in every case. Every arm still fails native-response accuracy. What remains unresolved is where that interaction is newly generated versus how earlier differences are transmitted. Exact regional causality/replay controls and the corrected saved-array audit pass; original reporting and older derivative failures remain retained. [Regional result](analysis/243-source-receiver-silu-result.md).

## Current boundary after244–246b

The243 interaction is conditional on approximate attention: its quantitative
transfer to native attention fails0/32. Native attention still shows a large
regional interaction. Its first attention2 seed is below the frozen materiality
guard and complete removal barely changes the final result. A large later
block9 jump has a finite matched-input SiLU-discrepancy account for the whole
receiving product32/32, but that account fails at the verdict row0/32 and does
not explain the final interaction under direct removal. This is a local
explanation of an approximation-policy contrast, not native Boolean-response
prediction. The next target should distinguish new generation from inherited
state specifically at the verdict position. [Full latest result](analysis/246b-block9-origin-result.md).
