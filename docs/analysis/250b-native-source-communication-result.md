# Continued source-value communication sustains the native response

**Qualified retrospective result; Question B remains open.** In all 32 old
contexts, clamping source keys and values to their native False versions during
receiving attention reads in blocks 10–28 removes at least 90% of both the
signed native verdict-vector projection and signed margin response. This meets
the frozen 24/32, 6/8-per-stratum rule in 32/32. It is a necessity result under
these interventions, not a prospective accuracy result.

The original250 stopped after all model work on a witness-count assertion.
Its selector saved **all32** contexts instead of the intended four, producing
1,056 witness QK products rather than132. The separately frozen250b correction
retains every array and check, reconstructs complete accounting, and audits
the expanded witness set. No model outputs, tolerances, primary criteria or
cases changed. [Original protocol](../experiments/250-native-source-communication.md)
and [exact correction](../experiments/250b-witness-completion-correction.md).

## What the intervention changes

All arms start with the native True full-sequence computation before block10.
For attention queries after the changed Boolean, a source key and/or value can
be replaced with its ordinary False counterpart. Earlier tokens and the source
keep their own native True states; receiver states evolve under the intervention.
The final normalization and full vocabulary head remain finite. No derivatives
or disputed whole-block Jacobians are used.

The B arm replaces both K/V for every receiving query. The H control instead
replaces only the residual source row with native False at the boundary and
then runs an ordinary finite suffix. **B and H match exactly at every receiving
coordinate in every remaining block and at the final scores**, while B's source
stays True and H's stays False. Thus continued source communication, rather
than just the previously changed receiving state, sustains most of this target.

## Values, routing, direct reads and relays

All figures below are separately pooled medians across32 dependent contexts:

- Replace source **keys only**:96.34% of the signed verdict projection and98.15%
  of the native margin response remain.
- Replace source **values only**:−4.43% of the signed verdict projection and5.80%
  of the native margin response remain. The remaining verdict displacement
  still has29.45% of the native vector's norm. A small projection is not a
  small full vector.
- Replace **both**:−2.02% signed projection,−1.95% margin response, and7.48%
  remaining vector norm. These are different metrics; the primary tests signed
  projection and margin, not90% removal of every vector coordinate or norm.
- Replace both **only for verdict queries**:16.77% signed projection and28.70%
  margin response remain.
- Replace both **only for intermediate queries**:69.21% signed projection and
  56.50% margin response remain.

Direct source reads have the larger effect; intermediate reads also matter.
The effects overlap through the evolving computation. In every context, the
interaction allocation for the removed vector effect is negative (median−6.10%
of native verdict projection). It would be wrong to add the two pooled effects
and call them independent paths or assign neurons semantic roles.

The K/V factorial similarly has a negative interaction allocation in every
context (median−6.90%). K includes the source key's normalized representation
and its softmax consequences. It does **not** isolate softmax alone. The result
does not say attention weights are irrelevant: receiver queries, other keys,
and later attention weights remain free to change. Direct/relay arms clamp
both K/V, so this test does not uniquely isolate a direct-value circuit.

Three random K/V perturbations match each actual source displacement's K and V
norms at every suffix layer. Their median remaining verdict projections are
95.47%,94.82%,100.02%, and median margin responses98.25%,99.03%,100.41%.
Thus generic perturbation magnitude does not reproduce the source-False clamp's
effect. These controls are not guaranteed to be on-manifold.

![All contexts and signed effects](../figures/250b/250-source-communication.png)

The teal points are generic-schema contexts; orange points are descriptive.
Black bars are pooled medians. Each point is a case, not an independent model
sample. The two panels measure different quantities; the vector panel is a
signed projection rather than the whole residual error.

There is a useful descriptive split within these old contexts. Across the two
orders, verdict-only clamps leave6.8–7.1% of the signed vector projection in
generic schemas, versus30.4–31.0% in descriptive schemas. Intermediate-only
clamps leave80.7–81.1% versus62.1–62.6%, respectively. This suggests different
allocations between direct reads and relays in these two prompt families;
it does not isolate field meaning from length, position or other prompt changes.
These are posthoc stratum summaries, not an independently confirmed effect.

## Qualification and limits

All ordinary False/True suffix coordinates and scores replay242b exactly.
The zero clamp replays native True exactly. The causal prefix/source invariants,
query-group installation checks and B/H identity pass. The separate auditor
passes3,776 finite checks, maximum absolute discrepancy1.252e−5. QK/softmax/PV
witness coverage is all32 contexts,11 arms,blocks10/19/28 (1,056 kernels).
Every endpoint gets a separate CPU64 finite-head check. This is not an
independent full-forward reconstruction of all19 intervened blocks. Coordinate
tolerance remains1e−4; head-score tolerance .005. The older derivative
disagreement is not resolved or used.

The experiment uses one pinned Qwen2.5-1.5B-Instruct checkpoint and32 existing,
dependent synthetic contexts. It does not test other models, tasks or fresh
examples. Clamped/hybrid states are artificial causal diagnostics.

Actual budget:352 full forwards;608 additional H suffix blocks and32 full
vocabulary heads;5,472 extra QK, softmax and PV recomputations **each**;
1,056 extra witness QK products. Numerical auditing is additional work. No
speedup or computationally compact predictor is demonstrated.

## What this changes about Question B

The native response cannot be understood simply as a changed verdict state
being carried forward.249 already showed that much of it can be regenerated
from the computed source. This test adds a primitive causal constraint:
continued transmission of the source's **value information** is much more
consequential than its changed keys in this late region. Direct reads dominate
these defined interventions, with non-negligible relays and interactions.

The source value difference at a layer has an explicit form:
`deltaV = Wv (RMSNorm(sourceTrue) - RMSNorm(sourceFalse))`.
The sourceTrue trajectory can be computed from the known input edit and
unchanged causal prefix, without observing native True receivers. But that
formula does not predict the unknown receiver queries, their attention weights,
the relayed states, or their nonlinear propagation. Those are the remaining
forecasting questions. A future false-state-only hypothesis must specify which
of those computations it approximates, freeze its predictions, and face native
measurements on fresh examples.250b alone does not answer them.

[All cases, checks and accounting](../../results/native-source-communication-250b/README.md)
· [Producer](../../research/boolean_coupling/communication250.py)
· [Bounded completion](../../research/boolean_coupling/communication250b.py)
· [Separate auditor](../../research/boolean_coupling/audit250b.py)
