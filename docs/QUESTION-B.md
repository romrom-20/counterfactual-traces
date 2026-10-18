# Question B

> Why does a Boolean input edit endogenously generate a particular state-dependent residual displacement, and can we predict the resulting behavior from the False computation alone?

More generally, can identifying the nonlinear operations that generate and propagate a transformer's displacement let us forecast its counterfactual response from one observed computational state?

The input edit and model weights are known. The desired explanation must specify an explicit update and computation budget, then forecast the native residual and verdict on fresh contexts before their True trajectories are measured. Replaying the whole counterfactual model is a useful reference, but is not the compact account this project seeks.

## What has narrowed the question

The starting observation is a conditional preference dependency: changing an explicitly irrelevant Boolean in a supplied completion prefix changes the later verdict's true-minus-false score margin. The preferred Boolean can remain unchanged while the margin moves substantially. This alone does not establish verifier competence or a deployment failure.

A source-only hypothesis failed all 16 fresh cases on its frozen criteria. Receiving state carried from earlier blocks matters. Earlier MLP writes and their nonlinear remainders affect the native receiving displacement and verdict, but restoring one selected final-neuron projection did not restore the behavior. Improvements in that diagnostic did not always improve the verdict.

Experiment 225b separates a local product interaction K from a baseline shift B at MLP23. K controls the tested sign reversal; the strict quantitative K-only account fails overall. The finite follow-up 226b localizes a consequential part of the remaining baseline-dependent mixing to MLP24. In 227b, a gate/up cross-product explains much of its margin effect but fails the combined vector/margin account; only the combined mixed term passes. Some later mixed responses partly counteract the discrepancy. The prospective follow-up 228 tests finite MLP22/23 updates against native alignment on 32 new pairs. Every case improves against controls, but no case meets either accuracy criterion. An exact first-block receiving state followed by a discrepancy at attention2 supplies an earlier local boundary. See the [latest result and numerical boundaries](analysis/228-false-only-alignment-result.md) and [research notes](RESEARCH-NOTES.md).

## What remains unresolved

1. **Generation:** which upstream operations create the particular gate/up displacement and alignment that make K's verdict effect negative here?
2. **Propagation:** can we extend the conditional MLP24 account to a complete vector prediction, including SiLU mixing and later compensation? The current primitive allocation is order-dependent.
3. **Prediction:** can a frozen, explicit False-only update forecast both native residual displacement and verdict on fresh examples?
4. **Scope:** does the account survive new templates and checkpoints? The current 32 synthetic contexts are dependent and previously observed.

These questions can overlap. A successful local intervention does not answer the prospective prediction problem. A further correction sweep that only repairs outputs retrospectively would not, by itself, resolve the missing generation/propagation account.

## Boundaries that stay in force

- Experiment 204 remains a qualified retrospective diagnostic, subject to its recorded checks. Invalid replacement-agent conclusions remain invalid.
- The original whole-block derivative disagreement and failed 224 coordinate bridge remain documented. Later local checks do not recertify them.
- Primitive calculations need numerical qualification before attribution. Learned-circuit claims need additional primitive-level causal evidence.
- Fresh forecasts must be sealed before native True-state measurement. Old-context diagnosis is explicitly retrospective.
- Grounding controls failed; competent-verifier shutdown, generality and novelty priority have not been demonstrated.

Read the [latest result](analysis/228-false-only-alignment-result.md), [reproduction guide](../research/README.md), and [preserved evidence](../evidence/HISTORY.md).
