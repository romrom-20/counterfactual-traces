# Question B

> Why does a Boolean input edit endogenously generate a particular state-dependent residual displacement, and can we predict the resulting behavior from the False computation alone?

More generally, can identifying the nonlinear operations that generate and propagate a transformer's displacement let us forecast its counterfactual response from one observed computational state?

The input edit and model weights are known. The desired explanation must specify an explicit update and computation budget, then forecast the native residual and verdict on fresh contexts before their True trajectories are measured. Replaying the whole counterfactual model is a useful reference, but is not the compact account this project seeks.

## What has narrowed the question

The starting observation is a conditional preference dependency: changing an explicitly irrelevant Boolean in a supplied completion prefix changes the later verdict's true-minus-false score margin. The preferred Boolean can remain unchanged while the margin moves substantially. This alone does not establish verifier competence or a deployment failure.

A source-only hypothesis failed all 16 fresh cases on its frozen criteria. Receiving state carried from earlier blocks matters. Earlier MLP writes and their nonlinear remainders affect the native receiving displacement and verdict, but restoring one selected final-neuron projection did not restore the behavior. Improvements in that diagnostic did not always improve the verdict.

The latest controlled test isolates a local product interaction K from a baseline shift B at MLP23. K controls the observed conditional sign reversal under the defined interventions. B still modulates magnitude enough that the strict K-only quantitative account fails overall. MLP22 does not show the same reversal. See the [full result and qualification boundaries](analysis/225b-baseline-interaction-result.md).

## What remains unresolved

1. **Generation:** which upstream operations create the particular gate/up displacement and alignment that make K negative here?
2. **Propagation:** which later operations produce the schema-dependent magnitude modulation?
3. **Prediction:** can a frozen, explicit False-only update forecast both native residual displacement and verdict on fresh examples?
4. **Scope:** does the account survive new templates and checkpoints? The current 32 synthetic contexts are dependent and previously observed.

These questions can overlap. A successful local intervention does not answer the prospective prediction problem. A further correction sweep that only repairs outputs retrospectively would not, by itself, resolve the missing generation/propagation account.

## Boundaries that stay in force

- Experiment 204 remains a qualified retrospective diagnostic, subject to its recorded checks. Invalid replacement-agent conclusions remain invalid.
- The original whole-block derivative disagreement and failed 224 coordinate bridge remain documented. Later local checks do not recertify them.
- Primitive calculations need numerical qualification before attribution. Learned-circuit claims need additional primitive-level causal evidence.
- Fresh forecasts must be sealed before native True-state measurement. Old-context diagnosis is explicitly retrospective.
- Grounding controls failed; competent-verifier shutdown, generality and novelty priority have not been demonstrated.

Read the [first-person investigation](drafts/lesswrong-context-dependent-corrections.md), [standalone code](../research/boolean_coupling/README.md), and [preserved evidence](../evidence/HISTORY.md).
