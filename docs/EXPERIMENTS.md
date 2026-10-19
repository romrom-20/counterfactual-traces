# Experiment index

The active repository focuses on Question B. The causal studies 225b–227b use the same 32 old, dependent synthetic contexts. Study 228 adds 32 new literal pairs with forecasts sealed before native True measurement; 229 diagnoses those now-observed contexts retrospectively. All use one pinned Qwen checkpoint; no successful general False-only native-response predictor is established.

## 225b — What reverses the local effect's sign?

Separate the gate–up interaction K from the baseline shift B at MLP23. K reverses the tested sign in 32/32 cases; B alone does so in 0/32. The strict K-only quantitative account passes 16/32, all generic, and fails overall.

[Protocol](experiments/225b-baseline-interaction-separation.md) · [Report](analysis/225b-baseline-interaction-result.md) · [All cases and audit](../results/baseline-interaction-225b/README.md) · [Code](../research/boolean_coupling/run225b.py)

Original 225 failed its tokenization preflight before any forward. Its code and failure record remain in the same release.

## 226b — Where is the remaining magnitude interaction generated?

Separate finite local generation from propagation of an existing interaction at attention and MLP updates in blocks 24–28. MLP24 removal passes the joint descriptive margin/vector reduction rule in 16/16 cases. No other individual tested sublayer passes that rule. Final normalization alone leaves more than .10 nats unexplained in all 16 descriptive cases.

[Protocol](experiments/226b-finite-tail-interaction.md) · [Report](analysis/226b-finite-tail-interaction-result.md) · [All cases and audit](../results/finite-tail-interaction-226b/README.md) · [Code](../research/boolean_coupling/tail226b.py)

Original 226 stopped before interventions because of a saved-state boundary indexing error. The original [protocol](experiments/226-finite-tail-interaction.md), [source](../research/boolean_coupling/tail226.py) and [failure](../results/finite-tail-interaction-226b/preserved-failure/failure.json) remain unchanged. The corrected experiment retains the same bounds and criteria.

## 227b — Which MLP24 operations generate that interaction?

Use a declared finite decomposition of normalization, SiLU mixing and the gate/up cross-product. Product-term removal halves the margin gap in 16/16 descriptive cases but passes the joint vector/margin criterion in 0/16. Only removing the combined mixed term passes both, 16/16. Contributions are conditional on the specified order; they are not learned features or unique causal fractions.

[Protocol](experiments/227b-mlp24-mixed-primitives.md) · [Report](analysis/227b-mlp24-mixed-primitives-result.md) · [All cases and audit](../results/mlp24-mixed-primitives-227b/README.md) · [Code](../research/boolean_coupling/tail227b.py)

Original 227 stopped at a failed final-coordinate comparison to a differently placed intervention. The [source](../research/boolean_coupling/tail227.py), [protocol](experiments/227-mlp24-mixed-primitives.md) and [failure](../results/mlp24-mixed-primitives-227b/preserved-failure/failure.json) remain. In 227b, 31/32 raw coordinate bridges still fail; the .005 two-score bridge passes. Two raw ideal-write comparisons also fail. Arithmetic-accounted closure does not turn those failures into passes.

## 228 — Can selected late nonlinear writes forecast native alignment?

All 96 forecasts were sealed, qualified and pushed before the 32 new native True observations. Finite MLP22/23 updates improve every case against linear and norm-matched orientation controls, but meet both the primary alignment and stronger native-response criteria in **0/32** cases. The first-block receiving displacement is exact; the first forecast discrepancy appears at attention block2. The responsible primitive remains unresolved.

[Protocol](experiments/228-false-only-alignment.md) · [Prior interpretations](analysis/228-prior-predictions.md) · [Report](analysis/228-false-only-alignment-result.md) · [Cases, seals and reproduction](../results/false-only-alignment-228/README.md)

The original final scorer stopped on a metadata key collision. [228b](experiments/228b-reporting-correction.md) corrects only the layer field/docstring with no model calls or numerical changes. The original source and traceback are preserved. The new smooth CPU64 transport maps do not recertify the original Float32 Jacobian.

## 229 — What generates the first attention discrepancy?

At the exact incoming native state before attention 2, receiver-only curvature meets the local verdict-position account in **32/32** contexts; the source–receiver cross term does so in **0/32**. Across all receiving rows the interaction is substantial. Direct QK, softmax and weight–value edits qualify its operation-defined signed accounting. The main verdict remainder still lacks a primitive explanation. Its isolated finite-suffix margin consequence is below the frozen materiality guard in **all 32** cases; the registered downstream rule is ineligible, not failed.

[Protocol](experiments/229-first-attention-generator.md) · [Report](analysis/229-first-attention-generator-result.md) · [All cases and audit](../results/first-attention-generator-229/README.md) · [Runner](../research/boolean_coupling/generator229b.py)

Two bootstrap metadata checks stopped before model loading. Original 229, the first 229b source and both failure logs are preserved; the [corrected bootstrap](experiments/229b-bootstrap-correction.md) changes no scientific comparison or threshold. The 96 smooth derivative checks do not recertify old Float32 Jacobians.

## 230 — What generates receiver-only curvature?

All 16 finite/False-affine attention graphs qualify. Every registered single-primitive account fails in all 32 old contexts. The first receiving row carries the normalization-only correction, but it does not suffice for complete R. A posthoc normalization-plus-probability/value-product combination achieves 5.45% median remainder error and motivates 231. The normalization/product mixed term alone worsens the normalization-only error.

[Protocol](experiments/230-receiver-curvature-primitives.md) · [Report](analysis/230-receiver-curvature-result.md) · [Cases, edges and audit](../results/receiver-curvature-230/README.md)

## 231 — Can that local rule forecast fresh native changes?

All 64 local forecasts are sealed, qualified and remotely verified before the 16 ordinary True runs. The attention-2 verdict-position displacement criterion passes 16/16, with 1.00% median relative error. The stronger whole-receiving criterion passes 0/16, at 5.88% median error. The budget includes an exact known-edit first block and source-only attention call. This is local prediction, not a final-response forecast or compact circuit.

[Protocol](experiments/231-local-normalization-product-forecast.md) · [Report](analysis/231-local-forecast-result.md) · [Cases, seals and reproduction](../results/local-normalization-product-231/README.md)

## Earlier research and preserved boundaries

The [historical evidence](../evidence/HISTORY.md) preserves the earlier Question B path, including the qualified 204 retrospective diagnostic, invalid replacement-agent conclusions, the original whole-block derivative disagreements, source-only forecast failures, and the failed 224 coordinate bridge. Its publication-filtered archive retains 2,867 original files with exact hashes; one unpublished article is withheld. Preservation is not endorsement of a superseded claim.

Start with [research notes](RESEARCH-NOTES.md) for the reasoning across studies, or [reproduction](../research/README.md) to verify and execute the current sequence.
