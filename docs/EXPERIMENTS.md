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

## 232 — Does the source-only expansion context recover all receivers?

All 128 forecasts are qualified, sealed, pushed and remotely verified before 32 new ordinary True runs. The unchanged finite normalization/product rule, anchored in the constructed source-only context, passes the all-receiving criterion in **32/32**, 8/8 per stratum. Median error is **1.33%**, versus **5.90%** for the old False-context rule; verdict accuracy is preserved. Source-conditioned linear transport alone fails all 32. A post-observation hybrid points mainly to the changed tangent, but is not a registered forecast arm. Exact block 1 and one finite source-only attention call remain declared work; final-response prediction is open.

[Protocol](experiments/232-source-conditioned-local-forecast.md) · [Report](analysis/232-source-context-result.md) · [Cases, seals and reproduction](../results/source-conditioned-local-232/README.md)

## 233 — Which primitive change accounts for the changed tangent?

On the 32 already-observed 232 contexts, cross source keys, the softmax Jacobian background, source values and receiver-value probability weights. Updating K/V together accounts for the whole tangent change in 32/32, leaving 8.58% median transport error and 1.513% native local error; every single-slot transport account fails. All 576 hybrid/native derivative comparisons qualify. The operator graphs are not native learned-circuit interventions. An explicitly posthoc additive-removal comparison motivates 234.

[Protocol](experiments/233-source-transport-primitives.md) · [Report](analysis/233-source-transport-result.md) · [Cases, witnesses and reproduction](../results/source-transport-primitives-233/README.md)

## 234 — Does the explicit source K/V product forecast fresh displacements?

All 256 forecasts are qualified, sealed, pushed and remotely verified before 32 fresh native True runs. The joint rule passes the whole-receiving attention-2 criterion in 32/32, 8/8 per stratum, at 1.52% median error. Removing the mixed product raises error to 2.47%, leaving 16/32 accurate; row-matched random controls fail all 32. Verdict accuracy is preserved, but the mixed term does not benefit every verdict row. Exact block 1 and a finite source-only attention call remain supplied computations. Final behavior remains unpredicted.

[Protocol](experiments/234-source-key-value-forecast.md) · [Report](analysis/234-source-key-value-result.md) · [Cases, seals and reproduction](../results/source-key-value-234/README.md)

## 235 — Where does the complete failed forecast's error come from?

The 47 write errors and final cast exactly sum to the original F parent discrepancy. A fixed ranking selects MLP23/22/21/20; finite recomputation shows that 22/23's errors are almost entirely responses to inaccurate inputs. A four-write terminal oracle repair reduces median margin error 6.588→1.668 nats but passes the joint criterion only 2/32. It is an oracle accounting intervention, not a prospective repair.

[Protocol](experiments/235-global-forecast-error-budget.md) · [Report](analysis/235-global-error-result.md) · [Every ledger, split and outcome](../results/global-error-budget-235/README.md)

## 236 — Does repairing MLP20/21 inside the forecast close the failure?

A two-switch factorial recomputes later states and finite MLP writes. Joint repair improves margin error 6.588→4.849 nats but has 0/32 passes on full-response cases. With an exact native reset before MLP20 it still has 0/32 passes on, at 0.344 nats and 24.4% whole-displacement error; only attention 21–24 remains approximate there. The fully finite oracle control replays exactly. All 11 arms, 32 old contexts and 1664 actual derivative directions remain qualified and retained.

[Protocol](experiments/236-upstream-mlp-factorial.md) · [Report](analysis/236-upstream-factorial-result.md) · [All branches and reproduction](../results/upstream-factorial-236/README.md)

## Earlier research and preserved boundaries

The [historical evidence](../evidence/HISTORY.md) preserves the earlier Question B path, including the qualified 204 retrospective diagnostic, invalid replacement-agent conclusions, the original whole-block derivative disagreements, source-only forecast failures, and the failed 224 coordinate bridge. Its publication-filtered archive retains 2,867 original files with exact hashes; one unpublished article is withheld. Preservation is not endorsement of a superseded claim.

Start with [research notes](RESEARCH-NOTES.md) for the reasoning across studies, or [reproduction](../research/README.md) to verify and execute the current sequence.

## 237 /237b — Which late attention primitive is missing?

237 stops at its original smooth/public final-coordinate bridge before a completed case.237b changes to explicit primitive omissions anchored to public finite attention. Under the native reset, restoring QK helps but leaves10.8% whole error; QK-only and softmax-only both have0/32 joint passes. Full attention is an exact replay control. From the wrong incoming state even all finite late computation leaves72.3%. Fixed-input signed effects change with depth.

**Qualification is partial:** the original237b audit stops at one Float64 MLP bridge. No237b audit pass is claimed. [Frozen237](experiments/237-late-attention-primitives.md) · [Preserved failure](../results/late-attention-237-failed/README.md) · [Frozen237b](experiments/237b-late-attention-removals.md) · [Report](analysis/237-late-attention-result.md).

## 237c — Inspect every remaining numerical check without hiding the failure

Posthoc numerical diagnostic, no new scientific intervention:25,119/25,120 comparisons pass; the original Float64 MLP bridge remains failed. All1536Float32MLP reconstructions and1920same-backend public replays pass. Attention derivatives/graphs qualify separately. Outcome counts remain conditional diagnostic measurements, not a fully qualified predictor or circuit account. [Protocol](experiments/237c-complete-numerical-diagnostic.md) · [Every check/case and reproduction](../results/late-attention-237b/README.md).

## 238 — Which upstream local generators create the wrong reset input?

Separate local native-input remainders from inherited error through all37 affine-prefix operations. The complete adjoint projection ledger closes, and a frozen pre-outcome rule selects MLPs16–19. Their own-input finite repair reduces final errors substantially but passes0/32; even four oracle remainders fail0/32. All numerical checks and independent finite endpoints pass. Original238/238b metadata bootstraps stop before models and remain preserved;238c changes only schema validation.

[Protocol](experiments/238-upstream-remainder-transport.md) · [Report](analysis/238-upstream-generators-result.md) · [Evidence/reproduction](../results/upstream-remainder-238/README.md) · [Information-budget clarification](analysis/238-information-budget.md).

## 239 — Can False curvature explain native-versus-own generator differences?

The full quadratic control fails all sites. The primary finite-normalization/quadratic-SiLU rule passes32/32 at MLP17 and18,0/32 at16,21/32 at19; the four-site account fails. A posthoc exact SiLU-tail identity localizes MLP16's remaining error to the first receiver and a target-dependent concentrated neuron projection. No sparse causal or global predictor claim follows. All Hessian/FD, primitive-allocation and primal checks pass.

[Protocol](experiments/239-curvature-generator-contrast.md) · [Report](analysis/239-generator-curvature-result.md) · [All evidence/reproduction](../results/generator-curvature-239/README.md).

## 240 — Can a forecast-derived activation mask recover the omitted SiLU term?

A fixed bound-based 1% mask is sealed before native receiving inputs are opened. Direct SiLU restoration passes the whole missing-contrast recovery criterion in 32/32 old contexts, but verdict-only recovery passes 16/32. Conditional final discrepancies shrink; the behavioral guard admits only 3/32 native-input cases and 0/32 forecast-input cases. Full finite continuation from the forecast input still leaves 25.2% native displacement error and 1.09 nats margin error. All numerical checks pass; this is not a native-state generator or a learned circuit.

[Protocol](experiments/240-target-blind-silu-tail.md) · [Report](analysis/240-target-blind-silu-result.md) · [Cases, masks and audit](../results/target-blind-silu-240/README.md) · [Runner](../research/boolean_coupling/run240.py)

## 241 — Is the missing input a few amplified error vectors?

Transport all29 native-input remainders before MLP16, retaining complete vectors and an exact squared-error budget. The predeclared greedy four-site account passes0/32; the first sufficient prefix uses16 sites and passes32/32. Posthoc metric checks show transport reduces individual-vector energy while weakening cancellation; small net transport does not mean unimportant transport. These are old-context oracle-vector diagnostics, not primitive causal attribution or a fresh predictor.

[Protocol](experiments/241-before-mlp16-vectors.md) · [Report](analysis/241-before-mlp16-vectors-result.md) · [All vectors, checks and reconstruction prefixes](../results/before-mlp16-vectors-241/README.md) · [Runner](../research/boolean_coupling/vector241.py)
