# 261 — does receiving MLP normalization generate consequential gate curvature?

Retrospective mechanism test on the four fixed first-per-schema/order258 witness
contexts, all28 blocks. These are old, dependent indigo contexts, not fresh
confirmation. Source and prefix stay finite; receiver queries use the unchanged
258 cubic rule; every receiving MLP uses the unchanged ranked1% SiLU rule.
Do not edit any258 producer, source seal or original failed numerical check.

## Questions and direct interventions

At an actual MLP input x0+d, split the smooth normalized input change into
J0*d and f(x0+d)-f(x0)-J0*d. Propagate BOTH through the learned gate/up matrices.
Measure complete signed displacement vectors, not just a verdict projection.
Does the gate displacement at native T and accurate258-S inputs have <=5%
first-order error at >=90% of eligible case/layer inputs, separately for whole
receiving coordinates and the input-conditional top1% mask? Norm guard1e-8;
retain zero/ineligible records. Report verdict and up-path outcomes as secondary.
Native T masks are retrospective diagnostics, never forecast inputs.

Then run all of the following full-network primitive policies from the input:

- F: ordinary False.
- Z: zero edit with L policy; must replay every False tail and score exactly.
- B:258-S reference, finite MLP-input RMSNorm; must replay saved258-S exactly.
- L PRIMARY: receiving MLP-input RMSNorm replaced by its False-anchored tangent.
- C: same with homogeneous quadratic+cubic RMSNorm terms added at its own input.
- R: tangent plus per-row norm-matched random direction of C's added correction,
  evaluated at R's own input. SHA256 seed key:layer:row:mlp-input-curvature-control.
- T: ordinary True; must replay saved native258-T exactly.

No native receiving state is inserted in L/C/R. All generate their own states;
gate/up/down projections, products, attention K/V, softmax and final head remain
finite. R matches normalization-correction norm, not projected gate effects or
manifold validity. Save all branches; never substitute a successful secondary
for L. These tests examine a normalization primitive, not a learned feature.

For each arm report final whole AND verdict relative error<=.05, cosine>=.99,
native norms>=.001 and margin error<=.10 nats, all four cases. L sufficiency
requires4/4. C/R are separately reported, no pooled new success claim. A bounded
finite-reference interpretation requires B4/4 and exact old controls. An arm
with nonfinite state or >100-times-False full-tail norm stops, is retained as
guard-failed and counts as failure; continue other arms only for that explicit
guard. Any other implementation/numerical failure stops the study with evidence.

## Numerical qualification before attribution

Freeze producer, auditor, this protocol and imported scientific source hashes
before any new model forward. Bind original258 audit/design/status/array hashes,
pinned checkpoint and runtime. Save False/current MLP inputs, original/installed
normalizations, actual gates/up/SiLU/products/writes, masks, query inputs, all
layer tails/scores and source/prefix checks. Different arms have different inputs.

The separate auditor reconstructs both query and MLP normalization maps from
saved actual inputs. Cubic coefficients must match independent polynomial jets
within1e-10. Qualify the smooth CPU64 RMSNorm derivative by Torch AD1e-9 and
centered differences on max-normalized actual directions: step1e-3 bound1e-3,
step1e-4 bound1e-5. These are not Float32 or whole-block derivative certificates.
Public original RMSNorm/normalized-input bridges1e-4; installed maps/products
exact; gate/up public-weight primals and displacement closures1e-4; finite
two-score head .005. Preserve any failure, no silent relaxed tolerances.

For B/T save complete gate/up first-order and curvature vectors. Use CPU64 smooth
normalization for their algebra; record numerical differences from public
Float32 gates separately at1e-4. Verify selected masks, whole/verdict/selected
geometry and per-row coordinate budgets. Gate/up curvature is a defined local
allocation conditional on the incoming state, not a unique causal feature.

Record exact forwards and component/AD/FD work. Public normalizations and SiLU
still execute before hooks; neither the sparse fraction nor these approximations
establish a runtime saving. No full independent model reconstruction, resolution
of older derivative disputes, fresh generalization, circuit claim or complete
answer to Question B follows from this retrospective test.
