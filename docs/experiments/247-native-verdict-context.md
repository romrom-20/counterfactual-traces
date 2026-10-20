# 247: native earlier-context versus verdict-state coupling at attention10

Retrospective on the32 observed242b contexts, one pinned Qwen model. The site is
selected POSTHOC from244's regional verdict profile: the first large jump occurs
at block10. This selection does not establish native relevance. Unlike243–246b,
the target here is the actual NATIVE False-to-True Boolean response, not a
quadratic-SiLU policy interaction. Freeze protocol, runner and separate scorer
before any new model work.

Replay ordinary native False and True, requiring every saved block-tail
coordinate and both scores to match242b exactly. At the input before block10
construct four finite attention parents: F(all False), P(earlier rows True,
verdict row False), R(earlier rows False, verdict row True), T(all True). Earlier
means all strictly preceding positions. Keep the unchanged prefix exactly
equal. These parent states are observed oracles, not prospective inputs.

Capture the verdict query, all keys/values, masked/unmasked logits, probabilities
and weighted-value output before output projection. The full native attention
write difference is B=YT-YF. Context influence at the True receiver is E=YT-YR;
at the False receiver it is E0=YP-YF. Coupling I=YT-YP-YR+YF=E-E0. Keep both
signed vectors; report I/B norm, cosine and projection after the output map.
The receiver-state effect at True context is YT-YP. No one scalar norm is a
substitute for a complete vector comparison.

Within E, the query must match T/R exactly. Earlier keys and values change:
W=(PT-PR)VR; V=PR(VT-VR); K=(PT-PR)(VT-VR).
E=W+V+K+rounding. W includes changes to normalization/key logits/softmax, not a
claim that softmax alone caused it. No learned features or unique allocation.

At native True attention10, directly subtract each pre-output weighted-value
term at the VERDICT ROW ONLY, leaving all earlier rows exact: E, YT-YP, I, B,
W,V,K, and3 deterministic orientation controls with I-matched whole-row norm.
Run the entire remaining computation and final readout finitely. Report each
final native displacement error and remaining native True-minus-False response,
whole receiving and verdict scopes, plus signed margin changes.

Primary: coupling-only sufficiency. Its projected I predicts native attention
write difference B at the verdict (target norm>=.001, <=10% relative error,
cosine>=.99); removing I reduces final native verdict-displacement norm >=25%
and absolute native margin response >=.10 nats (baseline response>=.10), and
beats all3 controls on verdict norm. Require24/32 and6/8 per schema/order stratum.
Whole-receiving results are mandatory secondary scope checks, not an altered
criterion for full Question B. Also report local and causal conditions separately.
No successful secondary replaces the primary. No end-to-end correction sweep.

Independent numerical qualification: CPU64 QK, softmax, probability/value and
output-projection bridges <=1e-4; explicit multiplication remainder <=1e-4;
finite head .005. Compare maximum absolute and signal-relative remainders.
Verify all crossed input rows, query invariance, probability installation,
unchanged pre-target states and all causal prefixes exactly. Keep masks/rotary
embeddings fixed. No block derivatives or approximation maps are used; old
derivative failures remain unresolved. Unexpected failure stops and is retained.

Bind all source/checkpoint/library hashes and prior input/reference array hashes.
Save all four primitive corners, signed terms, intervention states/scores,
checks and seal before separate scoring. Budget:64 ordinary native forwards,
64 extra crossed-parent attention calls,320 primitive-intervention forwards,
plus all additional diagnostic primitive products and independent audit work.
Every case and failure is reported. Known native inputs do not establish a
False-only prediction. Keep article drafts local.
