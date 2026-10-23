# 275: does adaptive source reading generate the contextual residual step?

Freeze this protocol and all new scientific code before any275 study tensors.
Use all32 OLD273 cases. No fresh prediction or learned-circuit claim is possible
from this diagnostic. Do not alter274's failed primary or older numerical checks.

## Hypothesis and alternatives

274 rejects the sufficient account that a portable residual step becomes
nonportable mainly through the source emission's False normalization background.
The information is already in the upstream step. A specific candidate is
attention reweighting of the unchanged causal prefix, followed by finite MLP
processing. A distinct candidate is propagation of the source's changed self
value through relatively fixed reads, with finite MLP background dependence.

Attention probability adaptation includes both query changes and the changed
source key. It is NOT an isolated query intervention or feature identification.
Every source MLP remains finite; this test does not isolate its internal cause.

## Native source-query factorial

Observe one ordinary False computation. At every layer save its full K/V,
source probability row, source states and primitive outputs. All candidate
trajectories start from the known True input and otherwise use pinned weights:

- N: native source probabilities and source self value.
- W: source QUERY row uses its observed False attention probabilities; current
  source self value remains. This removes probability adaptation at the source.
- V: source QUERY row retains its current probabilities but reads its own source
  value from the False observation. It may reread unchanged prefix values.
- B: both source probability and self-value changes removed from that query row.
- FB: False input with B clamps, requiring exact zero-edit replay.

Only the source query row's attention result is replaced. Receiver attention
reads the actual current source K/V and remains finite; receivers are not given
False or native True hidden states. The prefix before the source is causally
unchanged. Candidate source states depend only on that prefix and the carried
source, despite executing the expensive full sequence. This uses a complete
False observation plus finite counterfactual work, not one terminal vector or
a useful small-budget predictor. N is an old native reference, not a blind target.

Additionally save three fixed-native-input probability/value corners at every
source attention, alongside N. They give exact finite signed source-write
components: reweight=Vcorner-Bcorner, self-value=Wcorner-Bcorner, and
mixed=Ncorner-Vcorner-Wcorner+Bcorner. Save actual vectors, not just norms.
They are retrospective algebra at one native input, not additive downstream
fractions or learned features. Full factorial interventions test their evolving
consequences rather than summing local projections into a mechanism.

## Source emission and downstream tests

Record every candidate's emitted source dV across all28 layers. Then use those
fields in a separate native True continuation, changing only source V read by
later queries. Native source computation/K remain in these VALUE-ONLY arms:
U(native), W,V,B, Z(no source V change), FZ(False zero edit), T(native reference).
These conditional arms separate changed value communication from other source
K/state pathways. They are not complete False-only forecasts.

Primary hypothesis: adaptive source probability with a False self-value read
is sufficient for the native source message/effect, while fixed probabilities
are not. Require V to pass the following combined test in>=24/32 cases and
>=6/8 in every schema/order stratum, and W to pass in<=8/32:

1. Aggregate emitted-dV error<=10%, cosine>=.995 and at least24/28 layers meeting
   the same bounds. Target norms>=.001. Every layer reported.
2. In the value-only continuation, reconstruct native U-Z source-value effect
   with whole receiving AND verdict error<=25%, cosine>=.95, target norms>=.001;
   actual finite margin error<=.005+.25*abs(mU-mZ), target margin>=.02nat.

If this fails, do not replace it with a median comparison or another arm's
success. W/B and every component endpoint remain visible. Separately report
the ORIGINAL complete native N-F target for both full factorial and value-only
arms:<=5% vector errors,>=.99cosines, material targets, actual margin error<=.10nat.
Source-message sufficiency does not imply complete-response sufficiency.

Save the exact native source ledger
`h_next = h + deltaAttentionWrite + deltaMLPWrite + Float32Rounding`.
Also sum paired attention/MLP increment contrasts across layers to reconstruct
the native cross-background h contrast, retaining rounding. Signed projections
can cancel and are descriptive; they do not identify independent mechanisms,
causal importance, semantic features or on-manifold hybrids.

The block28 source output has no later attention layer in this model. Its
source-write ledger is reported for completeness, but cannot itself affect the
final receiving states. Block28 emitted V depends on the block27 source output;
do not turn the complete layer profile into a causal importance ranking.

## Numerical qualification and preservation

Require N/F source K/V, tail trajectories and scores to replay273 exactly;
native corner B source write equals F exactly. Full prefix hashes, unchanged
prefix K/V and source-query-only installation checks must pass at all layers.
FB/F, value-U/T, value-FZ/F and archived value-Z replay exactly. Keep all failures.

The separate auditor imports no producer factorial helpers. Reconstruct source
norm/Q/K/V from saved block inputs and pinned public primitives in full sequence
shape, including RoPE at the actual positions. Those nodes, source carry/block
addition identities, finite MLP replays from recorded mid inputs and actual full
vocabulary heads must match exactly. Rebuild source QK/softmax/PV/output from
saved causal prefix operands; public attention primitive replays must meet the
unchanged1e-4 absolute bound. Restored execution layouts may differ, so these
particular composed attention checks are not declared bit exact.

Independently compute CPU64 RMS, V projection, source softmax, PV and output
projection; report every raw original1e-4 failure without erasure or relabeling.
CPU64 finite-head bridge remains.005. These are component validations within one
investigation, not independent whole-model implementation or derivative proof.
No disputed whole-block Jacobian is used or recertified. Failed204 replacement
claims remain invalid, original numerical failures remain failed.

## Work and bounds

416 full forwards:32 False,128 full factorial,32 FB zero edits,224 value-only
continuations. Extra calls:3,584 source-intervention PV,2,688 native-corner PV,
2,688 native-corner output projections,5,376 value-intervention PV, plus audit.
All other model projections, normalization, SiLU, products, attention and final
readouts remain finite. No speedup is claimed;272 is still the causal cost baseline.

Require>=2GiB initially and>=768MiB per case, cap new raw NPZ arrays at1000MiB.
Never delete old arrays or overwrite a run. Save sources/checkpoint/runtime hashes,
every input/case/vector/score/control and actual counts. Old dependent templates,
whole prompt/position differences and one small checkpoint limit generality.
If a stable primitive account emerges, a frozen False-only fresh-context test
with an explicit useful budget remains separate future work.
