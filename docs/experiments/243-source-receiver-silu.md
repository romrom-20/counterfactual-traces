# 243: where does the finite-SiLU policy contrast originate?

Freeze this protocol and both numerical source files before intervention work.
Use the 32 now-observed 242b contexts. This is a retrospective causal diagnostic,
not a fresh forecast or a learned-circuit identification.

## Fixed computation and interventions

Reuse the archived False primitive anchors, original inputs and pinned public
model. Keep RMSNorm, projections, bilinear products and final readout finite.
Keep 242b's positive normalized quadratic softmax tilt at every edited/later
position. Generate every source and receiving state internally; supply no native
trajectory, reset, displacement or correction vector.

Cross public finite (F) and False-quadratic (Q) SiLU by position, at all 28 blocks:

- QQ: Q at the changed Boolean and all strictly later receiving positions.
- FQ: F at the changed Boolean, Q at all receiving positions.
- QF: Q at the changed Boolean, F at all receiving positions.
- FF: F at both regions.

The unchanged causal prefix is never intervened on. The public SiLU is evaluated
before replacing the selected rows. Its exact dense-work cost remains counted;
this is not a speedup. Actual replacement rows and untouched finite rows are
checked directly. Derivative qualification is restricted to the Q rows actually
installed; softmax retains all source/receiving query rows.

QQ must replay 242b100, and FF must replay 242b101, at every block-tail coordinate
and both final scores with zero tolerance. A separate zero-edit QQ control must
replay the saved False trajectory/scores exactly. Source trajectories must be
exactly equal for QQ/QF and FQ/FF: later receiving changes cannot affect a causal
source. All block prefixes must match saved False hashes exactly.
Source-only MLP changes cannot affect receivers in the same first block:
QQ/FQ and QF/FF must have identical first-block receiving states.

## Frozen question and signed vector effects

Let h be the final pre-readout residual, restricted to receiving rows. Define

`B = hFF-hQQ`, `S = hFQ-hQQ`, `R = hQF-hQQ`,
`I = hFF-hFQ-hQF+hQQ`, so `B=S+R+I`.

For whole receiving vectors and the final verdict row, retain each signed share
`dot(term,B)/dot(B,B)`, relative norm and cosine. Repeat the finite scalar
factorial on final margins (true score minus false score). This is an intervention
contrast between global policies, not a unique allocation to learned features.
Individual signed shares can exceed one or be negative.

**Primary:** source-only S recovers the FF-versus-QQ contrast: target norms >=.001,
relative error <=.10 and cosine >=.99 at both whole receiving and verdict scopes,
and `abs(mFQ-mFF)<=.10 nats`. Require 24/32 and 6/8 per schema/order stratum.
Receiver-only R is a declared secondary with the same criteria. Separately test
negligible interaction: `norm(I)/norm(B)<=.10` at both scopes and absolute margin
interaction <=.10 nats, with the same guards/count rule. Do not swap a successful
secondary into the primary. Small margin contrasts do not establish behavioral
materiality; retain their magnitudes explicitly.

Also report each arm's native displacement/margin accuracy under unchanged
242b bounds (5%, .99, .001 and .10 nats). Native references are already observed;
read their arrays only in the separate audit after all intervention arrays are
sealed. These diagnostics cannot establish prospective performance. No end-to-end
correction search follows from a failed primary alone.

## Numerical qualification and evidence

Reuse unchanged primitive242b/reference242b CPU64 derivative and installation
qualification. First and second normalized directional actions retain 1e-9 AD,
1e-4 FD and actual installed-map bounds. Softmax derivatives are of the public-
anchored exponential tilt, not the original rounded Float32 graph. Current
public finite SiLU values get separate CPU64 primal checks at 1e-4. Record all
checks, endpoint witnesses, direction scales/hashes and call counts. Final-head
reconstruction stays .005. Older derivative/bridge failures remain unresolved.

Keep 242b's scientific instability caps: nonfinite values, block-tail norm above
100 times False, approximated input displacement above 100*max(1,False input
infinity norm; unmasked logits), or approximated output above
min(1024,100*max(1,False output infinity norm)). A declared unstable new arm is a
prediction failure; retain its partial trajectory/first cause and do not exclude
the case. Unexpected qualification, replay or causality failures stop the study
and remain failures. No bounds are relaxed.

Bind source files, checkpoint/library versions, all reused forecast/reference
array hashes, the prior audit and input fixture hashes. Save complete block-tail
trajectories, scores, metadata and a global intervention seal. Do not duplicate
old anchors or the checkpoint. Separate audit reconstructs replays, regional
causality, signed contrasts, final readouts and every outcome. Publish all cases,
including failures, to the private repository. Keep the article local.
