# Both source and receiving SiLU matter; neither alone explains their combined effect

**Question B remains open.** The source-only account has **0/32 passes**.
The receiver-only account also has 0/32 passes. Their interaction is not
negligible under the frozen criterion. This resolves an ambiguity left by 242b:
finite SiLU's improvement cannot be assigned to either region alone.

![Regional interventions, native errors and signed vector effects](../figures/243-regional-silu.png)

The third panel shows a signed projection, not the interaction's total norm.
A small projection here must not be read as a negligible interaction.

## The intervention

On the 32 existing 242b contexts, cross finite versus False-quadratic SiLU at
the changed Boolean and at every strictly later receiving row. Apply that
position policy at all 28 blocks. RMSNorm, projections, bilinear products and
final readout stay finite; the positive quadratic attention tilt is unchanged.
Each arm generates its own states. No native state or correction is inserted.

- **QQ:** quadratic SiLU in both regions, replaying 242b100 exactly.
- **FQ:** finite SiLU at the source, quadratic at receivers.
- **QF:** quadratic at the source, finite at receivers.
- **FF:** finite SiLU in both regions, replaying 242b101 exactly.

This is a retrospective causal diagnostic, not fresh confirmation. All models
evaluate the public SiLU before replacing selected rows; this costs dense
computation and is not a speedup.

## Native-response errors

Every arm still has **0/32** passes under the unchanged joint native residual
and margin bounds. Median whole receiving errors are **50.30% (QQ)**,
**30.75% (FQ)**, **41.79% (QF)** and **17.38% (FF)**. Verdict-row errors are
40.43%, 26.51%, 32.03% and 14.87% respectively.

Every nonbaseline arm lowers whole residual error in all 32 cases. That does
not imply an improved verdict in every case. Median margin errors are
**1.444, 1.086, 1.412 and 1.361 nats** in the same order. Only 14/32 source-only,
9/32 receiver-only and 15/32 both-finite cases improve the absolute margin error
over QQ. Source-only beats FF on that scalar in 16/32 cases despite its worse
whole residual error. Those are measurements of different targets, not evidence
of two competing learned mechanisms.

## Signed vectors answer the regional question more sharply

For final receiving residuals define

`B=hFF-hQQ`, `S=hFQ-hQQ`, `R=hQF-hQQ`,
`I=hFF-hFQ-hQF+hQQ`, so `B=S+R+I` exactly.

The frozen primary asks whether source-only S recovers B within 10% relative
error, cosine .99, and .10-nat margin error, at both whole and verdict scopes.
It fails in all cases. Whole-vector reconstruction error is a median **62.69%**;
receiver-only reconstruction error is **81.59%**. Neither meets the overall
24/32 and per-stratum 6/8 rule.

The interaction's norm is a median **35.78% of B**, ranging from **26.24% to
61.31%** across cases. Its final finite margin factorial is a median **.667 nats**
in absolute magnitude. The declared negligible-interaction account has 0/32 passes.
All complete SiLU margin contrasts exceed the .10-nat descriptive materiality
threshold; none is discarded for being small.

The median signed projections onto B are .770 for S, .322 for R and -.019 for I.
They are separately pooled medians and are not causal percentages adding to
100%. In particular, a small signed projection can coexist with a substantial
vector norm. Counting only the projection would miss the interaction that the
full vector criterion detects.

The source intervention has the larger median projection in this setup, but
that is substantially weaker than a source-only account. The remaining native
error in FF also cannot be blamed on quadratic SiLU: its only approximated
primitive family is softmax. Its error includes the consequences of earlier
attention approximations on all later states. This does not isolate one head,
layer, learned feature or uniquely responsible primitive at a fixed input.

## Why the comparison is qualified

The protocol and original source were frozen at `2c29385` before model work.
All 160 forwards complete: 32 zero-edit controls plus 128 factorial arms.
There are no new ordinary native True runs. Reused inputs, model revision,
checkpoint bytes and public-library source match 242b's pinned records.

The two old endpoints and zero-edit controls replay every block-tail coordinate
and both scores exactly. Changing receivers leaves every source state exactly
unchanged for QQ/QF and FQ/FF. Source-only MLP edits also leave first-block
receivers unchanged, as the architecture requires. The finite activation rows
outside the selected intervention remain exact.

Live qualification retains **78,272** checks: 17,920 False primitive primals,
4,480 current finite-SiLU primals, 6,272 first and 6,272 second CPU64 AD actions,
their **25,088** FD endpoints, 8,064 installed maps and all region/replay/prefix
controls. Maximum first/second AD errors are 5.56e-16/2.78e-16, first/second FD
errors 3.54e-10/9.22e-8. Installed-map discrepancies are exactly zero. The new
separate saved-array audit passes **512** checks, including finite-head .005
bridges and vector closure. No bound changes.

Derivative qualification uses normalized directions, checks the actual Q rows
installed and scales back before Float32 casting. Softmax derivatives belong to
the public-anchored exponential tilt. They do not resolve or recertify older
Float32 block-Jacobian disagreements.

The original scorer stopped after checking all cases because the serialized
counter omitted its zero unstable count. The separately frozen **243b** scorer
changes only that metadata lookup and records both source hashes. Original code,
failure JSON and traceback are retained. No model rerun or numerical-rule change
was used to repair that reporting failure.

## What remains unresolved

We now know that both regional policies change the native approximation error,
and that their combined response needs a substantial interaction vector. We
do not know whether that interaction is mainly generated anew at receiving
SiLU operations whose inputs changed with the source, or inherited and reshaped
by attention, normalization and products later in the network.

A useful next diagnostic would separate those two terms at the same receiving
inputs, then remove the identified local contribution and measure the resulting
vector and finite margin. Only a subsequent frozen rule evaluated on fresh
contexts could turn that account into prospective progress on Question B.

The 32 templates are dependent and use one small checkpoint. No learned-circuit,
priority, competent-verifier shutdown or general predictive success is claimed.
Earlier qualifications, failed grounding controls and raw numerical failures
remain unchanged.

[Protocol](../experiments/243-source-receiver-silu.md) ·
[Reporting correction](../experiments/243b-reporting-correction.md) ·
[Every case, seal and numerical record](../../results/regional-silu-243/README.md)
