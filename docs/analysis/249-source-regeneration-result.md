# 249: the known source regenerates much of the verdict response, not full accuracy

The source-only algorithm starts with one ordinary False computation and the
known Boolean edit. It computes the changed source row through the first nine
blocks using unchanged False prefix states. Every later row is reset to False
at each of those source component inputs and its output is discarded. Only the
computed source row is installed before block 10. The remaining 19 blocks then
generate every receiving state finitely, without native receiving-state inputs.

All 32 forecasts were sealed before the control stage opened native True state
arrays. These are already observed contexts, so the ordering is a leakage check,
not fresh confirmation. Subsequent controls verify every computed source row
through blocks 1–9 and its entire future trajectory against native True exactly.
The source calculation uses only the known edit, weights and False prefix.

## What it predicts

Against the actual native True response, the source-only S algorithm reaches
median **7.01% verdict-vector error**, cosine **.99760**, and **.194-nat margin
error**. Whole receiving-vector error is **20.63%**, cosine **.97841**. It passes
**0/32** under the unchanged joint native criterion. None passes the verdict
vector bound; seven pass the margin bound. This is not an accurate general
native-response prediction.

An oracle control C supplies the True intermediate receiving rows but keeps the
source and verdict states False at the boundary. It has 101.08% verdict-vector
error and 11.02-nat margin error in median. Those intermediate states alone do
not regenerate the native response in this tested continuation.

For the narrower context-only target P from 248, S has 4.91% verdict-vector
error, 20.24% whole receiving error and .145-nat margin error. This secondary
also passes **0/32**. Relative to P-F, the source contribution's median verdict
projection is .9901; intermediate rows contribute -.0100 and their interaction
.0210. These are separately pooled, allocation-dependent projections.

## What this clarifies

The dominant tested verdict dependence can be regenerated from the changed
source state during later computation. It is not maintained by the earlier
verdict state alone, and a large non-verdict local discrepancy was not the
behavioral bottleneck. Earlier receiving history still matters for exact native
residuals and score margins. The source-only reset loses substantial complete
vector information even when the verdict direction is close.

This does not establish a direct source-to-verdict attention edge: the source
could first update intermediate rows which then influence the verdict. Nor does
it identify a learned feature or explain the source state's trained semantic
content. A useful next primitive test would separately clamp source keys and
values during the later reads, preserving source computation and allowing all
receivers to evolve. That would distinguish communication paths from initial
state-sufficiency and identify whether routing, values or their interaction
accounts for the late effect. It has not been run here.

## Budget, qualification and limits

This is not a speedup. The source stage executes **288 full dense block calls**
for its row extraction; the forecast adds 32 ordinary False calls, 608 finite
suffix blocks and 32 full heads. Controls add 608 suffix blocks and 32 heads,
using oracle inputs. Dense model work is substantial and is not a compact learned
circuit. No unknown True receiving states enter S, but known-context results
cannot establish fresh generalization.

All source/prefix replay and control invariances pass exactly. The separate
finite-head/source/factorial audit passes **256 checks**, maximum 8.41e-6.
No derivatives are used or recertified. Forecast/control timestamps, hashes and
every outcome are retained. Old failures and grounding limitations remain.
Question B is still open.

[Protocol](../experiments/249-source-regeneration.md) ·
[Forecast/control seals, all cases and audit](../../results/source-regeneration-249/README.md)

![State controls and the remaining source-only forecast error](../figures/247-249/248-249-native-states.png)
