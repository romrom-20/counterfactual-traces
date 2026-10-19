# 240: the local SiLU omission is predictable; the native-response gap remains

**A mask derived from False and forecast gates recovers the missing MLP16 SiLU
contrast in all 32 old contexts.** Direct activation restoration reduces the
original generator-contrast error from 45.2% to 2.94% (whole receiving sequence,
medians). This converts239's target-dependent concentration pattern into a
qualified, conditional primitive intervention result.

**It does not solve Question B.** Even a fully finite MLP16 and all remaining
blocks, starting from the forecast input, leave 25.2% native whole-displacement
error and 1.09 nats margin error. The selected restoration changes those to
25.2% and 1.09 nats too. The remaining failure cannot be explained by this local
SiLU approximation omission alone.

## What was frozen and what the mask can see

Protocol, runner, selection rule and separate auditor were frozen in commit
`3affecb` before model work. All32 masks were sealed before the intervention
phase opened native receiving inputs. These are the same previously observed,
dependent228 contexts; this is not fresh confirmation.

The selection rule reads the public False gate g0, the gate g1/up value computed
at238's installed own4 input, and down-projection column norms. It uses a
conservative analytical bound on the omitted quadratic SiLU Taylor remainder:

`B = max_interval sigmoid(g)(1-sigmoid(g)) * [3+max(|g0|,|g1|)] * |g1-g0|³/6`.

Coordinates are ranked by `B * |up_own| * ||down_column||`, with a fixed1% budget
over all receiving neuron-position coordinates. There is no fitted readout,
native gate, native error or final score in the selection. Three SHA-seeded
random masks match the selected count in each row. The own4 path itself uses
sites selected in earlier retrospective work, so mask blindness does not turn
the experiment into a fresh prospective model evaluation.

At each of two supplied inputs (native and forecast), we directly replace the
public SiLU activation by its False-anchored quadratic approximation, then
restore the exact public activation on the selected coordinates. Normalization,
the changed up value, product and down projection remain finite. All subsequent
blocks and the final norm/head are finite. Native inputs are oracle diagnostics.
The own-input reference is its own full-SiLU continuation, not the native True
response. Prefix/source rows retain their declared causal-computation provenance.

## Local recovery succeeds, with a position-dependent limit

The primary criterion is recovery of the missing native-minus-own SiLU-tail
contrast on the **whole receiving sequence**: remaining error<=25%, cosine>=.95,
at least24/32 and6/8 in each schema/order stratum.

- **32/32 pass**, with8/8 in each stratum. Median remaining error is6.88%,
  recovery cosine .99756.
- The separately reported verdict-only scope passes16/32, all descriptive;
  generic passes0/16. Median remaining error35.2%, cosine .903. The mask does
  not provide the same proportional recovery at every position.
- Adding the actually restored vector to239's original formula lowers that
  full generator-contrast error45.2%→2.94% on the whole receiving sequence,
  and11.4%→2.75% at the verdict. This is a new conditional intervention result;
  it does not rewrite239's frozen failures or its joint four-site conclusion.

The distinction between the last two points matters: failing to recover a large
fraction of a small omitted verdict term can still leave a small error relative
to the complete generator contrast. Both denominators are published.

## Final consequences are measurable, mostly below the materiality guard

Relative to the **same-input full-SiLU continuation**, restoration reduces
median final discrepancies as follows:

- Native incoming input: whole-vector gap5.752→.449, verdict-vector gap
  1.418→.100, margin gap .0809→.00393 nats.
- Forecast incoming input: whole-vector gap1.911→.117, verdict-vector gap
  .551→.0559, margin gap .0600→.00137 nats.

Matched random masks leave much larger vector gaps and margin discrepancies:
native random median margins .0774–.0842 nats; forecast .0450–.0458 nats.
As a descriptive observation, restoration halves both verdict-vector and margin
gaps in32/32 for each incoming state. This does not override the frozen guard.

The registered behavioral criterion additionally requires an original margin
gap>=.10 nats, a verdict-vector gap>=.001, and improvement over all three
controls. Only **3/32 native-input cases are eligible**, and all three pass.
**0/32 forecast-input cases are eligible.** Counts stay out of32; the required
24/32 and6/8-per-stratum behavioral account therefore fails for both states.
Calling this a general consequential sparse mechanism would overstate the test.

![Local repair and the remaining final response gap](../figures/240-target-blind-silu.png)

## Why this does not answer the native generation question

The native-input full-SiLU control replays the original native final state and
scores exactly. Restoring only the selected1% from that oracle input leaves
.417% median whole native-displacement error and .00393 nats margin error.
That shows the defined local approximation error can be repaired conditionally.

From the forecast input, however, the **full** finite continuation still has
25.16% whole native-displacement error, verdict-vector error11.55, and margin
error1.088 nats. The selected approximation gives25.15%,11.55 and1.090 nats.
Accurate local SiLU treatment does not recover the missing incoming state.
This new continuation makes every operation after the supplied MLP16 input
finite; it is a diagnostic computation budget, not a proposed compact predictor.

The supported explanation is precise: under finite normalization/up/product,
239's MLP16 formula omits higher-order SiLU activation changes; a forecast-derived
bound locates enough of those terms to repair their whole-sequence contrast.
This explains an approximation failure. It does **not** explain why the input
Boolean generated the unknown native gate/input displacement in the first place.
The useful next target is the error already arriving before MLP16, rather than
another refinement of this same local Taylor omission.

## Qualification, work and remaining boundaries

All **1,568 separate audit comparisons** pass, including128 exact mask/control
reconstructions,384 installed-activation checks,384 independent Float32 local
reconstructions and384 finite-head checks. Local Float32 reconstructions agree
exactly; all384 raw Float64 bridges pass1e-4, maximum4.39e-5. The bridge to239's
missing term is at most4.00e-5 under its original1e-4 bound. Public anchor
rounding is at most4.29e-7 and is retained explicitly. CPU64 autograd validates
the analytic False SiLU first and second derivatives within4.45e-16; no new block
derivative is used or recertified. Native final replay and score replay are exact.

New work:32 ordinary False forwards,32 source-only blocks,32 mask-preparation
MLPs,384 intervention MLPs,4,608 suffix blocks and384 finite heads. No new
ordinary True forward, paid API call or browser/literature search is used.
These work units do not establish a speedup.

No learned feature, circuit, generality or novelty priority is established.
The source-only budget clarification and all older numerical failures, failed
whole-response forecasts and failed grounding controls remain in force.

[Frozen protocol](../experiments/240-target-blind-silu-tail.md) ·
[All cases, audit and reproduction](../../results/target-blind-silu-240/README.md) ·
[Direct intervention](../../research/boolean_coupling/tail240.py) ·
[Separate auditor](../../research/boolean_coupling/audit240.py)
