# A coherent nonlinear update predicts the native Boolean response

**All 32 sealed forecasts pass the native, trajectory and joint-product-effect
criteria.** This advances the prediction part of Question B under an expensive,
explicit budget. It does not identify a compact learned circuit or explain why
training produced this particular schema dependence.

Earlier studies recovered conditional sensitivity changes while supplying edited
source, attention and head computations. Study 291 showed why a scalar fit is
insufficient: partial coefficient restorations could predict margins while
missing most of the gradient and receiving-vector contrast. This experiment
instead starts from one ordinary False computation, the known Boolean edit and
weights. It forecasts the complete response from its own evolving states.

![All-case native errors and layerwise primary forecast](../figures/joint-gate-292/joint-gate-forecast.png)

## What is actually predicted

The edit replaces token 895 (`false`) with 830 (`true`) in an instruction-declared
irrelevant earlier field. Later conditional True/False candidate scores define
the verdict margin. The 32 complete input pairs contain new literals across four
prefix layouts, two schemas, two fact orders and two question targets. All 64
full-input hashes are absent from the available 2,176-hash historical catalog.
These are sixteen dependent schema pairs within related synthetic templates,
not 32 independent tasks. One pinned Qwen2.5-1.5B-Instruct checkpoint is tested.

The primary forecast P replaces SiLU at every affected source AND receiving
MLP, in all 28 blocks. Rows before the edited position remain causally independent
and finite. Every other operation remains finite: attention, query/key/value
adaptation, RMSNorm, learned gate/up/down projections, multiplication and the
full-vocabulary score head. No native True source or receiving state, edited
SiLU value, True gradient or earlier accurate edited forecast is supplied.

At each affected activation call the actual public SiLU is given the saved
False gate **before** execution. Its output must replay the False activation
exactly; P then installs its own predicted activation. Thus a native edited
activation is not secretly calculated and discarded. False-zero controls replay
every candidate's False trajectory exactly.

## One coherent value-and-slope update

For current predicted gate g, construct a rational SiLU-shaped function R by
scaling g, approximating tanh with a bounded rational core, then applying two
fixed tanh double-angle steps. The exact formula is in the
[frozen protocol](../experiments/292-joint-gate-forecast.md).
With observed False gate gF, actual Float32 activation phiF and smooth False
SiLU derivative d1F, use

```text
lambdaF = d1F - R'(gF)
phi_hat(g) = phiF + R(g) - R(gF) + lambdaF*(g-gF)
slope_hat(g) = R'(g) + lambdaF
```

Value and slope describe the same surrogate function, anchored at False. The
predicted activation is cast to Float32, multiplied by the current predicted up
value, and projected by the real down matrix. Current gates and up values come
from the candidate's own preceding states. The core is bounded; the anchoring
correction means the full function is not claimed globally SiLU-range-preserving.

This is standard analytic approximation of a known primitive, not discovery of
a learned feature, a new approximation principle or a priority claim. Its
near-native accuracy should be read partly as successful alternate numerical
execution. Replacing a known activation with a very accurate approximation is
a much weaker explanatory achievement than identifying which learned neurons
and messages create the schema effect.

## Frozen forecasts, then outcomes

The 102 scientific source files, cases and decision rules were frozen before
checkpoint use. The complete unequal-length tiny CPU pipeline and five coherent
negative fixtures passed first. An earlier tiny run completed its numerical
checks but hit a duplicate table-field name during outcome assembly; the original
run/log were retained, the field was renamed before the scientific freeze, and
the complete corrected pipeline was rerun. No endpoint or bound changed. All 32 production forecasts then passed 76,128
required numerical checks. The global seal and forecast records were pushed in
commit [`ffc9707`](https://github.com/romrom-20/counterfactual-traces/commit/ffc9707722102b1d0c9e9ef8f9239f453c8094dc)
**before** native measurement. The original native gate verified that exact
remote branch head, public seal bytes and every local archive/metadata hash.
No production True outcome was measured to choose this rule or its parameters.

The primary population threshold was at least 24/32, with at least 6/8 per layout:

- **H1: complete native response.** Receiving rows strictly after the source and
  the verdict displacement each need norm>=.001, relative error<=5% and
  cosine>=.99, plus finite margin error<=.10 nat. P passes 32/32, eight per layout.
- **H2: trajectories.** H1 plus the stacked 28-block source-state, receiving-state
  and source/verdict MLP-write trajectories, each within 10% and cosine>=.995.
  P passes 32/32. These are complete native displacement trajectories, not a
  small forecast-minus-donor correction.
- **H3: joint-product effect.** P minus the product-affine control must predict
  the unknown native-minus-control vector and signed margin effect, under the
  separately frozen 25%/.95 vector bounds and materiality rule. P passes 32/32;
  all 32 targets are material. Exact native T is separate calibration.

P's median final receiving relative error is 0.0000130; its maximum is 0.0000223
(0.00223%). Median verdict error is 0.00000877. Median margin error is 0.0000191 nat;
maximum 0.0000573 nat. All source, receiving and selected-write trajectory criteria
pass. At six predetermined source/verdict probes, all 192 compound-coefficient
records pass too. Those coefficient diagnostics evaluate finite SiLU **after
the seal at P's own forecast inputs**, not at supplied native True parents.

## What the failed controls tell us

All controls use their own evolving states, known edit and False anchors:

- **D: coarser rational core, zero double-angle steps.** Native 0/32; median
  receiving error 6.11%, margin error .138 nat. Fine transition accuracy matters.
- **C: False cubic at every affected activation.** Native 0/32; median receiving
  error 57.0%, margin error 1.409 nat. A local Taylor description does not compose
  accurately over this finite edit with these rules.
- **B: C with elementary activation range clipping.** Native 0/32; median
  receiving error 9.17%, margin error .313 nat. Clipping helps substantially, but
  range constraints alone do not recover the tested response.
- **A: P's activation approximation, False-affine gate/up product.** Native 0/32;
  median receiving error 355%, margin error 6.308 nat. Omitting the joint operand
  change at all affected MLPs badly disrupts the trajectory in this family.

For a single product, the exact finite identity is

```text
(phiF + delta_phi)*(uF + delta_u) - phiF*uF
  = uF*delta_phi + phiF*delta_u + delta_phi*delta_u.
```

A deletes the last term. P retains it as the current gate/up states evolve.
This is a direct intervention on a known operation throughout the affected
network. It establishes the failure of this specified simultaneous removal,
not universal necessity of every individual product or a sparse learned circuit.
Later incoming states differ between the arms, so the experiment does not assign
a unique missing direction to one layer or neuron.

The looser H3 test alone is insufficient: D passes 32/32 and B 30/32 despite
both failing every H1 case. Native-minus-A margin targets span 1.832–12.829 nats;
that large reference permits sizable absolute errors under H3's relative bound.
P's full-response success is the stronger evidence. All controls and every
original denominator remain in the published tables.

## What this adds to Question B

There is now a constructive answer at this budget: one ordinary False observation
plus the known edit and weights suffices to forecast these native displacements
without native edited SiLU evaluations, using coherent nonlinear gate updates
and joint products throughout the adaptive network. Previously supplied edited
receiving parents are not necessary inputs to this tested full forecast.

The evidence supports a computational constraint rather than a named circuit.
The product-affine and local cubic rules lose information that this
much more accurate finite-transition surrogate retains. Clipping repairs part
of the cubic error but does not fix it. This is consistent with the finite edit
moving through state-dependent gated responses. It does not prove that SiLU is
the sole cause of schema dependence: adaptive attention and all normalization
remain fully available and may be essential to the generated directions.

The harder explanatory question remains: can we identify which learned
source/receiver operations generate the consequential displacement, and predict
it while replacing substantially more of the finite network? A full dense
forward with a highly accurate activation surrogate does not answer that.
An honest next step must fix an additional information/work restriction before
testing it; more precision at the same full-computation budget would add little.

## Numerical qualification and retained failures

The pre-native gate passes 76,128 records: exact substitution/prefix/installation
and zero/replay witnesses, rational value/slope reconstruction and independent
Torch Double AD/finite-difference checks, and CPU64 finite readout comparisons.
No finite edited native SiLU was needed for that qualification. The native phase
passes 96 additional checks: 32 exact unhooked full-True state replays, 32 exact
score replays and 32 CPU64 head comparisons within .005 nat. This qualifies these
calculations; it does not independently certify every library operation.

There are no new whole-model Jacobian claims. Earlier primitive/coordinate
failures, 291's two narrow failures, 290's eight, 288's seven and 287's failed
original gate remain failed. They are not retroactively recertified by this
finite experiment. Historical grounding failures still prevent a competent
verifier or deployment-failure claim. Novelty priority and a direct attribution
patching comparison remain unestablished.

Posthoc signed Float64 vector accounting is published separately. It reconstructs
all final native/effect vector statistics, and shows each block's P displacement
error and signed projection onto the native displacement. This uses sealed arrays
only, no model calls or new forecasts. It is descriptive, not primitive causal
localization. Public CI checks byte seals, record arithmetic and decisions;
large omitted activation arrays and the checkpoint are not independently rerun.

## Work, resources and reproducibility

The complete run uses 512 full forwards across 32 cases: 352 for False/forecasts/
zero controls, 96 pre-native replay forwards, 32 native measurements and 32
unhooked native replays. A single P forecast still executes all 28 dense blocks,
including host Float64 approximation and device copies. This is not a demonstrated
speedup or a compact whole-network method.

Original 292 stopped at the unchanged 384 MiB guard after one passing forecast;
292b stopped after loading, before its first archive. Both stops remain preserved.
A separately frozen resident resource executor retained the same original
Float32 MPS model across phases. It required 640 MiB after loading, kept the
original per-case/archive guards, replayed all 42 original first-case arrays
exactly before case 2, and verified parameter objects/versions throughout.
Model loads: one. No scientific archive, weight, private article or frozen source
was deleted/recompressed to continue. Lossless filesystem CoW/compression of
identical checkout/runtime bytes is documented as resource handling, not a
changed scientific method. Production resident wall time 736.31 s includes the
public-seal wait and all audits; it is not a marginal inference benchmark.

Forecast/native case archives total 216,243,178 bytes, below 256 MiB; they remain local.
[All scalar evidence and inventories](../../results/joint-gate-292/README.md),
[reproduction commands](../../research/README.md#joint-gate-292), and the
[frozen resource continuation](../experiments/292c-resident-execution.md)
state exactly what is available. The private article is excluded from Git.
