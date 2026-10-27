# 285: native source-write prediction across changed prompt structures

Freeze this protocol, all scientific sources, exact inputs and historical donor
selection before computing the new forecasts. All earlier results and failures
stand. This test addresses a limitation of 283/284: their small conditional
correction directions transfer between matched literal examples. It returns to
an actual native first-source write, rather than another approximation-error
selection sweep.

## Questions and fixed information

H1: does the unchanged False-coefficient cubic source rule predict the native
first-source write and final response when the causal prompt prefix changes
structure? H2: does regenerating that direction predict a material native effect
beyond a fixed historical direction with the same predicted magnitude?

The current case supplies one complete False computation, the known single
Boolean token edit and checkpoint weights. At source MLP1, retain finite
attention, input RMSNorm, gate/up projections, product and down projection.
Only the source SiLU output is approximated using its observed False gate gF
and activation phiF. With dg equal to the edited gate minus gF:

```text
L = phiF + d1(gF)*dg
C = L + d2(gF)*dg**2/2 + d3(gF)*dg**3/6
```

Derivatives are those of ordinary SiLU, evaluated only at False. Cast phi to
Float32, use the actual current up branch, finite multiplication and public down
projection. No complete current source SiLU is evaluated for any forecast:
replace its input source row with the False gate BEFORE calling the activation,
then install the predicted phi. All other source tokens, all receiving rows and
all 27 later source MLPs remain finite. The current first attention and projections
are regenerated, rather than inferred from a final False residual alone.

This is the same first-source polynomial rule as 279/280 C1, with a new
structural transfer test and donor control. It is not a novel product rule,
whole-block Taylor approximation, identified feature or compact whole-model
predictor. The model still performs substantial finite work.

## Cases and donor bank chosen without new outcomes

There are 32 cases: four new prefix layouts, each crossing two schemas, two
fact orders and two question targets. Day assignment is fixed to zero. Literal
facts remain those of the earlier apatite examples, isolating structural changes
from new nouns. All four layouts change tokens BEFORE the edited Boolean:

- Evidence is presented as labeled table rows rather than a JSON record.
- The verification procedure moves into the user message under a short system rule.
- An additional user/assistant exchange precedes the evidence record.
- Evidence fields use XML-style tags, with question and answer preceding passage.

Instructions retain the arbitrary metadata bit's explicit irrelevance and the
same meanings of the later verdict. The task is identical, but equivalence of
model behavior is not assumed. The edit remains one 895-to-830 token change;
False and True lengths match within every case. Positions and lengths vary
across layouts. All 64 full input hashes must be absent from the 2,112-sequence
historical catalog. These are related synthetic cases in one checkpoint, not
32 independent facts, new domains or established verifier competence. Earlier
failed singleton grounding controls remain failed.

The donor index is frozen in bank285.json before current forecasts. It uses
ONLY the sealed 279 forecast packages, never native observations, native target
vectors or native margins. For each schema/order/question-target group at
assignment zero, take the equal mean of the two old literal blocks' forecast
vectors `Float64(C_write)-Float64(false_write)`. Source files, forecast audit,
seal, case metadata and array hashes bind the bank. Old bank production is
historical work, not a cost-free new observation.

## Arms and generation order

Run one ordinary False observation, then C, L, D, M, Z and FZ for every case:

- C is the primary cubic write, L the affine activation control.
- D installs `Float32(current_False_write + historical_mean_delta)`.
- M installs that historical direction rescaled to the norm of the current
  C-minus-False forecast write, before the same Float32 addition. It explicitly
  spends the C calculation and tests direction specificity conditional on that
  forecast magnitude. No target norm enters the rescaling.
- Z installs the current False write. FZ uses the False input and must reproduce
  the entire observed False result exactly.

D/M/Z also prevent current source SiLU evaluation before replacing their write;
merely overwriting a native True MLP output would violate the information budget.
Their discarded branch may use the same cubic phi as C. All interventions alter
only the first source MLP write; the outer residual and later computation adapt.
C runs before M. No source/receiving trajectory, native write or native True
completion supplies any candidate. There are 224 forecast full forwards. No
sparse coordinate selection, iterative correction or new hyperparameter search.

## Global seal before native measurement

Complete all 32 forecasts, then their numerical qualification, then a global
write-once forecast seal. Publish exact forecast/qualification records before
allowing native True. Production native entry checks the exact seal in a public
commit and the pushed branch tip. Record that receipt before the first True
call. Synthetic CPU preflight receipts are explicitly labeled and cannot serve
as production public proof.

After the seal, measure one ordinary True computation per case. The separate
native audit also performs 32 unhooked True replays. Preserve any interrupted
attempt; do not silently resume, overwrite or recompress evidence. Release only
unused MPS cache during production; initial reserve 1 GiB, per-case reserve
512 MiB, combined new arrays capped at 96 MiB. These guards do not guarantee
host resources. A changed runtime or scientific method needs separate qualification.

## Frozen decisions

Evaluate every case in the denominator, including materiality failures. H1 and
H2 are separate preregistered claims; neither arm nor one claim rescues the other.
For each, require at least 24/32 and at least 6/8 in EACH of the four layouts.
Report C/L/D/M/Z/T metrics and each component decision separately.

**H1, primary native prediction:** C's first-source write change from False must
predict the ordinary native T-minus-False write with target norm at least .001,
relative error at most .10 and cosine at least .995. Its final residual change
must predict T-minus-False both over receiving rows STRICTLY AFTER the source
and at the final verdict row, with norm at least .001, relative error at most
.05 and cosine at least .99. Actual finite True-versus-False candidate margin
error is at most .10 nat. All criteria must pass jointly.

**H2, direction-specific native repair:** C-minus-M must predict T-minus-M at the
first-source write, all receiving rows and final verdict, with norm at least
.001, relative error at most .25 and cosine at least .95. The actual T-minus-M
candidate-margin effect must have magnitude at least .02 nat; error is at most
`.005 + .25*abs(actual effect)`. These conditions must pass jointly. Ineligible
cases remain failed. Report the same arm-minus-M account for all arms and exact
T calibration; calibration can also fail eligibility. A local-only advantage
or H1 success does not establish H2. Donor/source vectors share coordinates;
whole suffixes from different cases are never subtracted or aligned.

If L also passes, cubic curvature is not uniquely required. If D/M also pass,
H1 does not select recipient-specific regeneration. If H1 passes and H2 fails,
report structural prediction transfer with the native-effect explanation open.
Even H2 success is conditional on the stated intervention, checkpoint and budget;
it does not identify learned semantic features or establish general Question B.

## Numerical qualification and scope

A complete unequal-length, varying-position tiny CPU pipeline must pass before
freezing or loading the checkpoint. Test causal prefixes, source-only clamps,
False primals, all arms, exact FZ/F, ordinary True equivalence, stage ordering,
premature-native/wrong-seal rejection and endpoint denominators. Random tiny
weights need not produce material effects or successful scientific outcomes.

Before native True, independently reconstruct first-source attention from known
embeddings and saved public mask/RoPE, then replay source input normalization and
gate/up projections at the exact executed shape. Public Float32 maps, products,
down projections, source additions, installations and full-vocabulary readouts
must replay exactly. Check False coefficient vectors against separate Double
AD and two central differences at absolute tolerance 1e-4; False SiLU and RMS
Double bridges retain 1e-4. Recompute donor means, casts and forecast-only norms
independently. CPU64 full-vocabulary candidate-head bridge tolerance is .005.

CPU64 gate/up/product/down and polynomial raw comparisons at 1e-4 are explicitly
DIAGNOSTICS; preserve every failure. This predeclared public-execution scope is
not a reconciliation of raw projection differences or validation of a rounded
whole-block Jacobian. No full current source activation is used in qualification.
After the seal, verify prepared first-source operands against ordinary True,
replay its finite product/down/head and independently replay all native outputs.
Original numerical failures, qualified 204 and invalid replacement-agent limits
are not recertified. No priority, efficiency, deployment-verifier or circuit claim.
