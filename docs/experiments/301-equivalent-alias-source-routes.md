# 301: does an equivalent alias act through the Boolean source or its readers?

This targets a controlled input-content distinction left unresolved by the
older source/receiver studies. Their prompt backgrounds changed instruction
wording, both field names, lengths and positions together. Earlier field-name
studies already controlled some lengths; this is not a claim that they did not.
Preserve all old source seals, failures, grounding limits and numerical caveats.
No learned circuit, novelty priority or general Question B solution is claimed.

## Controlled content intervention

Within each pair, the instruction, passage, task, final `supported` field,
sequence length and absolute token positions are IDENTICAL. The instruction
explicitly declares `first_value` and `record_flag` interchangeable names for
arbitrary irrelevant metadata. The supplied completion chooses one alias, then
the metadata Boolean, then `supported`. Only the two alias token IDs differ
between backgrounds A/B; only one Boolean token differs within either background.
Both aliases are valid under the same instruction. This isolates the chosen
alias spelling under this instruction, not every semantic effect of field names.

32 fresh task pairs cross four related encodings, both alias-mention orders,
both fact orders and both question targets. They are 128 full inputs absent from
the available historical hash catalog, not independent task domains. Failed
grounding controls remain failed; the target is conditional preference dependence.

Let receiver background r and source-generator background s be A or B. Observe
ordinary False in each background. The alias edit is close to the Boolean; only
the short causal-prefix tail from its first changed token to BEFORE the Boolean
can differ. Prefix K/V before that tail are exactly shared, and the tail is
identical under False/True because it precedes the Boolean.

To generate AB, use A's actual inputs and evolving receiving computation, but
replace ONLY the source-query attention's prefix-tail K/V with B's observed
False tail at every layer. Keep current source self-K/V, source query, all
MLPs/norms/residuals, and all receiving-query attention finite. Install only the
source attention result; other attention outputs stay untouched. By causality,
this generates B's complete source state while the receiving rows directly read
A's prefix. BA does the converse. Source False signatures must match their
donor before sealing; source True signatures/emitters must match their donor
native True after measurement, at all28 blocks. These are exact controls, not
independent discoveries. Hybrid receiving states can be off manifold.

Both False and True versions of each hybrid evolve freely. Do not reuse an
ordinary False receiving state as a hybrid baseline. This changes source centers
and Boolean differences together; it does not identify a pure truth feature.

## Frozen alternatives and targets

Define D_rs as final True-minus-False receiving displacement in each corner,
and C_rs as the actual finite True-minus-False candidate margin. The native
alias contrast is T=D_BB-D_AA, with K=C_BB-C_AA. Test both reference contexts:

- Source account: D_AB-D_AA and D_BB-D_BA each predict T.
- Remaining-context account: D_BA-D_AA and D_BB-D_AB each predict T.
- Random control: a single seeded source-prefix K/V perturbation at A, matched
  to the actual A-to-B prefix change per layer/KV head/tail position, is reused
  for both bits. Its Boolean-response change is compared with T.

Each account must pass BOTH receiving and verdict vectors in BOTH reference
contexts, norms>=.001, relative error<=.25, cosine>=.95, |K|>=.25nat, and both
finite coupling-contrast errors<=min(.10nat,.25|K|). Random uses its one defined
A context. All32 remain in denominators; below-materiality cases stay failed.

H1: source passes>=24/32,>=6/8 each layout,>=12/16 each mention order, while
remaining/random each pass<=8/32. H2: remaining passes under analogous source/
random restrictions. These hypotheses concern defined causal routes; "remaining"
includes direct prefix reads and receiving dynamics, not a learned receiver feature.
Neither hypothesis rescues the other. Report every signed vector, coupling and
interaction D_BB-D_BA-D_AB+D_AA, including large norms with small projections.

Separately freeze a full-response transfer forecast:

    P_B = Float32(Float64(B_F) + Float64(AB_T) - Float64(AB_F))
    P_A = Float32(Float64(A_F) + Float64(BA_T) - Float64(BA_F)).

Apply the actual finite final RMSNorm and FULL vocabulary head at these final
states. Native success requires both receiving/verdict displacements within5%,
.99cosine, norms>=.001, and actual margin error<=.10nat. A pair passes only if
BOTH aliases pass. This cannot rescue H1/H2, especially if the alias contrast
itself is small. Two complete False observations and substantial finite hybrid
computation are used; this is not a compact or single-False-state predictor.

## Numerical qualification and temporal seal

First verify known source-only, remaining-only and joint-only causal response tables, then qualify four unequal-length tiny CPU fixtures through BOTH phases. Require
exact untouched-prefix, donor-source, emitter, zero-alias and native self-prefix
controls. Reject a coherent wrong donor with consistent QK/softmax/PV/projection,
a coherently rescored wrong False baseline, source-signature corruption and
immutable progress overwrites. The head fixture uses four additional candidate
head/norm calls for the deliberately wrong baselines. No checkpoint is used.

Production witnesses: eight fixed pairs (both mention orders,reverse0,target0
in each encoding), blocks1/15/28. Save actual source Q, recipient/used K/V,
source mask/raw logits/probabilities/head outputs and projected source write.
Independent NumPy reconstructs the prefix route, CPU64 QK/softmax/PV/output map,
random norms and finite candidate heads. Probability/head-output max<=1e-4;
projected source-write max<=1e-3 AND relative L2<=1e-5; finite head<=1e-4.
Original raw scaled-QK and output-projection1e-4 diagnostics are separately
retained even if failed. Random prefix norm error is scaled by max(norm,.001)
per head/tail row and must be<=1e-4. No derivatives are used or recertified.
Withhold scientific outcomes if required checks fail; preserve any attempt.

Freeze inputs/code/protocol before checkpoint execution. Complete all forecasts,
zero controls and their numerical audit, then publish a global hash-bound seal
before measuring ordinary native True in either alias. Verify remote head,
public seal bytes, sources/runtime and all forecast/metadata hashes first.
Source True states are generated within the hybrid computation, not supplied
from observed native True trajectories. Native self-prefix controls must replay
all corresponding native source/receiving traces and scores exactly.

## Budget, storage and stopping rule

Forecast: A_F,B_F,AB_F,AB_T,BA_F,BA_T,RA_F,RA_T,AZ_F,BZ_F =320 full forwards.
Native: A_T,B_T,AO_T,BO_T =128 full forwards, including64 ordinary True endpoints.
Source-prefix interventions add one full QK/softmax/PV computation each:7168
forecast plus1792 native. Witness raw diagnostics add192+96 full QK products.
The two final transfer predictions per pair add64 full final norms and64 full
vocabulary heads. CPU audits/output maps and original model operations are
additional work. No speedup claim is made.

Use one shared offline checkpoint,8GiB reserve,512MiB forecast/witness cap,
256MiB native cap and64MiB output-map cap. Only the short alias-dependent cache
tail is saved, not a new model or a full prefix cache for every run. Never refill
pruned old arrays. Progress receipts are immutable and count-specific.

A supported route would locate the consequence of a controlled alias edit under
these interventions. It would not identify semantic neurons, explain all older
schema effects or solve Question B. If neither passes, close these sufficient
accounts without tuning layers, aliases, materiality or population thresholds.
The private article remains local and is backed up before any revision.
