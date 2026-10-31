# 298: do selected contextual messages explain the final-block query effect?

Retrospective primitive diagnostic on the eight preselected297 native witnesses.
It is not a fresh forecast, full native Boolean-response explanation, learned
feature identification, or general Question B solution. Freeze code, position
sets, parent hashes and criteria before measuring new terminal interventions.

## Hypothesis and fixed parents

297's private posthoc position readout suggested message-end positions for
generic prompts and instruction Boolean-value positions for descriptive prompts.
This motivates, but does not validate, a schema-conditioned group G. All eight
available witnesses are retained:four encodings,two schemas,order0,target0.

At native pre-block28 receiving inputs, use the saved current-query and False-
query attention probabilities pN,pB with identical native K/V. Define each
position group's attention-write change by

    dW_group = Wo concat_heads(sum_j_in_group (pN_j-pB_j) V_j).

This is a message contribution to a defined local query contrast. It includes
softmax competition and need not correspond to another realizable attention
distribution. It does not identify a feature from token identity. Message-end
states may contain summaries of the whole preceding message.

E=all im_end positions before the edited source. I=all true/false token variants
(with/without leading space) in the system message, before its first im_end.
G=E for generic and I for descriptive. X is the swapped choice. U=E union I.
H=all key positions except G. J is a same-cardinality neighboring-prefix control:
for each sorted G position choose the nearest preceding unused position outside
E union I. R1/R2 are seeded random residual directions matched to G's norm at
each receiving row; not on-manifold controls. A=all key positions; Z=empty.
Every exact position and parent hash is frozen in parents298.json.

Use actual full-shape Float32 output projections of the saved N/B head outputs
to form WN/WB. Start from the SAME native pre-block28 receiving state r. Execute
the official post-attention RMSNorm, MLP, residual, final RMSNorm and FULL
vocabulary head at r+WN or r+WB. For every group execute both:

- addition: r+WB+dW_group, followed by the finite terminal computation;
- removal: r+WN-dW_group, followed by that same finite computation.

No query, key or value is regenerated. All later nonlinear and readout operations
are finite. After attention, remaining operations are row-wise. Zero padding of
unused earlier rows retains the original full sequence shape; comparisons cover
only saved receiving rows. Exact N final/scores replay is REQUIRED before any
other terminal intervention. This does not claim full-prefix reconstruction.

## Frozen scientific criteria

Target the final conditional query response D=finalN-finalB, not the much larger
native Boolean response. Both the whole receiving vector and verdict vector
must have norm>=.001. A group's addition effect finalAdd-finalB AND removal
effect finalN-finalRemove must each match D within25% relative error,.95cosine.
Let dm=marginN-marginB. Require |dm|>=.02nat; both finite margin errors must be
<=min(.10nat,.25*|dm|). All eight cases stay denominator, including ineligible
low-effect cases. Score-only success cannot establish a vector explanation.

H1:G passes>=6/8,>=3/4 each schema,>=1/2 each encoding; X,H,J,R1,R2 each pass<=2/8.
U is a separately reported broader diagnostic and cannot rescue H1. A is a
calibration, not a discovered mechanism. Z is an identity control. Reject H1 if
its specified criteria fail; no group, token, head, layer or threshold search.
If last-block effects are not material, retain that result and stop here.

Report signed vector projections, full vector errors and finite margin changes
for both addition and removal. Do not infer a learned circuit or absence of
other sufficient causes from this single contrast and small retained family.

## Numerical qualification and work

First qualify tiny unequal-length CPU Qwen fixtures, role/neighbor selectors,
native reconstruction, additions/removals and deliberate corruptions. For the
checkpoint, require exact native N receiving states and candidate scores before
running further conditions. Require exact zero-group identity and correct input
installation. A's all-position message must bridge WN-WB at1e-4; its finite
addition/removal must reconstruct the appropriate N/B final at1e-3 and scores.005.

Independently reconstruct each finite terminal MLP in CPUFloat64 from its actual
saved middle state and checkpoint weights. Required final-state max error<=1e-3
AND relative L2 error<=1e-5; finite candidate-head bridge<=.005. Preserve original
1e-4 coordinate diagnostics separately, even if they fail. These explicit bounds
qualify this finite terminal computation, not old Jacobians or bitwise-rounded
derivatives. Random norm matching relative<=1e-6 (zero rows exact).

Only the final MLP/norms, attention output matrix and full vocabulary head need
loading from the ONE shared pinned checkpoint. No full-model forward, old
pipeline replay, model copy, download, backward pass or derivative is needed.
Two output projections and20 terminal MLP/head continuations per case:160 MLP,
160 full heads,16 output projections total. CPU audit/projection work is additional.
Native-calibration failure stops the case before the remaining19 continuations.

Use the shared-cache launcher with8GiB reserve, cap new raw archives at64MiB,
keep count-specific immutable progress records and all failed attempts. Bind
every native/witness parent to297's public inventory; preserve all133 earlier
scientific source bytes. Withhold scientific outcomes on required numerical
failure. Only after a stable finite account survives may a False-state-only
hypothesis be frozen and tested on fresh inputs. The private article stays local.
