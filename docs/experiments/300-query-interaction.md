# 300: where is the middle-block joint query effect generated?

This is a retrospective primitive diagnostic on the eight already measured 299
witnesses. It is not a fresh-input forecast, an explanation of all 299 failures,
a compact predictor or a learned circuit. Freeze code and criteria before the
new conditional interventions. Preserve all 299 sources, outcomes and failures.
Block 15 is selected from the posthoc middle-witness contrast: small query-corner
interaction norm, larger attention-write interaction norm. No layer search follows.

At fixed native parents, qN,qC,qM,qB are 299's four receiving-query corners.
Define qA=(qC+qM)-qB with actual Float32 arithmetic. Let f(q) be finite attention
with shared native K/V and the learned output projection. Define the residual
writes WN=f(qN), WC=f(qC), WM=f(qM), WB=f(qB), WA=f(qA), and L=(WC+WM)-WB.

The real-valued joint discrepancy separates exactly as

    WN-L = (WN-WA) + (WA-L).

WN-WA is the query-generator interaction transmitted THROUGH attention. Because
normalization is nonlinear while query projection/rotary are affine/linear, it
includes that upstream finite normalization interaction, with Float32 rounding.
It is not a named normalization feature or a component independent of attention.
WA-L is the nonlinear attention integration of the two query components at the
same parents. It includes finite softmax and rounding; the output projection is
linear. This is ordinary composition algebra, not a new general law.

Only block 15's receiving attention-output residual writes are intervened on:

- L: additive corner writes, both joint pieces absent.
- A: WA, restoring the nonlinear attention-integration piece.
- R: L+(WN-WA), restoring the transmitted query-generator interaction.
- J: L plus a seeded random residual direction matched to WN-L norm per row.
- O: WN, restoring both pieces; exact native calibration.
- T0: ordinary native True, explicitly repeated to qualify saved parents.

All earlier blocks are native and identical; source and prefix outputs are
protected. The entire remaining suffix, all norms/MLPs/attention and the
full-vocabulary head are finite and live. Earlier native queries and states
are supplied by the known native computation. This is never False-state-only.
T0 must replay every saved 299 native receiving trace and candidate score
exactly before any of the five conditional continuations. A failed T0 saves
partial arrays and stops after one full forward. O must replay T0 exactly.

## Frozen decision and qualifications

Target the final conditional displacement T0-L, including receiving AND verdict
vectors. Require target norms>=.001,relative error<=.25,cosine>=.95,material
|mT0-mL|>=.02nat,and finite margin error<=min(.10nat,.25*|mT0-mL|). All eight
remain denominator, including below-materiality cases. H1: A passes>=6/8,>=3/4
in each schema,>=1/2 in each encoding; R/J each pass<=2/8. H2: R passes under
the analogous A/J restrictions. Neither alternative rescues the other. O is
calibration, not discovery, and low-materiality cases remain failed even in O.

Measure every signed block-write and final-vector effect, plus the final
T0-A-R+L suffix interaction. A local write's large projection cannot establish
final recovery. A successful conditional account cannot explain the full input
Boolean effect or the source of the learned query components.

Independent CPU64 audit reconstructs all five query-conditioned probability,
PV and output-map paths, the operand/restoration identities, installed random
norms and full finite candidate scores. Required probability/head-output max
error<=1e-4; projected-write max<=1e-3 AND relativeL2<=1e-5; native and component
write-contrast discrepancy divided by max(||WN-L||,.001)<=1e-3; head<=1e-4.
Actual installed random norm error is scaled per row by max(||WN-L||,.001) and
must be<=1e-4. Original raw projected-write1e-4 coordinate diagnostics remain
separate even if failed. No Jacobian or old derivative recertification is used.
If any required check fails, withhold scientific outcomes and retain the attempt.

First run tiny CPU fixtures at four unequal sequence lengths, with stronger
random Q/K weights to make nonlinear effects observable. Generate tiny native
parents through the 299 implementation; qualify both controls and finite suffixes.
Reject coherent wrong-primitive substitutions and doubled random controls; a
corrupted reference must stop after one T0 forward. These are software fixtures,
not checkpoint confirmation. All old native arrays/source hashes are checked.

Checkpoint budget:8 repeated native T0 plus40 conditional full forwards=48.
Only the T0 preparation adds an extra full QK/softmax/PV kernel per case and four
full output projections:8 extra attention kernels and32 output projections.
Independent CPU audits and projection work are additional. No speedup claimed.
Use one shared offline checkpoint and8GiB reserve. Cap new case archives64MiB.
Do not copy models, repeat old forecast pipelines or refill pruned arrays.

This can identify where a conditional joint effect is generated and whether
restoring it recovers the tested final displacement. It does not identify learned
features or fully solve Question B. Failed sufficient accounts close this local
line rather than triggering another arbitrary layer or threshold sweep.
