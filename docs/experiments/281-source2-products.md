# 281: target-blind source-MLP2 product selection

Freeze this protocol/code BEFORE checkpoint runs.32OLD280b contexts,16dependent
matched pairs. This is retrospective, not fresh support. Original failed all-
source composition and all numerical qualifications remain unchanged.

## Hypothesis and budget

A False-anchored cubic can miss finite gate movements across SiLU's transition.
Use the known edit, weights, full False observation and one own-input preparatory
B forward (cubic at source MLP1+MLP2). At MLP2 let current g/up and cubic phi come
from B, not native True. Primary score is

    abs(phiF + ReLU(gCurrent) - ReLU(gF) - phiCubic) * abs(upCurrent) * ||downColumn||.

This is a global shape proxy for missing products, not an exact SiLU remainder.
Top ceil1% =90/8960 coordinates, deterministic index ties. Fixed comparator H is
|g-gF|^4 * |upCurrent| * ||downColumn||. R0-R2 are seeded random90 masks. No ranking
uses a native omission, C1 finite activation, True trajectory or final score.
ReLU here only defines a score: no derivative at its zero boundary is claimed.

All first-source MLP1 activations are False-cubic except native F/T. At MLP2:
B uses cubic everywhere; P (PRIMARY) uses finite SiLU only at the90 primary-mask
coordinates and cubic elsewhere; H/R0-R2 use their fixed masks. Before actual
SiLU at approximate sites, unselected source gates are replaced by their observed
False gates. Thus complete current source activation is not secretly computed.
All later26source activations, receiving work, attention/norm/projections/product
and final full-vocabulary readout stay finite. No efficiency/circuit claim.

D keeps cubic activation but directly restores only the primary-mask products;
its source write, full final state and scores must equal P exactly. G0-G2 instead
add seeded norm-matched random source writes to B, matching P's local correction
before Float32 rounding. Their incoming gates/up must match B/P, and no True
information is used to choose them. FZ is a False zero edit with the same rules;
zero score ties choose the first90. It must reproduce F exactly.

Only AFTER these forecasts: C1 makes MLP2 finite (source1stillcubic); A removes
only the primary-mask finite-minus-cubic products from C1. A0-A2 remove random-mask
products from C1. T/F are native arithmetic with observational hooks. These full
finite source2 controls are explicitly calibration/intervention, not forecast inputs.

## Fixed primary and separate stronger accounts

P must jointly pass:
- Original native T-F whole receiving (exclude source) and verdict vectors each
  norm>=.001, relative<=.05, cosine>=.99, actual margin error<=.10nat.
- Local P-B versus missing C1-B source-MLP2 write: norm>=.001,relative<=.25,cos>=.95.
- Actual P-B versus C1-B whole receiving/verdict vectors: norm>=.001,relative<=.25,
  cosine>=.95; |margin(C1)-margin(B)|>=.02nat and effect-margin error
  <=.005+.25*abs(target). Ineligible effects are failures in32.
Require>=24/32 and>=6/8 every schema/order stratum. H/control success cannot rescue
P's primary. Added local/effect targets never replace original native accuracy.

Separately register selective restoration: joint P plus>=2/3 random masks AND
>=2/3 norm-matched random directions failing the same local/actual-effect account.
Separately register removal: C1-A reconstructs C1-B local write and actual receiving/
verdict/margin effect at the same25%/.95/material bounds, with>=2/3 random removals
failing. Both use24/32+6/8 thresholds. A must change only selected products.
A stronger original-behavior necessity account ALSO requires B to FAIL original
native accuracy. Its per-case account is selective restoration AND controlled
removal AND B native failure; it needs>=24/32 and>=6/8 each stratum. Without that
damage guard, even
selective local recovery does not establish necessity for the original tolerance.
Every account/failure is reported separately; no precision-to-necessity inference.

## Numerical checks and data integrity

Use pinned Qwen2.5-1.5B revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306, Float32
MPS/eager/eval and CPU64 bookkeeping. Qualify first/second/third SiLU coefficients
at64False anchors by independent TorchDouble AD and two finite differences<=1e-4.
Check score/proxy with independent TorchDouble ReLU and exact rank/tie/seed replay.
Actual masked source gate/primal hashes, polynomial installs, product coordinates,
Float32 padded public down projections and full-vocabulary heads must replay
exactly (independent smooth polynomial bridge<=1e-4; CPU64head<=.005). Verify source
additions and unchanged rows/prefixes; P/D exact equivalence and FZ/F exact zero.
All candidates' MLP2 mid/gate/up must equal B's own input; source1 unchanged.

Record EVERY raw CPU64 phi/product/write diagnostic at original1e-4, including
failures. These raw diagnostics are not all required under this predeclared
component qualification, and no old whole-block/raw-logit failure is recertified.
Separate code within one investigation is not an independent whole-model proof.
All F/T/C1 archive final/source1-2/scores and B's first-two primitive inputs must
replay280b exactly. Archives are read only after candidates/control execution.

Synthetic CPU-only three-block preflight checks selection/ties, source-only
interventions, direct-product equivalence/removal/random controls and native
observational controls before checkpoint execution. No study cases/checkpoint.

576full forwards (18/case), including shared B preparation, zero, all controls,
native references. No extra local masked maps in producer; masks run in actual
forwards. Audit replays are counted separately. Only source1/2 anchors, shared
own-input rows, unique MLP2phi rows, source writes/final receiving states/scores
and metadata hashes saved; no28-layer primitive duplication. Rawcap64MiB,
initial600MiB/per-case512MiB disk reserve; keep bytecode off, never delete or
recompress old evidence. If controls/qualification fail preserve failure and
stop attribution; do not silently edit frozen code or numerical bounds.

A stable account could motivate a separately frozen fresh test with ALL forecasts
and numerical qualification sealed before any native True measurements. Neither
this old-context test nor sign regimes identify a learned semantic feature.
