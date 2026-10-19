# 241: generation, amplification and vector sufficiency before MLP16

Freeze before new transport calculations.240's fully finite continuation cannot
repair the input it receives before MLP16. On all32 old228 contexts, examine
the29 operations attention2 through attention16 (MLP2–15 included). Reuse238's
qualified native-input local remainders and original False-state Jacobians.
No new ordinary model forward, native observation or correction sweep.

For each operation: `e_next=J e+R`. Decompose its exact squared-error increment:

`||e_next||²-||e||² = (||J e||²-||e||²) + ||R||² + 2<J e,R>`.

Call these transported-error norm change, new-remainder energy and their cross
term. The cross term may dominate or be negative. These are a defined algebraic
account, not unique causal percentages or learned features.

Unlike238's scalar adjoint projections, explicitly transport EVERY R vector to
the input before MLP16 by the original remaining J maps. Retain each complete
terminal vector, its local/terminal norm ratio, signed projection and pairwise
Gram matrix. Check that their vector sum equals the original raw forecast error.
Add the explicit known installation term `p_raw-p_installed` to reach240's
actual supplied input error. Bind its exact installed input to240's mask file.
Do not call the installation term a nonlinear generator.

## Predeclared retrospective concentration test

Greedily select a shared ordering of the29 operation vectors: at each step choose
the site minimizing median whole-sequence remaining relative error across all32
cases (chronological tie break). This uses observed vectors and is retrospective,
not a predictive selection rule. Keep the known installation term in every
reconstruction. Report all prefixes1–29, all cases and all strata.

The primary compact account is the first FOUR sites. It requires remaining
relative error<=.25 and cosine>=.95 at BOTH all receiving rows and verdict,
norm guards>=.001,24/32 total and6/8 per schema/order stratum. Publish the first
prefix meeting this rule too, without substituting it for the four-site result.
Failure rejects this defined compact account, not all possible sparse bases.

No selected vector is injected into a real continuation in241, and no causal
primitive conclusion follows from this ledger alone. If a stable high-contribution
site appears, freeze a subsequent primitive test after this diagnosis and before
its outcomes; do not quietly turn a vector ranking into another correction sweep.

## Qualification and storage

Construct the same CPU64 receiving maps from the cached pinned checkpoint,
with False prefix fixed. Batch each previously generated component direction
and the saved actual input-error direction. Every used JVP gets Torch central
FD(h1e-5) at1e-4; save all used JVP vectors, FD errors and endpoint hashes. Compare every tangent
with independent explicit NumPy derivatives at1e-9 and independently computed
NumPy finite differences at1e-4. Actual-error directions must reproduce238's
saved tangents at1e-9. False primals: public bridge1e-4, NumPy/Torch1e-9.
All intermediate vector recurrences and terminal closure:1e-8. Energy identities:
1e-8. Stop and preserve unexpected failures; do not widen any bound or modify
older evidence. No public Float32 whole-block derivative is recertified.

Bind raw228/238/240 inputs to their published inventories, source hashes,
checkpoint and runtime. Store full terminal components, all used directional tangents,
checks and ranking. Intermediate direction stacks and finite differences are reconstructible
from the frozen remainder order and maps; retain hashes. Avoid checkpoint copies.
This remains a retrospective decomposition of a failed forecast, not an answer
to the prospective native-generation question, a learned circuit or a speedup.
