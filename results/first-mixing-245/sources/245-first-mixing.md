# 245: where is the first regional interaction generated?

Retrospective, one pinned checkpoint and the 32 observed 242b/244 contexts.
Freeze runner, scorer and protocol before model work. Preserve all earlier
results and qualifications, including 244's failed portability test.

Use 244's finite-attention QQ/FQ/QF/FF trajectories. Before attention2, changing
source SiLU affects only the source row; changing receiver SiLU affects only
receiver rows. Require their full incoming interaction to be EXACTLY ZERO, and
the corresponding Q/K/V additive identities to be exact. No earlier mixed state
can then explain this first attention interaction. This is a regional policy
interaction, not the full native False-to-True displacement or learned circuit.

Capture attention2 receiver queries Q, all keys K/values V, unmasked logits L,
probabilities P and the weighted-value output Y before output projection. Denote
QQ=0, FQ=A, QF=B, FF=C. Form L*=LA+LB-L0 (CPU64 sum, cast to Float32), with the
same causal mask and public finite softmax P*. All subsequent computation stays
finite. At this input, IV=VC-VA-VB+V0=0 and the only analytic mixed logit is
(QB-Q0)(KA-K0)^T times scaling. Check its CPU64 reconstruction at 1e-4; retain
rounding explicitly rather than calling it a new nonlinear term.

The exact probability interaction separates as
IP=(PC-P*) + (P*-PA-PB+P0). Define weighted-value contributions
QK=(PC-P*)VC, SM=(P*-PA-PB+P0)VC,
PV=(PA-P0)(VB-V0)+(PB-P0)(VA-V0).
Then IY=QK+SM+PV+E, where E is the explicitly retained Float32 multiplication
remainder. Require independent finite softmax and matmul bridges <=1e-4 and
maxabs(E)<=1e-4; retain both absolute and relative remainder sizes. Each term's
signed projection, norm and direction are measured on the complete receiver
vector and final verdict row. These allocations use this specified finite
factorial; no claim of unique primitive attribution across decompositions.

Direct interventions on the otherwise native FF run, at attention2 only:
remove QK through logits/probabilities (install P*); remove SM through signed
probabilities PC-(P*-PA-PB+P0); remove PV at the actual weighted-value product;
remove the complete IY there. Also remove 3 deterministic random directions,
matched to PV's per-receiving-row vector norm (all heads together). Signed
probabilities can be negative in the SM diagnostic: this is an explicit
arithmetic intervention, not an alternative native attention distribution.

Primary, testing a possible overgeneralization of 234: PV alone recovers the
local IY with <=10% relative error/.99 cosine at both scopes (target >=.001),
and removing PV reduces the final regional interaction vector norm by >=25%
at BOTH scopes, reduces absolute finite-margin interaction by >=.10 nats
(baseline absolute interaction >=.10), and beats all 3 matched controls at both
vector scopes. Require 24/32 and 6/8 per schema/order stratum. No secondary may
replace a failed primary. QK, SM, full-IY effects are descriptive secondaries.
Also report native residual and margin error for every intervention; a reduced
policy interaction is not a better native predictor. These removals use oracle
factorial components and are not prospective forecasts or correction sweeps.

Every regional trajectory and score must replay 244 exactly. Prefix hashes,
untouched source rows, intervention installation and finite final readout are
checked (head .005). Reuse 244's Q-SiLU directional CPU64 qualification and its
1e-4 primal checks. No whole-block derivatives are required or recertified.
No changed tolerance after observation. Unexpected failure stops and is retained.

Count 128 regional replay forwards (32 ordinary native FF, 96 modified), four
extra local attention products/softmax work per context as actually executed,
and 224 primitive intervention forwards. Seal arrays before separate scoring.
Publish all cases and numerical checks; keep article drafts local.
