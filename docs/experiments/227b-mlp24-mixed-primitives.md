# 227b: finite primitive decomposition of the MLP24 interaction

This follows the completed and separately audited 226b: only MLP24 passes the joint descriptive-context reduction rule (16/16). The question is now which local operation generates that particular mixed product term. All32 contexts remain old and dependent. No new forecasts, learned features/circuits, or solution of QuestionB are claimed. This protocol is sealed before 227b model calls. Existing 204/208/224/225b/226 qualifications and failures are unchanged.

At MLP24 take the four actual post-attention parents x0,xD,xB,xBoth from the 225b suffixes. Set xAdd=xB+xD-x0 in CPU64 then cast to Float32. All source/prefix rows are unchanged. Capture actual finite normalized activations, gates, SiLU outputs, up values, products and down writes for 0/B/D/Both/Add using public modules.

Create two additional local product conditions at xAdd:

* X: replace only the receiving normalized activation by N(xB)+N(xD)-N(x0), after the Float32 cast. All following nodes finite/public.
* Y: same normalized intervention, additionally replace the SiLU output by phiB+phiD-phi0, after the Float32 cast. Up values stay those from X. The product is finite/public.

Define signed product-coordinate terms, in this declared order:

    RN = pAdd-pX
    RS = pX-pY
    RP = pY-pB-pD+p0
    Rall = RN+RS+RP = pAdd-pB-pD+p0.

This is an order-specific finite decomposition. RN includes the effect of the normalization mixed response through all later MLP nodes; RS is the finite SiLU mixing at the additive normalized input; RP is the remaining gate/up cross-product plus retained cast/projection arithmetic. They are conditional operation contributions, not intrinsic features or unique causal percentages. Record projection/cast deviations from additive gates/up explicitly. No new derivative is used.

At the native Both MLP24 incoming state, install pBoth-RN, -RS, -RP or -Rall at actual receiving product coordinates before the public down projection. Continue blocks25-28 and final learned normalization/head finitely. For each removal run a SHA-seeded product-norm-matched random edit. Include sham and remove/reinsert-all controls: ten suffixes per case. Prefix/source installation is exact; later prefix replay bound is1e-4. Keep every signed vector and every case.

Use the same W and final interaction-vector definitions and frozen joint reduction criterion as226b: original |W|>.10; |Wremove|<=.5|W|; final interaction norm<=.75 of original; both better than random. A stable descriptive component requires12/16 and6/8 per field order. Record the raw whole-Rall bridge to226b's MLP24 write removal at the unchanged1e-4 final-coordinate bound, including all failures. Require the .005 two-score bridge; do not infer coordinate equivalence when the raw bound fails. The present claim concerns these directly executed product interventions, not identical reproduction of the post-addition intervention. No outcome-directed scaling or further correction sweep.

Reference blocks must replay saved trajectories at1e-4; finite two-logit heads independently replay in NumPy64 at .005. Save actual MLP inputs, products, writes and post-addition states. Independently account for input-cast, down-projection and residual-addition arithmetic when comparing the mixed write to down(Rall). Report raw1e-4 comparisons separately; arithmetic closure at1e-9 does not convert a raw failure into a pass. Reconstruct all product terms, seeded controls and installed arrays independently. No whole-block derivative qualification is claimed.

Expected work:128 reference blocks,96 additional local MLP calls,320 direct down projections,1280 continuation blocks,320 finite heads. Model revision/device/dtype/eager setting remain pinned as in225b. Source/protocol/input hashes and copies precede execution and are verified afterward. The NumPy auditor is a separate code path in this investigation, not an independent research team. No browser, publication, staging, commit or push.

## Preserved 227 failure and narrower 227b scope

227 completed its first case, then stopped on the second case's raw product-versus-post-addition bridge:1.373291015625e-4 exceeds1e-4. Its code, source seal, first case, traceback and failure remain. Those interventions enter at different arithmetic nodes, so their final-coordinate equality cannot be assumed. 227b retains that exact comparison and bound as a reported raw failure rather than a prerequisite for measuring its own product interventions. The score bridge remains a hard .005 check. Prompts, product recipes, random seed suffix227, action gates and all other tolerances are unchanged. An arithmetic ledger independently checks local product/write accounting; it does not certify final-coordinate equivalence or explain away a raw failure. This narrower scope is explicit before the new run.
