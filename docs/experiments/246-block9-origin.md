# 246: generation versus inherited interaction at the first macroscopic jump

Retrospective follow-up, selected AFTER 244/245: median regional receiving
interaction norm grows .02495 -> 1.02339 at block9 (one-indexed). This is an
observed selection, not a fresh prediction. No layer sweep or learned-circuit
claim. Freeze this protocol and runner/scorer before new interventions.

Replay all four 244 policies, capturing block9 input, attention write, MLP
input, gate/up/SiLU/product/write. For attention, form full input X*=XA+XB-X0,
cast Float32 and evaluate its finite write. Decompose I_attention as
P=F(XC)-F(X*) and G=F(X*)-F(XA)-F(XB)+F(X0).
P transmits a mixed input in this specified allocation; G is new finite mixing.

For receiving MLP products, use public finite RMS/projections/SiLU/product map M.
P=MC-M*, G=M*-MA-MB+M0, D=(MA-MQA)-(M0-MQ0), where MQA/MQ0 are actual quadratic
receiver products in FQ/QQ. Thus actual mixed product equals P+G+D exactly.
D isolates sensitivity of the finite-minus-quadratic SiLU discrepancy to the
source-induced receiving-state change. With R=finite_phi-quadratic_phi,
D_gate=(RA-R0)u0, D_up=R0(uA-u0), D_mix=(RA-R0)(uA-u0).
Keep D_round=D-D_gate-D_up-D_mix explicitly (Float32 multiplication rounding).
Also retain residual-addition rounding in the whole block identity.

Primary hypothesis: D is sufficient for the local receiving-product interaction
(<=10% relative error/.99 cosine, target norm>=.001 at whole and verdict scopes),
and its direct removal from otherwise native FF's MLP9 product lowers the FINAL
regional interaction norm >=25% at both scopes, reduces its absolute margin
interaction >=.10 nats (baseline >=.10), and beats 3 per-row norm-matched random
D removals at both scopes. Require24/32 and6/8 per stratum. Secondary removals:
P, G, all local mixed product, D_gate, D_up, D_mix. All10 arms reported. These
are oracle primitive interventions, not a False-only predictor or improvement
sweep. Report native displacement and margin errors too; reduced policy
interaction is not native accuracy.

Require bitexact regional replays and causal prefixes, actual installations,
untouched source/prefix product rows. Independent CPU64 finite MLP RMS/SiLU
primals and product/down bridges <=1e-4; final head .005. Keep explicit rounding
terms, never infer mechanism from closure alone. No whole-block derivatives are
used or recertified. Q-SiLU retains 244/245's actual-direction AD/FD qualification.
Preserve every failure, stop on unexpected qualification errors, do not relax
bounds. Seal all arrays before separate scoring. Count128 regional replays,
32 local attention calls,96 local MLP calls and320 primitive intervention
forwards, plus any qualification calls. Contexts/checkpoint unchanged.

This tests the origin of a native-attention regional POLICY interaction, not the
entire native Boolean response. Even a pass does not solve Question B.
