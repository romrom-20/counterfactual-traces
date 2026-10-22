# 263 — causal gate/up branches and the residual-energy rescaling hypothesis

Retrospective intervention on the SAME four old261b contexts/all28 layers.
262's primary baseline-only curvature account fails its90% rule; its secondary
energy term gives accurate whole gate predictions locally. That motivates a
NEW intervention hypothesis, not promotion of262 to a fresh/global success.

Let x0,d,s,a,b,rho have262's definitions and f0=gamma*x0/s. Use actual False
normalized output z0. Linear L=z0+J0*d. Define P=(rho-1+a)*f0,
D=(rho-1)*gamma*d/s, E=-(b/2)*f0. For each receiving MLP's own incoming state,
intervene on the actual normalized inputs to gate_proj and up_proj separately.
Unchanged finite source/prefix, cubic receiver queries, ranked1% SiLU, finite
K/V/softmax/projections/products/readout. No observed native input is supplied.

Arms, all frozen and all reported:

- F ordinary False; Z zero edit with linear inputs to both branches.
- B finite normalized inputs to BOTH branches: old258-S reference.
- L linear inputs to BOTH branches: exact old261b-L reference.
- G finite gate input, linear up input.
- U linear gate input, finite up input.
- P linear+P to both branches (existing-pattern curvature only).
- D linear+D to both (projected-direction curvature only).
- E PRIMARY linear+E to both (squared-displacement-energy rescaling).
- H linear+P+D to both (complete smooth factorization, actual False anchor).
- T ordinary native True.

F/Z/B/L/T must replay original saved trajectories/scores EXACTLY; source and
earlier prefix remain unchanged by receiver interventions. H need not be exact
public Float32 normalization: require H-B difference <=.001 of the native
displacement at whole AND verdict, and margin difference<=.005nats, as a separate
rounding-control bound fixed now. If it fails, qualify that limitation before
interpreting P/D as a complete public normalization allocation.

E sufficiency requires4/4 original joint native-response passes: whole AND
verdict relative error<=.05, cosine>=.99, native norm>=.001, margin error<=.10.
Report B/L/G/U/P/D/H by identical criteria without substituting them for E.
G/U form a separately reported2x2 normalization-input factorial with B/L. Save
the full interaction B-G-U+L and finite margin interaction; no unique additive
causal percentages or learned features. Report all arms, guard failures included.

Numerical scope is declared BEFORE new model calls: retain original CPU64
gate/up whole-primal errors at1e-4 as raw passes/failures; require exact matched-
full-shape public Float32 gate/up replays. Keep public norm bridges1e-4, smooth
RMS AD1e-9, FD steps1e-3/1e-4 at1e-3/1e-5 bounds, independent polynomial jets
1e-10, exact installed query/branch inputs, selected SiLU/product nodes and
finite head .005. Failed old261b/c absolute displacement bridges remain failed;
no new real-arithmetic displacement equivalence or old derivative certification.

Save full receiving raw/current normalization, separately installed gate/up
inputs, gates/up/SiLU/products/writes/masks and every layer tail/score. Source,
checkpoint, original references and all records are hash bound. All output writes
stay in the new directory; never overwrite input evidence. Nonfinite or>100x
False tail norm guards stop that arm and count as failure. Other failures stop
the experiment. Expected44 full forwards; audits/projections additional.

This tests a specific generation primitive and its behavioral consequence on
old examples. Even success would not establish a fresh forecast, cheap model,
learned circuit or complete Question B answer. These are ordinary normalization
components whose measured importance, not algebraic existence, is under test.
