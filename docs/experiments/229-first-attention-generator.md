# 229: what generates the first attention discrepancy?

Registered before this diagnostic's model calls, 18 October 2026. Retrospective:
all 32 contexts and native targets were observed in 228. No fresh-context or
prospective claim is available. The failed 228 hypothesis and all its numerical
limits remain unchanged. No parameter fitting, site search or end-to-end sweep.

## The single boundary

Use the exact native full sequence after block 1, before attention 2. Reconstruct
it from a replayed ordinary False computation and the sealed known-source and
receiving rows from 228. Verify these rows against the saved native observation.
Only rows strictly after the Boolean source are intervened on. Let x be False,
s the source-row edit and d the exact receiving-row edit. A includes attention
and its residual addition. J is precisely the newly declared smooth CPU64
receiving derivative from 228, with False prefix fixed, not the old disputed
Float32 whole-block Jacobian.

Four finite public corners give the signed decomposition

```
E = A(x+s+d) - A(x+s) - Jd = R + C
R = A(x+d) - A(x) - Jd
C = A(x+s+d) - A(x+s) - A(x+d) + A(x).
```

Report every coordinate and the receiving/verdict-position norms, signed
projections onto E (dot(component,E)/dot(E,E)), and vector-removal errors. Signed
shares can be negative or exceed one. No absolute-value attribution shares.
Input RMSNorm is row-local, so it cannot itself create an s/d mixed normalized
input on disjoint rows. It can contribute to R; we do not further partition R.

## Primitive partition of C

Use a declared smooth CPU64 attention graph with promoted actual parameters,
mask and RoPE, and independently reconstruct the finite public MPS graph.
For receiving queries, obtain Q, K, V at all four corners. Write P0, Ps, Pd, Pb
for softmax probabilities and Vb = V0 + deltaVs + deltaVd. Let

```
Zadd = Zs + Zd - Z0                   # UNMASKED logits
Padd = softmax(Zadd + causal_mask)
L = deltaQd @ deltaKs.T * scaling
Zb - Zadd = L

C_QK = (Pb - Padd) @ Vb
C_SM = (Padd - Ps - Pd + P0) @ Vb
C_AVs = (Pd - P0) @ deltaVs
C_AVd = (Ps - P0) @ deltaVd.
```

After the linear output projection these four vectors sum exactly to C in real
arithmetic. This is a fixed algebraic allocation, not a unique causal mediation
decomposition. In particular QK's effect includes the subsequent finite softmax.
There is no RMSNorm cross term on disjoint rows. Report the public-versus-smooth
rounding remainder rather than allocating it to a learned mechanism.

Direct primitive interventions start at the native both-corner graph: remove L
from its logits; remove the defined softmax mixed term from its probabilities;
or remove each AV mixed product from the weighted context. Run the actual output
projection. The softmax edit is an artificial internal intervention and can
produce signed weights; it is not another naturally occurring attention pattern.
Compare each actual signed write effect to its separately computed CPU64 vector.

## Does it matter downstream?

Use the native source/prefix for every continuation. Starting from A(x+s)+Jd,
add R only, C only, both, or one deterministic Gaussian vector matched to each
candidate's receiving-space L2 norm. Separately continue the four primitive
deletions from native attention and one matched random-write deletion per
primitive. Include the unmodified native replay. All remaining MLPs, blocks and
the final normalization/readout run finitely. Save installed mids, final
receiving vectors and both candidate scores for every arm (15 suffixes/case).

This measures conditional downstream sensitivity to an isolated early error.
It does NOT test repairs of 228's remaining linear transport, forecast new
targets, explain later errors, or establish a compact False-only predictor.
R+C recovery is a reconstruction control, not a predictive success.

## Frozen qualification and decision rules

- Existing array hashes and 228 observation/forecast record checks must pass.
- Replay False boundary/mid arrays, source rows and native both midpoint within
  1e-4; prefix protection exact. Public manual attention corners within 1e-4.
- Public versus smooth attention corner/write and primitive-effect comparisons
  within 1e-4. Keep any failures; do not widen tolerances or claim attribution
  for an unqualified comparison.
- Recompute the used Jd and independent sine/cosine directions in CPU64; central
  difference h=1e-5, coordinate error <=1e-4. Original saved Jd comparison <=1e-9.
  This qualifies the declared smooth map only.
- CPU64 vector closure and disjoint-row/logit identities <=1e-9; mixed-probability
  row sums <=1e-9. Public primitive ledger <=1e-4, with arithmetic remainder saved.
- Native and R+C final replay <=1e-4 per residual coordinate, <=0.005 per score.
  The same bounds apply to a finite NumPy final-head reconstruction.
- Local materiality guard: verdict E norm >=0.001. A component accounts locally
  for the discrepancy only if deleting it leaves <=25% of E's norm, in >=24/32
  contexts and >=6/8 in each schema/order stratum. Report all other components.
- A downstream R or C repair is supported only if it reduces BOTH final
  verdict-residual error norm and absolute margin error by at least 50% from
  baseline, and beats its matched random control on BOTH, with the same case
  and stratum counts. Baseline guards: vector error >=0.001, margin error >=0.10.
  Ineligible cases are reported, never reclassified as successes or failures.
- Primitive deletion materiality is reported as signed residual/margin changes
  versus native, with matched random controls; no sign, circuit, neuron-feature,
  novelty, generality or Question-B-solved inference is licensed by this alone.

Expected main work: 32 ordinary False replays, zero ordinary True model calls,
128 public attention corners, 128 manual attention corners, 128 primitive
attention interventions, 96 CPU64 JVP/central-difference comparisons, 480 MLP2
continuations, 12,480 subsequent public blocks, 480 finite heads. CPU64 finite
corner and derivative endpoint evaluations are recorded separately. No training,
paid API calls, browser or literature search. Raw arrays stay local; publish
compact records, all cases, source seals, qualification, figures and reproduction.
