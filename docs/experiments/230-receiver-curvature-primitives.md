# 230: which primitive generates receiver-only curvature?

Registered before new calculations on 18 October 2026. Retrospective on all 32
observed 229/228 contexts. The incoming full sequence before attention 2 remains
exact. Source/prefix stay False; only receiving rows carry their native edit d.
No ordinary model computations or new native True measurements are required.

## Hypotheses and fixed test

The receiver remainder R from 229 has not been explained by a primitive. Test
four operations: input RMSNorm (N), QK multiplication (K), softmax (S), and the
attention probability–value product (V). Each is finite or replaced by its
False-anchored affine map, receiving actual upstream node outputs. Run all 16
settings in smooth CPU64 and the reconstructed public MPS Float32 graph. QKV
and output projections, RoPE, masking and residual addition remain explicit.
Actual MPS affine-node replacements are calculated in CPU64, anchored to each
recorded public False node, then cast back to Float32. This is a declared graph,
not a use or recertification of the disputed original Float32 block Jacobian.

All-zero smooth graph must equal A(False)+Jd; all-finite graph must reproduce
the saved receiver-only finite corner. This supplies a qualified signed R.
Use the fixed telescoping allocation (bit order N,K,S,V):

```
N = Y1111 - Y0111
K = Y0111 - Y0011
S = Y0011 - Y0001
V = Y0001 - Y0000
R = N + K + S + V.
```

This allocation is order-defined. Also report all 32 signed factorial edges
per context so a dominant term under one allocation is not called a unique
mechanism. No learned features, circuits or novelty priority are inferred.

## Predefined position test

With K/S/V affine, make RMSNorm finite separately at the first receiving row,
the last (verdict) row, and the intervening rows. These fixed groups partition
all receiving rows; no data-dependent neuron/row selection. Because the rest
of this graph is affine in normalized inputs, their signed effects must sum
to Y1000-Y0000 in smooth arithmetic. Report every row's norm change and the
relative input-edit norm. Split the first-row normalization correction's
False-linear downstream attention response into query, key and value paths.
At the verdict its query part must vanish since the edit is at a different row.

## Frozen local decision and numerical qualification

- Main native remainder guard: verdict norm >=0.001.
- A primitive's finite-only graph accounts locally if its signed correction
  approximates R with relative L2 <=0.25 and cosine >=0.95.
- Necessity in the finite background is separately tested: removing it leaves
  remainder norm <=0.25 of R. Both tests need >=24/32 and >=6/8 in every
  schema/order stratum. Neither is evidence about a learned circuit.
- First-row-only normalization uses the same finite-only accuracy rule; its
  necessity is only relative to the all-normalization/otherwise-affine graph.
- Report all verdict and all-receiving signed vectors, projections, errors,
  factorial edges and preserved failures. No threshold adjustment after results.
- Raw input/source/weight and array hashes must match 229. Public all-finite
  graph must match the public attention module/saved receiver corner at 1e-4.
- All smooth/public graph endpoints and signed increments compare at 1e-4;
  keep measured arithmetic remainders. Smooth all-affine/old Jd and algebraic
  closure compare at 1e-9. All protected-prefix comparisons must be exact.
- Every distinct used smooth affine-node direction is checked against CPU64
  forward AD and independent central finite differences (h=1e-5). Analytic/AD
  <=1e-9; FD <=1e-4. These are local derivatives of the declared smooth graph.
- A separate NumPy implementation reconstructs all smooth factorial and row
  graphs, normalization corrections, path splits, decisions and edges. No
  passing arithmetic ledger erases a failed raw comparison.

Expected finite work: 64 public attention-module replays, 32 public and 32
smooth False graph anchors, 608 public plus 608 smooth intervention graphs
(16 factorial +3 fixed row groups per case). Derivative/FD counts are recorded
after exact-direction deduplication, approximately 24 comparisons per case.
No suffix/head sweep: 229 already showed small isolated finite-suffix margin
effects. This experiment does not test propagation under 228's remaining linear
transport and cannot explain its full forecast failure by local success alone.

If a specific local account passes, a separate prospective local forecast may
be frozen and tested on fresh inputs. That rule must declare the known-edit
computation budget and seal predictions before native target observations.
