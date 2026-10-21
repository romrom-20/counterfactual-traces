# 258b — qualify the recorded Float32 down projection at its actual shape

258's original pre-audit FAILS its absolute1e-4 whole-MLP Float64 projection
bridge. The failure is preserved, not relabeled passed. Across all72 forecast
witnesses,24 fail; maximum .000735865. It occurs in ordinary False and zero
controls too. No ordinary native True run has occurred.

A separately source-bound primitive replay reconstructs the public learned
down projection from saved product and public Float32 weights. At the ORIGINAL
full-sequence shape, all72 outputs replay EXACTLY. At the sliced receiving-row
shape, even Float32 differs by up to .000747681. Therefore the recorded public
Float32 path is shape-sensitive at these absolute tolerances. This does not
make the Float64 bridge pass or recertify earlier derivatives.

Before any native True measurement, freeze this narrower qualification:

1. Every original check stays unchanged except the failed whole-primal bridge.
2. Record EVERY original CPU64 whole-write error at its ORIGINAL1e-4 bound,
   including failures in raw-down-checks-*.csv.
3. Independently reconstruct the public Float32 down projection at its actual
   full-sequence shape from saved products and checkpoint weights; require
   exact0 agreement, including post-seal native controls.
4. Keep ALL separate CPU64 signed restoration/remainder/gap down-vector bridges
   at the original1e-4 bound. Keep SiLU CPU64 AD/FD, mask reconstruction, exact
   installed products, cubic-query checks, finite heads, source/prefix/zero/native
   controls and all behavioral criteria unchanged.

This qualifies operation-defined forecasts and signed component calculations
under the declared Float32 computation. It is NOT full independent whole-model
real-arithmetic reconstruction. Raw-primal numerical failures and old whole-block
derivative disputes remain unresolved. Do not claim the original numerical
protocol passed unchanged or identify a learned circuit.

Original forecast arrays, source hashes and outcome criteria are unchanged;
no arm is selected after observing native outcomes. A qualification source seal
binds this protocol, auditor and replay to the original global forecast status.
Pre-audit, global forecast seal and public remote receipts must precede native
True measurements. Extra primitive replays are counted separately from model
forwards; this revision supplies no speedup claim.
