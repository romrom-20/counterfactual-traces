# 282b: narrower numerical qualification of the public Float32 up map

Original282 producer, protocol, source freeze and every output remain immutable.
The completed cold retry reproduced all23 sealed cases from the interrupted run
byte-for-byte. Original audit282 then failed CPU64_U_up at .00010400542640809363
against the frozen .0001 absolute bound. That audit remains FAILED.

A separately recorded numerical-only diagnostic examines all128 local corners,
256 gate/up comparisons, without inspecting scientific outcome counts. It finds
28 chained up-map failures, maximum .00021562054442370027. Consuming the actual
Float32 normalized input does not fix them:27 up comparisons still fail, maximum
.00020469122384270122. The gate comparisons pass. Normalization error alone is
not an explanation. Retain these signed errors and all original bounds.

## Changed scope, unchanged predictions

Freeze this qualification and its audit BEFORE the qualified audit. No new model
forecasts or behavioral criteria are changed. Exact public Float32 full-shape
norm/gate/up/product/down/head replays remain required, as do the original CPU64
normalization, gate, SiLU, polynomial and derivative-coefficient bounds. Only the
CPU64 **up-projection** absolute bridge becomes a retained diagnostic rather than
a required pass. This CHANGES the numerical validation scope. It does not pass
or reconcile the original up-map checks, and is not independent high-precision
validation of that projection.

Additionally decompose each up error at the actual consumed normalized input:

`observed32 - W*z64 - b = W*(z32-z64) + (observed32-W*z32-b)`.

Require signed closure <=1e-10. As a numerical consistency check, require every
coordinate to lie within a dimension-derived Float32 rounding envelope:

`abs(W)*abs(z32-z64) + gamma_(2d+3)*(abs(W)*abs(z32)+abs(b))`,

with unit roundoff u=2^-24 and gamma_k=k*u/(1-k*u). This is the standard conservative
dot-product accumulation bound with additional bias/output rounding allowance,
under ordinary Float32 rounding assumptions. It is not a new .0001 accuracy
claim or proof of the GPU's internal reduction algorithm. Record the envelope,
ratio, original errors, isolated projection errors and signed closure for every
corner. The envelope can be much looser than .0001; exact public execution is
the primary scope of this qualification. No failed raw comparison is relabeled.

Every original CPU64 product/down diagnostic is retained too. All saved inputs,
direct product installations, prefix identities, zero/full/archive controls,
matched random directions, fixed finite readouts, signed-vector ledgers and
frozen endpoint decisions retain their original requirements. No whole-block
Jacobian or older failed raw check is recertified.

## Execution gate

Before freezing, rerun the complete separate audit on the existing32-case tiny
CPU producer fixture, using a new directory with hardlinked immutable arrays.
Mark its mocked prior failure SYNTHETIC, and verify that the qualifier rejects
a failure at a different primitive. All endpoint counts must match the original
CPU audit. The actual original CPU producer and audit passed before282 freeze;
this test adds no new real-checkpoint forecast and is not fresh support.

For actual use, the qualifier binds completed design/status, the original audit
failure, numerical-only diagnostic and all new source hashes before the rerun.
If any other required check fails, stop, retain it and do not silently edit this
qualification. A successful282b audit supports only the declared Float32
intervention outcomes. It is not an independent research team, learned circuit,
efficient/fresh predictor, priority claim or complete answer to Question B.
