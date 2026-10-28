# 290: receiving normalization versus gated MLP response

Freeze before checkpoint execution. Retrospective follow-up to289's failed
attention-interface sufficiency account on the same32oldcontexts. No newTrue
measurement or gradient, no source/circuit/efficiency/generalQuestionB closure.

Keep source recurrence and head E, as before. Keep ALL289 attention-interface
factors E: source inputnorm for K/V, QK/softmax and probability-value mixing.
Arm00 must exactly replay289's111 baseline. Change two remaining families:

- N: receiving inputRMSNorm and post-attentionRMSNorm responses, False or E.
- M: native gatedMLP response (linear gate/up, SiLU, gate-up product, learned
  down projection), False or E. This groups SiLU and product; it does not isolate
  either operation or identify a learned feature.

Each normalization is a finite map `normF + norm(E_parent+offset)-normE`.
For M, use `writeF + MLP(E_normalized_input+normalized_offset)-writeE`.
When off, the actual native False-parent function remains in the receiving
branch. Source inputnorm for K/V is already E under289's interface; N additionally
updates receiving rows. Source recurrence stays E for every arm. Final head E.
All arms preserve the SAME E zero-displacement trajectory. These are actual
finite functions with finite-difference tests, not backward-only patches.

N1M1 should reproduce the unmodified E gradient and finite correction. It is a
calibration by construction, not the scientific discovery. The two sufficient
accounts are N-only10 and M-only01. Both are primary and must pass>=24/32 AND
>=6/8everylayout. Target is the REMAINING gradient/finite-effect difference between
unmodified E and289111. Each account is compared by norm>=.001, relative<=.25,
cosine>=.95 and signed finite error<=.005+.25absD. Separate .02-nat materiality
retains all cases. Report all00/01/10/11, finite receiving correction vectors,
and old native scalar calibration with ORIGINAL M margin baseline. No complete
native residual forecast is implied. E remains source-onlyMF, not fullMM orTrue.

Numerics: source84oldcodebytes unchanged; new sources frozen before checkpoint.
Matched00 replays289111 boundarygradient<=1e-4, finitecoordinates<=.002 and scores
<=.001. Matched11 replays freshEgradient<=1e-4, nativefinitecoordinates<=.002 and
scores<=.001. Report narrower1e-4 finite-coordinate failures separately. Every
zerooffset is exactlyzero with E head. Full-tape/blockwise source boundary<=1e-4.
Later trace differences remain diagnostics. Actual finite functions have FDs along
unitd plus two seeded coordinates at .05/.025, error<=.001+.02absAD and halving
<=.002+.03absAD. CPUFloat64 head<=.005. Receiving input/postRMSNorm VJPs at the
verdict row are independently checked at both anchors by saved-parent Float64
formulas, all coordinates for one fixed cotangent, at1e-4. The gatedMLP family's
composed derivatives are qualified by modified-function FDs and two reverse
compositions, not independent primitive AD of every SiLU/product coordinate.
Do not use this family test to claim one such primitive is the mechanism.

Any required failure saves numerical records/failed audit and withholds scientific
tables. No automatic posthoc exceptions or frozen-bound change. CPU preflight has
four unequal-length cases, all four arms, all256source-coordinate FD checks,
and coherent altered-bound/FD/norm-probe rejection. Tiny scientific outcomes do
not need to succeed. Original289/288/287 failures and scopes remain unchanged.

Initial/per-case reserve384/256MiB and total case archives96MiB. Store boundary
gradients, actual finite head/offset vectors, trace differences and receiver norm
parents/adjoints. Count all anchor/model/modified/reference/head/reverse and
additional norm evaluations. No speedup. A positive result is a conditional
family-level explanation of this remaining sensitivity discrepancy. A negative
result can require the two families jointly without identifying a learned circuit.
