# 250 — continued native source communication

Frozen retrospective diagnostic on all 32 existing 242b contexts. This follows
247–249's native-response results, not the approximation-policy target of 243–246b.
Question B remains open; this is not a prospective forecast or a learned circuit claim.

## Intervention

Keep the full native True computation through block 9. In blocks 10–28,
replace the changed source token's rotary-applied key and/or value with its
ordinary native False counterpart, only when queries strictly after that token
read it. The source and earlier rows retain their native True computation.
All receiver rows evolve from their own intervened history; final RMSNorm and
the full vocabulary head remain finite. No derivatives are used.

Arms: F ordinary False; T ordinary True; Z zero clamp (same full-shape attention
recomputation, current K/V); K False source key; V False source value; B both;
D both for verdict queries only; L both for intermediate receiving queries only;
random0/1/2 both source K/V changed in deterministic random directions, matched
separately to the actual K and V displacement norms at each layer. Random seeds
are SHA256(case key + arm + layer + K/V). These controls match norms, not
on-manifold validity. K includes normalized key changes and their softmax effect;
it does not isolate softmax from upstream key generation.

H is a separate oracle suffix control: native True pre-block-10 full state with
only the source row replaced by native False; ordinary finite blocks 10–28.
B and H must match on all receiving rows and scores at every suffix block.
Their source rows differ deliberately. This identity checks that source
communication has really been removed while receiver history is preserved.

## Outcomes and frozen hypotheses

For every arm retain the signed final displacement from F, its projection onto
the native T−F vector, relative norm, reconstruction error, cosine, and signed
true-minus-false margin change. Also retain every suffix receiving trajectory.
Report all 32 contexts, the four dependent strata and matched random controls.

Primary test of continued-communication necessity: B removes at least 90% of
the signed native verdict-vector projection AND absolute margin response in
at least 24/32 cases and at least 6/8 in every stratum. Negative projections
are allowed; require |remaining projection| <= .10 and |remaining margin
change/native margin change| <= .10; native guards vector norm >= .001,
absolute margin change >= .02. This is necessity under an artificial clamp,
not sufficiency or a prospective Question B answer.

Secondary crossed finite contrasts: T−K, T−V, and K+V−T−B sum to T−B.
Direct/relay contrast: T−D, T−L, and D+L−T−B sum to T−B. Save the raw
second finite differences (T−K−V+B and T−D−L+B) as well; their signs are
opposite to these interaction allocations of the removed effect. Each is an allocation
among these defined interventions, not an additive path mechanism; receiver
states differ across arms. Do not equate separately pooled medians to shares.
D tests direct source reads at the verdict; L tests reads into intermediate
rows which can relay to the verdict. No layer/head/neuron selection or new
end-to-end correction sweep. A later prospective experiment requires a new freeze.

## Numerical qualification and storage

All F/T suffix coordinates/scores must replay old 242b exactly; Z must replay
T exactly. All arms' prefix/source rows must match their native trajectory;
all pre-block-10 receiving states are unchanged. H's source must remain native
False, B's native True. All B/H receiving coordinates and head scores must
match exactly. Installed K/V and preserved query outputs are checked exactly.
Recompute QK, softmax and PV at full original query shape, then replace only
the selected query outputs. Dropout is zero; eager Float32 MPS is fixed.

Separate NumPy Float64 audit checks the finite head for every endpoint (.005)
and QK/softmax/PV (1e−4) from saved witnesses at blocks 10, 19, 28 on the
first case of each fixed stratum, all 11 forward arms. This is a bounded
primitive witness audit, not an independent reconstruction of every forward.
Record this coverage explicitly. Preserve full outcome trajectories locally;
publish case hashes, checks, CSVs, source snapshots and source-bound seals.
Existing evidence is never overwritten. Failure is retained, with any correction
separately named and frozen. Disk space must exceed 256 MiB before each case save.

Budget: 352 full forwards (11 × 32), 608 additional H suffix blocks and 32
full-vocabulary H heads. Each of 9 modified arms also recomputes 19 QK,
softmax and PV kernels per case: 5,472 each. These are extra work; no speedup.
CPU64 witness audit and endpoint scoring are additional diagnostic work.
