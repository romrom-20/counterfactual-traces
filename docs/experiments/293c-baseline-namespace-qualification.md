# 293c: restore the False baseline without changing any intervention

The original293/293b producer used F both for the observed False baseline and
for the fixed-response-range condition. Candidate F overwrote the baseline
fields in the saved archive. The original native/paired science tables and
primary decisions are INVALID. Preserve those files and their53,832 passing
primitive records; they did not catch this namespace collision.

This is a separately frozen, retrospective, read-only correction. Recover the
actual baseline from292's immutable forecast archive, verify its original
archive/metadata/design hashes, and name it B0. Preserve all five candidate
arrays exactly, including condition F. Preserve native targets and verify them
against292 native archives. No candidate trajectory, basis, threshold, rank,
seed, operator, checkpoint observation or intervention changes. No model calls.

Recompute the original native and paired criteria with B0 in a separate output
folder; retain original tables under an explicit INVALID label. Use independent
Float64 vector statistics and actual candidate/native/B0 score differences.
Publish baseline fingerprints and native/paired vector sufficient statistics.
The old primitive qualification applies to unchanged calculations only.
Add exact baseline/native/reference bindings and eight CPU64 baseline-head
checks at the original .005 bound. Check the zero-edit recorded scores against
B0; the zero path must not silently use candidate F. Pair order is descriptive
minus generic, as in the original audit's sorted schema order.

Before production corrected scoring, a tiny CPU fixture must preserve distinct
B0 and F, verify all restored B0 values, and reject a coherently corrupted B0
against the independent original False computation. Scientific tiny-model
performance is not required. Keep all original scientific thresholds and
qualify corrected outcomes only after these bindings pass. No fresh-prediction,
learned-circuit, rank lower bound or general Question B solution follows.
