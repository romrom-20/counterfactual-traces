# 249: sealed source-only regeneration and oracle controls

[Report](../../docs/analysis/249-source-regeneration-result.md) and
[protocol](../../docs/experiments/249-source-regeneration.md).
The 32 source-only predictions are globally sealed before oracle-control reads.
Native primary and narrower context-recovery secondary both fail 0/32. This is
a retrospective test with substantial dense component work, not a speedup or
learned circuit. No True receiving state enters the source-only algorithm.

forecasts/ and controls/ retain every metadata record; raw arrays remain local
with sizes/hashes in local-array-inventory.json. Forecast/control timestamps and
source seals are explicit. Regenerate using reproduce_native.py; matching 242b,
247 and 248 prerequisites can be reused. Every output must be new. Unexpected
failures stop. See research/README.md for the pinned Mac/MPS environment.
Release verification is model-free and does not rerun omitted arrays.
