# Source/receiver local normalization diagnostic 257

The hypothesis that the first Boolean source step leaves the RMS Taylor radius
is rejected: radius 1.06607, versus minimum receiver radius 2.12861. Nevertheless
the cubic source-embedding normalization update has 31.77% error, and degree 12
still has 8.54%. Convergence does not establish low-order accuracy.

- `source-receiver-errors.csv`: all native case/layer/source-or-receiver outcomes,
  Taylor degrees 1/2/3/4/6/8/12, including zero updates.
- `audit.json`: input/checkpoint/source seals, all coefficient and state-boundary
  checks, and the explicitly rejected hypothesis.
- `sources/`, `logs/`: frozen diagnostic/protocol and original execution log.
- `manifest.json`: complete publication integrity hashes.

This is retrospective, with zero model calls. The first source embedding pair
is unique, not 32 independent facts. The field `outside_rows` counts eligible
CSV case/layer/scope rows, using their maximum token b; it is zero in both scopes.
No causal explanation of old failed policies, prospective forecast or learned
circuit is established. Raw arrays and checkpoint weights remain local.

[Report](../../docs/analysis/257-source-receiver-radius-result.md)
· [Reproduce](../../research/README.md).
