# 247: native context/verdict attention factorial

[Report](../../docs/analysis/247-native-verdict-context-result.md) and
[protocol](../../docs/experiments/247-native-verdict-context.md).
All 32 cases, finite primitive components, 320 removals, controls, numerical
qualification, source seals and runtime are retained. Native False/True states
replay exactly. The coupling-only local/causal primary fails 0/32. This uses
oracle incoming states and is not a False-only forecast or learned circuit.

Raw arrays stay local, inventoried with sizes/hashes. Regenerate from public
inputs with research/boolean_coupling/reproduce_native.py. Existing qualified
242b outputs can be reused. Every output must be new; failures stop and remain.
Pinned Mac/MPS setup is in research/README.md. Model-free verification checks
integrity, not numerical replication or independent confirmation.
