# Complete vectors before MLP16: experiment 241

The predeclared greedy four-site input-error account passes **0/32** old cases. Its first sufficient prefix uses **16 of 29 operations**, passing 32/32. Complete transported native-remainder vectors close the missing input error within 2.78e-13. This is a retrospective oracle-vector account, not native-state prediction or proof that all compact mechanisms fail.

New errors partly oppose inherited error. Posthoc metric accounting shows that transport reduces individual-vector energy while weakening mutual cancellation; the two larger changes leave a small net. These identities do not identify learned mechanisms or intentional self-repair.

[Full report](../../docs/analysis/241-before-mlp16-vectors-result.md) · [Frozen protocol](../../docs/experiments/241-before-mlp16-vectors.md)

## Evidence

- `sources/`, `design.json`: exact frozen code/protocol and input/checkpoint bindings; source freeze `7f335d2`.
- `cases/*.json`, `status.json`: all 32 cases, complete directional counts, numerical checks and FD endpoint hashes.
- `audit.json`: 3,872 separate numerical checks, all passed; original runner adds 6,592 checks.
- `selection.json`: the full retrospective shared greedy order, every prefix decision and all strata; four-site failure retained.
- `energy.csv`: every generation, transport and cross term at all 29 operations.
- `transported.csv`: full-vector norms/gains and signed projections for each native local remainder.
- `reconstruction.csv`: every case and all 29 prefixes, both whole and verdict scopes.
- `profile-summary.json`: explicitly posthoc energy arithmetic audit (3,712 checks), no new model calls.
- `metric-description.{csv,json}`: explicitly posthoc terminal Gram comparison (32 identity checks).
- `transport-metric.{csv,json}`: explicitly posthoc split of transport into individual and pairwise effects (992 identity checks).
- `logs/`: actual execution output of all five stages.
- `local-array-inventory.json`: every large local NPZ by hash/size. Arrays are excluded from Git. They retain all used JVP vectors, complete terminal components/Gram matrices, targets and the installation term. Intermediate directions/finite differences are reconstructible from the frozen map/order; hashes and FD errors are retained.

All 13,920 actual tangent directions pass Torch FD and explicit NumPy derivative checks. Independently calculated NumPy finite differences also pass. Both finite-endpoint sets contain27,840 evaluations. No new ordinary False or True forward. Old native remainders are supplied oracle diagnostics; this is component work, not zero computation or a speedup.

## Reproduction

Use the [recorded environment](../../research/README.md). 241 itself requires CPU64/checkpoint arithmetic; rebuilding its older prerequisites requires Apple Silicon/MPS.

```sh
python -S research/boolean_coupling/reproduce_early.py --verify-only --history --plan
python research/boolean_coupling/reproduce_early.py --run --offline \
  --input-238 runs/existing-238 --input-240 runs/existing-240 --out runs/my-vector-test
```

Omit both input options to regenerate the fixed prerequisite chain. Output directories must be new. The241 protocol binds the archived input-array hashes; any different regenerated arrays stop rather than silently becoming a new dataset. Cross-environment tolerance-based substitution is not part of this fixed-array release. Integrity verification checks archived records and hashes without regenerating the omitted arrays; it is not numerical replication.

The primary analysis is frozen; the three metric descriptions are explicitly posthoc. No primitive causal intervention, learned-circuit claim, fresh test or successful final-response forecast follows from this release. All earlier failures remain preserved. Article drafts stay off GitHub.
