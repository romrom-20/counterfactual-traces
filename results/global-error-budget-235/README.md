# 235: complete global forecast-error diagnostic

The full 47-write ledger plus input cast closes exactly. Its four largest signed
contributors are MLP23/22/21/20. Finite recomputation shows that MLP22/23's
approximately 25-unit write errors come almost entirely from inaccurate inputs;
those operations were already finite. A terminal four-write oracle repair
reduces median margin error 6.588→1.668 nats but passes the joint rule only 2/32.

[Report](../../docs/analysis/235-global-error-result.md)
· [Frozen protocol](../../docs/experiments/235-global-forecast-error-budget.md)
· [Actual upstream continuation](../upstream-factorial-236/README.md)

All 32 contexts are previously observed 228 cases. `ledger.csv` retains all 48
terms at both scopes; `selection.json` records the fixed rank rule and selected
sites before finite outcomes. `local-splits.csv`, `outcomes.csv` and
`decisions.csv` retain every conditional component and terminal control.
Original logs, source copies, checks, every case and array inventories remain.

The terminal interventions use oracle native writes and do not recompute the
forecast upstream. They are diagnostic, not a prospective predictor. New
derivatives are not used. Separate NumPy reconstruction qualifies all selected
finite operations and finite heads. Original numerical failures remain.

Verify every release and historical record without model dependencies:

```sh
python -S research/boolean_coupling/reproduce_failure.py --verify-only --history --plan
```

Reproduce from fixed public inputs and the pinned checkpoint in the
[recorded MPS environment](../../research/README.md):

```sh
python research/boolean_coupling/reproduce_failure.py --run --study 235 --out runs/my-235
```

The wrapper regenerates 228 first. Reuse existing arrays with
`--input-228 runs/my-existing-228`; add `--offline` for a cached checkpoint.
Outputs must be new; failures stop the pipeline and preserve logs. Repetition
of these old inputs is numerical replication, not fresh confirmation.

Raw arrays and selected checkpoint tensors remain local, inventoried by size
and SHA256. No article draft is included. Question B remains unresolved.
