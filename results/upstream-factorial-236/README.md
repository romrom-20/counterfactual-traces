# 236: upstream MLP20/21 factorial and native-reset controls

Recomputing both MLPs in the actual forecast reduces median margin error from
6.588 to 4.849 nats but passes the full-response criterion in 0/32 cases. With an exact
native reset before MLP20, the same continuation leaves 0.344 nats median error
and 24.4% whole-receiving displacement error, also 0/32. Only the fully finite
oracle replay passes 32/32, exactly. Later attention 21–24 approximations remain
insufficient even after the native reset.

[Report](../../docs/analysis/236-upstream-factorial-result.md)
· [Frozen protocol](../../docs/experiments/236-upstream-mlp-factorial.md)
· [Global diagnostic prerequisite](../global-error-budget-235/README.md)

All 11 branches and 32 old 228 cases remain in `outcomes.csv`. `layerwise.csv`
retains each branch's incoming MLP states; `decisions.csv` includes raw and
controlled criteria and factorial interactions. Source snapshots, original
logs, runtime, case records, checks and derivative-witness hashes are retained.
The audit qualifies 1664 actual used directions by FD, 1024 finite MLPs with
NumPy, and 352 finite heads. These are separate code paths, not an independent
research team. The original Float32 derivative disagreement is not recertified.

Verify all releases using the standard library:

```sh
python -S research/boolean_coupling/reproduce_failure.py --verify-only --history --plan
```

Reproduce the full fixed-input diagnostic chain:

```sh
python research/boolean_coupling/reproduce_failure.py --run --out runs/my-global-diagnostic
```

It regenerates 228, executes 235 and its independent audit, then 236 and its
independent audit. Reuse existing 228 with `--input-228 runs/existing-228`.
To run 236 using a qualified local 235 prerequisite:

```sh
python research/boolean_coupling/reproduce_failure.py --run --study 236 --offline \
  --input-235 runs/existing-235 --out runs/my-236
```

Use the [recorded MPS setup](../../research/README.md); every output directory
must be new. The public checkpoint is sufficient; no private archive or API key
is required. Logs and failures are preserved. Oracle resets use old native
states and cannot count as prospective inputs or a new fresh-context test.

Raw arrays and the 235 prerequisite weights remain local; inventories and the
release manifest bind them. Unpublished article drafts remain excluded.
