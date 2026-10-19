# Source-conditioned local forecast: experiment 232

The native displacement **after attention 2 across all receiving rows** is
predicted within the frozen criterion in 32/32 fresh pairs, 8/8 per stratum.
Median relative error is 1.33%, versus 5.90% for the original False-context rule.
Verdict accuracy is preserved in 32/32. No final-response forecast is made.

[Report](../../docs/analysis/232-source-context-result.md)
· [Protocol](../../docs/experiments/232-source-conditioned-local-forecast.md)

The budget includes an exact edited first block and one public source-only
attention call per pair. It is not a compact predictor, source-generation
explanation or speedup. The finite normalization/product rule is unchanged;
the expansion point moves to the constructed source-only context.

`cases.csv` retains every registered arm and both scopes. `forecast-audit.json`
contains 960 pre-target checks; `forecast-seal.json`, exact `sources/`, both raw
array inventories and the remote receipt bind all original predictions. They
were pushed at `617668f58595024316e3327d36738479d4ee53ac` before any ordinary
True target. Every forecast/observation JSON and original phase log is retained.

`hybrid-description.json` is explicitly post-observation. It recombines sealed
tangent/correction terms; its hybrid success is not a registered 232 forecast
result or a native-circuit intervention. Raw arrays and weights stay local,
inventoried by hash and size. The manifest excludes unpublished writing.

Verify all releases and the retained history without model dependencies:

```sh
python -S research/boolean_coupling/reproduce_source_context.py --verify-only --history --plan
```

Reproduce from fixed public inputs and the pinned checkpoint in the
[recorded MPS environment](../../research/README.md):

```sh
python research/boolean_coupling/reproduce_source_context.py --run --out runs/my-source-context
```

Add `--offline` for a cached checkpoint. The wrapper executes forecast, separate
qualification/global seal, native observation and unchanged scoring. Existing
outputs are refused; failures preserve logs. Replication reuses fixed inputs,
not another fresh confirmation. No GitHub push or private archive is required.

Individual commands, with `PYTHONPATH=research` from the repository root:

```sh
python -m boolean_coupling.local232 --phase forecast --out runs/new-232 --offline
python -m boolean_coupling.audit232 --phase forecast --out runs/new-232
python -m boolean_coupling.local232 --phase observe --out runs/new-232 --offline
python -m boolean_coupling.audit232 --phase score --out runs/new-232
python -m boolean_coupling.describe232 --run runs/new-232
python -m boolean_coupling.plot232 --run runs/new-232 --out runs/new-232/figure
```

The last two commands are optional model-free reporting. Every output must be
new. This experiment leaves the first-block generator and later/final response
open and does not recertify earlier numerical failures or identify a learned
circuit, universal mechanism or novelty priority.
