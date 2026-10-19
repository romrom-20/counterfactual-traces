# Prospective local forecast: experiment 231

The frozen native **attention-2 verdict-position displacement** criterion passes
16/16 fresh literal pairs, 4/4 in each stratum. The stronger whole-receiving
criterion fails in all 16. No final residual or verdict-margin forecast is made.

[Report](../../docs/analysis/231-local-forecast-result.md)
· [Protocol](../../docs/experiments/231-local-normalization-product-forecast.md)

The computational budget includes one exact edited first block and one finite
source-only attention-2 call per pair. It is not a compact predictor or speedup.
Native ordinary True targets are measured only after all 64 forecasts are
qualified/sealed and pushed at `d4cf03e78bd08cfc5d3648293178f742c645faeb`.

`cases.csv` retains every arm and both scopes. `forecast-audit.json` contains the
304 pre-observation NumPy checks; `forecast-seal.json` binds the unchanged source
snapshots, forecasts and metadata. `forecasts/*.json` and `observations/*.json`
retain all qualification/parent-replay records. `pre-observation-git.json`
records the remote commit verification before targets. `sources/` are exact
pre-target copies; the old core225b copy keeps only its existing narrow style
exemptions, while correctness checks remain active.

`forecast-array-inventory.json` was pushed before observation and remains
unchanged. `local-array-inventory.json` additionally inventories the later native
arrays. Raw arrays and weights stay local. The release manifest binds compact
records, code, protocols, reports and figures; it excludes article drafts.

Verify all releases without numerical dependencies:

```sh
python -S research/boolean_coupling/reproduce_local.py --verify-only --history --plan
```

Reproduce the local forecast from the public pinned checkpoint and fixed inputs
in the [recorded MPS environment](../../research/README.md):

```sh
python research/boolean_coupling/reproduce_local.py --run --out runs/my-local-reproduction
```

The wrapper runs forecast → separate qualification/seal → ordinary True
observation → unchanged scoring. Add `--offline` for a cached checkpoint. It
refuses existing outputs and preserves phase logs on failure. Repeating these
fixed contexts is replication, not another fresh-input confirmation. No GitHub
push or private activation archive is needed for replication.

Individual commands, from the repository root with `PYTHONPATH=research`:

```sh
python -m boolean_coupling.local231 --phase forecast --out runs/new-231 --offline
python -m boolean_coupling.audit231 --phase forecast --out runs/new-231
python -m boolean_coupling.local231 --phase observe --out runs/new-231 --offline
python -m boolean_coupling.audit231 --phase score --out runs/new-231
python -m boolean_coupling.plot_local --kind forecast --run runs/new-231 --out runs/new-231/figure
```

The separate [230 guide](../receiver-curvature-230/README.md) reproduces the old
context factorial that motivated this hypothesis. Existing research numerical
failures, grounding limits and the unsolved full Question B remain in force.
