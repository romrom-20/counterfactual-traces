# 229: the first attention discrepancy

**Qualified retrospective diagnostic; Question B remains open.** Receiver-only
curvature accounts for the first verdict-position error in 32/32 previously
observed contexts. Source–receiver interaction is substantial elsewhere in the
sequence. The isolated finite-suffix margin error is below the frozen 0.10-nat
materiality guard in every case, so the downstream success criterion is
**ineligible in all 32**, not a failed or successful behavioral account.

[Scientific report](../../docs/analysis/229-first-attention-generator-result.md)
· [Frozen protocol](../../docs/experiments/229-first-attention-generator.md)
· [Bootstrap corrections](../../docs/experiments/229b-bootstrap-correction.md)
· [Research guide](../../research/README.md)

## Evidence retained

- `design.json`, `runtime.json`, `status.json`: pre-model source/input hashes,
  arithmetic, actual compute counts and all 32 completed array/record hashes.
- `cases.csv`: every local signed vector norm/projection and frozen decision.
- `outcomes.csv`: all 480 finite-suffix outcomes, including random controls.
- `audit.json`: the separate NumPy implementation's 1,440 checks, original
  numerical bounds, every runner-check maximum and all stratum counts.
- `cases/*.json`: every original per-case qualification record, byte unchanged.
- `position-effects.csv`, `description.json`: explicitly post-observation,
  model-free position/global geometry and descriptive error ratios. These do
  not revise eligibility, thresholds, decisions or intervention selection.
- `bootstrap/`: both zero-model-call failure logs and intermediate 229b source.
  Original generator229.py is retained in the research module. The successful
  generator229b.py only corrects the inherited audit metadata gate and source
  inventory; no graph or scientific criterion is changed.
- `runner.log`, `audit.log`: actual original completed execution logs.
- `local-array-inventory.json`: SHA256 and size of omitted weights/raw arrays.
- `manifest.json`: exact committed scientific code/protocol/report/figure and
  release-file hashes. The article draft is excluded.

No raw activation arrays or checkpoint weights are published. They can be
regenerated from fixed public inputs and the pinned public model. A raw-array
hash is an inventory, not independent validation of an omitted array.

## Verify without model dependencies

```sh
python -S research/boolean_coupling/reproduce_generator.py --verify-only --history --plan
```

This verifies the new manifest/source seals, metadata and recorded checks, then
all prior releases/historical evidence. It does not reexecute the omitted
arrays, rerun attention, or replicate the scientific findings.

## Reproduce from public inputs

Use the [recorded MPS environment](../../research/README.md). Then:

```sh
python research/boolean_coupling/reproduce_generator.py --run --out runs/my-generator
```

The wrapper regenerates the 228 prerequisite from the original fixed prompts,
including globally sealing its forecasts before observing its native True
targets. It then runs the corrected 229b diagnostic and separate audit. Add
`--offline` when the pinned checkpoint is cached. No API key is needed.

If a completed 228 run is already available:

```sh
python research/boolean_coupling/reproduce_generator.py --run --offline \
  --input-228 runs/existing-228 --out runs/my-new-generator
```

Direct phases, from the repository root with `PYTHONPATH=research`:

```sh
python -m boolean_coupling.generator229b --offline \
  --input runs/existing-228 --out runs/new-229
python -m boolean_coupling.audit229 --run runs/new-229
python -m boolean_coupling.describe229 --run runs/new-229
python -m boolean_coupling.plot229 --run runs/new-229 --out runs/new-229/figure
```

Every output/audit/description must be new; reruns refuse overwriting. The
wrapper records logs and `pipeline.json` and stops on a failed phase. The
description/figures are optional zero-model-call reporting, separate from the
frozen numerical/outcome audit. Reusing the original contexts does not become
a new fresh-context confirmation.

## Computation and boundaries

The main run uses 32 ordinary False replays and **zero new ordinary True calls**,
but saved native True states are retrospective inputs. It is not False-only.
There are 128 public attention corners, 128 reconstructed manual public corners,
128 primitive attention edits, 128 smooth CPU64 corners, 96 JVP/FD comparisons
(192 endpoint evaluations), 480 MLP2 continuations, 12,480 later public blocks,
and 480 finite heads. The separate NumPy auditor makes zero model calls.

The finite suffix is different from 228's remaining linear transport. An exact
R+C reconstruction is a control, not a successful prediction. Individual signed
components are order-defined, and projections are not unique causal fractions.
No learned feature/circuit, generality, priority or solution to Question B is
claimed. The original derivative disagreement and earlier failed experiments
remain unchanged.
