# 228: a prospective native-alignment hypothesis fails

**Primary alignment: 0/32. Stronger native response: 0/32.** Finite MLP22/23 updates improve every case against both controls, but none meets the frozen accuracy bounds. Read the [report](../../docs/analysis/228-false-only-alignment-result.md), [original protocol](../../docs/experiments/228-false-only-alignment.md), and [interpretations recorded before observation](../../docs/analysis/228-prior-predictions.md).

The experiment retains 32 new literal pairs and all three forecast arms. The first-block receiving displacement is exact in every case; a post-observation check locates the first forecast discrepancy at the next attention update. That does not identify its individual nonlinear primitive or explain the later large error.

## Reproduce

Use the recorded Python 3.12.14 / Apple Silicon MPS / Float32-eager environment. Install the [recorded dependencies](../../research/boolean_coupling/requirements-recorded.txt), then:

```sh
python research/boolean_coupling/reproduce_alignment.py --run --out runs/my-alignment-reproduction
```

The wrapper caches the pinned checkpoint if needed. Use `--offline` when it is already available. A new output directory is required. The six phases are the old-case False-only preflight, separate preflight audit, all False observations/forecasts, separate qualification audit and seal, later native True measurements, and corrected final scoring. A failing phase stops before subsequent phases.

The experiment's exact inputs are published in `cases228.json`. Replication reuses those inputs and their originally frozen historical catalog; it is not another fresh-context confirmation. The original catalog contains 1,184 hashes from available recorded paired inputs. Its scope is explicit and does not assert coverage of every historical artifact format.

The rule evaluates the known source and predicted-state finite tail. Its substantial counterfactual computation is documented in the protocol; it is not a compact learned circuit. Derivatives are those of declared smooth promoted CPU64 functions, reanchored to public MPS False states. The original Float32 whole-block Jacobian is not recertified.

Verify every release without a model:

```sh
python -S research/boolean_coupling/reproduce_alignment.py --verify-only --history --plan
```

This checks published bytes and source seals. Numerical qualification against raw arrays requires regeneration. Raw activations and weights are excluded, with their sizes/hashes inventoried in `local-array-inventory.json`.

## Evidence map

- `design.json`, `runtime.json`, and `preflight/`: frozen hypothesis, runtime and old-case numerical qualification.
- `forecast-manifest.json`, `forecast-status.json`, `forecast-audit.json`, `forecast-qualification-seal.json`: all 96 forecasts and their qualification before native True observations.
- `pre-observation-git.json`: the source/forecast commit verified remotely before observation. Its system-interpreter receipt clock is not used in the experimental ordering test. The experiment's own qualification/observation timestamps and file hashes establish the checked phase order.
- `observation-start.json`, `status.json`, `audit.json`: observation ordering, exact compute counts and all outcomes.
- `cases.csv`, `layerwise.csv`: every arm/case and registered post-observation trajectory diagnostic.
- `first-attention-boundary.json`: an explicitly post-observation exact first-block check and attention2 error measurements; no new model calls.
- `original-score-failure.log`, `score-correction.json`: preserved metadata failure and the original/corrected scoring source hashes.
- Runner/scorer logs: actual execution records, including the unsuccessful final scorer.

The original scorer used `block` for both literal family and numeric layer and stopped while building its first diagnostic row. [228b](../../docs/experiments/228b-reporting-correction.md) changes only that field to `layer` and the scorer docstring. No forecast, arithmetic, model call, criterion or bound changes. `audit228.py` remains immutable; the reproduction wrapper uses `audit228b.py` only for final scoring.

The forecast manifest and code were pushed at [104b8e8](https://github.com/romrom-20/counterfactual-traces/commit/104b8e89765ef274a70d7a557f8240b7c71580bc) before ordinary full True measurement. The repository remains private; article drafts are excluded.
