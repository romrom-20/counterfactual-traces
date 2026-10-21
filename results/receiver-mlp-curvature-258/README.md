# Experiment258 / qualified258b: complete receiver-MLP forecast evidence

[Readable report](../../docs/analysis/258-receiver-mlp-result.md) ·
[Original protocol](../../docs/experiments/258-receiver-mlp-curvature.md) ·
[Explicit numerical boundary](../../docs/experiments/258b-matched-projection-qualification.md)

Primary ranked1% finite SiLU **32/32**; finite reference32/32;
quadratic and matched-random1% both0/32. All forecasts, new input hashes,
behavioral criteria and the qualified pre-audit precede native measurements.
The remaining99% use quadratic SiLU; the declared other components stay finite.
Actual instrumentation evaluates full public SiLU; no speedup is established.

## Read these records first

- `design.json`, `sources/`: original pre-execution source/input freeze.
- `pre-native/`: immutable public forecast receipts committed in
  [`0c834b9`](https://github.com/romrom-20/counterfactual-traces/commit/0c834b9d7cd65ac2a0eb8867ae456f696e52d8a6).
  `measurement-proof.json` binds the remote head and seal accepted before True.
- `outcomes.csv`: all128 arm/case outcomes, vector norms/cosines/relative errors,
  finite margin errors and passes; no cases omitted.
- `layer-errors.csv`, `policy-increments.csv`: every layer and both scopes.
- `signed-local-vectors.csv`: complete/selected/unselected local gap geometry,
  signed shares, norms, closure and explicit eligibility. Different arms have
  different incoming states; these are not unique global causal percentages.
- `forecast-status.json`, `status.json`, `forecast-seal.json`, `seal.json`:
  case array/record hashes and exact recorded work counts.
- `description.json`: source-bound posthoc summaries, all-case figure hashes,
  selected-coordinate count and per-layer eligibility.

## Preserve the original failed audit

`failure-audit-pre.json`, `preserved-original-audit/` and the original
`sources/audit258.py` retain the failed1e-4 CPU64 whole-down comparison.
`projection-investigation.json` records72 exact matched-full-shape Float32
replays,24 failed original CPU64 comparisons, and sliced-Float32 errors.
`qualification258b.json` and `qualification-sources/` bind the separately frozen
pre-native numerical revision. Forecasts and behavioral criteria did not change.

`pre-audit.json` passes19,416 checks and `audit.json` passes20,016 under258b.
All96 final full-shape down replays are exact. Raw comparison CSVs retain the
original1e-4 tolerance:96/288 final projection/rotary/QK failures;32/96 final
CPU64 whole-down failures. No independent complete whole-model real-arithmetic
or original whole-block derivative certification is claimed.

`logs/` retains actual forecast, original failed audit, shape investigation,
native measurement, qualified final audit and plotting output. The qualified
pre-audit redirected log is empty; its source-bound structured checks and
timestamp provenance are in `pre-audit.json`, not a reconstructed console log.

## Reproduce or inspect

```sh
python -S research/boolean_coupling/reproduce_receiver_mlp.py --verify-only --history --plan
python research/boolean_coupling/reproduce_receiver_mlp.py --run --offline --out runs/my-258-reproduction
```

The first command verifies metadata, summaries, sources and all historical
releases without loading a model. It does not recalculate omitted tensors.
The second needs the pinned Python3.12.14/Torch2.14.0/Transformers4.57.6
Apple Silicon/MPS Float32 eager environment, NumPy2.5.3, cached public Qwen2.5-1.5B
revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306, and several GB of free space.
See the [environment and stage guide](../../research/README.md).

The wrapper uses explicit local replication seals, never Git publication. It
runs258b directly rather than fabricating another historical original failure.
The original individual stages ran; a separate full wrapper replay was not run.
Repeating these public literals is numerical replication, not fresh confirmation.

Raw arrays, weights and primitive witnesses remain local, inventoried by size
and SHA256 in `local-array-inventory.json`. Regeneration needs no private data
service. Unpublished article drafts and private agent handoffs are excluded.
