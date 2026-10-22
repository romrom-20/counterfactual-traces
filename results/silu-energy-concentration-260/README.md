# 260: local concentration versus cancellation

[Report](../../docs/analysis/260-silu-energy-concentration-result.md) ·
[Protocol](../../docs/experiments/260-silu-energy-concentration.md) ·
[Source](../../research/boolean_coupling/energy260.py)

This retrospective diagnostic uses all 24 saved C/S MLP witness inputs from258:
four contexts, blocks1/14/28, two forecast arms. No model forward, new native
measurement or new global intervention. C/S first-block inputs coincide.

The original ranked 1% mask captures >=90% of individual omitted-write energy
in 20/24 whole inputs, meeting the fixed 18/24 rule. Median 97.73%, minimum 89.69%.
At verdict rows the count is 16/24, median 94.69%, minimum zero. These are local
neuron-coordinate energy fractions, not fractions of the final behavioral effect.
All 48 scope records and 192 numerical checks remain in the CSV/audit; no cases
are removed. Component bridges inherit258b; original failed raw checks stay failed.

The source and protocol were committed in 1c639d5 before this diagnostic's energy
statistics were calculated, after258/259 outcomes were known. This is not a fresh
predictive registration or evidence of an upstream learned circuit.

```sh
PYTHONPATH=research python -m boolean_coupling.energy260 \
  --run runs/my-258-reproduction/experiment258 --out runs/my-energy-diagnostic
python -S research/boolean_coupling/verify_reader_guide.py
```

The first command requires locally regenerated258 witnesses and the pinned cached
checkpoint for CPU down weights. It must use a new output directory. The second
verifies public records, summaries and reader-figure provenance without a model;
it does not recalculate the omitted tensor products. No private data service or
unpublished article is required. See the [research guide](../../research/README.md).
