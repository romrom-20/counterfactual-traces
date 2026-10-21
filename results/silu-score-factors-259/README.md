# Experiment259: all retrospective score-factor records

[Report](../../docs/analysis/259-silu-score-factors-result.md) ·
[Protocol](../../docs/experiments/259-silu-score-factors.md) ·
[Source](../../research/boolean_coupling/score259.py)

This is a fixed-input diagnostic after258's outcomes, with **zero model
forwards or new native measurements**. It uses all24 saved C/S MLP witnesses:
four contexts, blocks1/14/28, two existing forecast states. C/S block1 inputs
coincide; these are not independent observations.

- `score-factors.csv`: all432 input/score/scope records, including every signed
  share, restored norm, remaining-gap error, eligibility and original-mask overlap.
  Scores BUW/BU/BW/UW/B/U/W/DUW/D use exactly the same floor1% budget.
- `audit.json`: all72 passing numerical checks, original-source/protocol hashes,
  input258 audit/seal hashes, all summaries and both registered question outcomes.
- `description.json`: posthoc all-score/all-input figure provenance.
- `logs/`: unedited diagnostic and plot output.
- `manifest.json`: published source/report/figure/data SHA256 bindings.

Dropping B leaves6.7644× as much median whole-gap error. Dropping W stays within
.05 additional error24/24. D with U/W also captures similarly locally, so the
full bound is not uniquely necessary. The alternative masks have **not** been
tested as new global interventions, fresh native forecasts, or learned features.
The component qualification inherits258b; failed raw whole projections remain
failed in the258 release.

First regenerate258 arrays as in the [research guide](../../research/README.md),
then run:

```sh
PYTHONPATH=research python -m boolean_coupling.score259 \
  --run runs/my-258-reproduction/experiment258 --out runs/my-259-diagnostic
```

The output must be new. This needs the pinned cached checkpoint for CPU down
weights and the locally generated forecast witness arrays, not a private
activation archive. CI verifies recorded integrity and recalculates CSV
summaries; it does not rerun the tensor calculations.
