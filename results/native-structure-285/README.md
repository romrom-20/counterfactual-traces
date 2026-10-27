# Native source prediction across prompt structures: Experiment 285

**Forecasts and numerical qualification are complete; native outcomes are not yet
measured.** This version is published before the first native True computation.
All 32 forecasts and their audit are sealed in [pre-native](pre-native/forecast-seal.json).
The scientific code and exact inputs were frozen before checkpoint execution.

The test changes the causal prompt prefix in four ways while retaining the same
facts and one Boolean edit. It compares the existing False-coefficient cubic
first-source rule with fixed directions from historical False-derived forecasts.
The matched donor uses the current cubic forecast's magnitude, so the comparison
tests direction specificity conditional on that calculation. All 27 later source
MLPs and receiving operations remain finite; this is not a compact predictor.

[Protocol and both frozen hypotheses](../../docs/experiments/285-native-structure.md) ·
[Runner](../../research/boolean_coupling/run285.py) ·
[Separate numerical audit](../../research/boolean_coupling/audit285.py)

The 224 forecast forwards completed in 88.86 seconds. The separate audit passes
4,288 required checks, including independent False-coefficient derivatives and
exact public primitive/head replay. All 320 raw 1e-4 diagnostics also pass in
this run; older numerical failures remain unchanged. No complete current first
source SiLU or ordinary True trajectory was evaluated before this seal. Forecast
arrays occupy 69,512,921 bytes and remain local; the omitted-array inventory binds
their exact hashes and sizes.

H1 requires native first-source-write and final-response prediction. H2 separately
requires a material native repair beyond the matched donor, including the frozen
.02-nat margin guard. A local precision gain cannot substitute for H2. Every case
remains in the denominator. Related templates, one checkpoint and substantial
finite computation limit the claim; no learned circuit, priority, efficiency or
complete Question B solution is established.

```sh
python -S research/boolean_coupling/verify285.py --phase forecast
```

This verifies sealed records and decisions, not an independent checkpoint run.
Private article drafts, model weights and activation arrays are excluded.
