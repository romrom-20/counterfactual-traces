# 228b: reporting-only correction to the 228 scorer

The 228 model run, all96 forecasts, separate qualification audit, global seal and all32 later native True observations completed. The original final scorer then stopped on its first case's layerwise row construction:

    TypeError: dict() got multiple values for keyword argument 'block'

The literal-family metadata already used `block`; the diagnostic also attempted to name its numeric layer `block`. Preserve the original `audit228.py`, its pre-observation hash, failure log and all source/forecast/observation arrays unchanged.

The separately named `audit228b.py` changes only the layerwise diagnostic field to `layer` and its docstring to name this correction. No arithmetic, model call, forecast, criterion, threshold, numerical scope or input changes. The plotting reader and reproduction wrapper follow the corrected reporting field/module. Original preflight and pre-observation qualification still use the immutable original auditor.

The correction's original/corrected source hashes and original traceback hash are recorded in `score-correction.json` before executing the corrected scorer. This is a NumPy-only reporting correction after observation, not a new fresh-context experiment or a rescue of a failed scientific hypothesis.
