# 229b: inherited audit metadata correction

The frozen 229 runner stopped before creating an output directory, loading the
model, or calculating any new result: it expected `audit228b.py` to emit
`status: passed`, while that scorer emits `status: complete`. The original
runner and failure log are retained. No scientific comparison failed at this step.

An initial 229b bootstrap also stopped before directory creation or model loading:
228 stores aggregated check counts and maximum errors, not a list of individual
checks. That exact source snapshot and second log are retained in the release.

The corrected 229b runner changes only its docstring, source-seal inventory and
bootstrap gate: require the actual `complete` value, passing global forecast
ordering, and each of the five recorded maximum errors within its original bound (exact
prefix/source, 1e-4 gate/up, 0.005 scores).
The attention graph, derivatives, inputs, interventions, criteria, computation
budget, and final audit remain unchanged. This is the execution of 229 after a
pre-model metadata correction, not a new hypothesis or fresh-context test.
