# 227b: finite MLP24 mixed-operation contributions

Read the [result, including failed comparisons](../../docs/analysis/227b-mlp24-mixed-primitives-result.md), [frozen protocol](../../docs/experiments/227b-mlp24-mixed-primitives.md), and parent [226b result](../finite-tail-interaction-226b/README.md).

Run in the pinned 225b environment, with the checkpoint cached and parent arrays regenerated:

```sh
PYTHONPATH=research .venv/bin/python -m boolean_coupling.tail227b \
  --input runs/baseline-interaction-225b \
  --tail runs/finite-tail-interaction-226b \
  --out runs/mlp24-mixed-primitives-227b-replication
PYTHONPATH=research .venv/bin/python -m boolean_coupling.audit227b \
  --out runs/mlp24-mixed-primitives-227b-replication
```

For independent finite normalization and projection checks, pass the pinned revision's cached `model.safetensors` file (the parent directory name must match revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306):

```sh
.venv/bin/python research/boolean_coupling/validate227b.py \
  --out runs/mlp24-mixed-primitives-227b-replication \
  --checkpoint /path/to/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/model.safetensors
PYTHONPATH=research .venv/bin/python -m boolean_coupling.describe227b \
  --out runs/mlp24-mixed-primitives-227b-replication
```

The last script is explicitly posthoc algebraic accounting. It does not create a new forecast or identify a learned circuit. No script above uses the disputed whole-block derivatives. The runner and validators refuse to overwrite their artifacts. New output directories are required for repetition.

`audit.json` retains all component/stratum counts and numerical maxima; `cases.csv` contains all 128 context/component results. `finite-projection-validation.json` records 608 finite-forward checks from checkpoint tensors. `posthoc-product-algebra.json` retains every actual-versus-ideal product comparison and arithmetic residual. Local arrays are inventoried by hash and stay in `runs/`.

**Important limits:**31/32 raw final-coordinate bridges to 226b's differently placed intervention fail 1e-4, despite passing the .005 two-score bridge;2/32 raw ideal-write comparisons also fail 1e-4. The latter local arithmetic ledger closes but does not qualify the former bridge. Both failure sets remain failures. Original 227 stopped at the first failed bridge in its second case; its complete first case remains local, and its original source/protocol/seal/traceback are preserved. 227b explicitly narrows the claim to its own product interventions without widening bounds.

The combined term passes the joint reduction rule16/16 descriptive contexts; no individual primitive contribution does. The product component halves the margin gap16/16 while failing the vector criterion16/16. Generic cases are below the eligibility guard. All contexts were previously observed and dependent. Question B remains open.

Plot both recorded studies with:

```sh
.venv/bin/python research/boolean_coupling/plot226_227.py
```

The repository includes code, reports and compact evidence. Article drafts and raw activation arrays remain excluded. See the [joint release record](../tail-interactions-226-227/README.md) for file integrity and the original local snapshot.
