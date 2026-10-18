# Reproduce the current research

The latest test is **228**, a prospective alignment forecast that achieves 0/32 passes on its frozen criteria. The earlier causal sequence is **225b → 226b → 227b**. It starts from exact public prompt pairs and one pinned checkpoint, then isolates a local sign reversal, downstream mixing, and its primitive contributions. It remains retrospective. There is no successful general False-only predictor of the native Boolean response.

## Inspect without a model

From the repository root, with Python 3.12:

```sh
python research/boolean_coupling/reproduce_alignment.py --verify-only --history --plan
```

This uses only the standard library. It checks all three releases, the unchanged pre-execution scientific source hashes, and the publication-filtered historical archive. It also prints the actual reproduction commands. It does not load weights, rerun numerical audits against omitted activations, or constitute model replication.

The [experiment index](../docs/EXPERIMENTS.md) links every active protocol, report, audit, failed attempt and data summary. The [research notes](../docs/RESEARCH-NOTES.md) explain which hypotheses changed and why.

## Run the latest prospective test

Follow the environment setup below, then run:

```sh
python research/boolean_coupling/reproduce_alignment.py --run --out runs/my-alignment-reproduction
```

This runs the unchanged 228 preflight and forecast machinery, seals and audits all forecasts, and only then measures the native True targets. Final scoring uses the separately named 228b metadata correction. See the [228 guide](../results/false-only-alignment-228/README.md) for ordering, budget, failed criteria, original failure record and fixed inputs. Replication is not a second fresh-context demonstration. The wrapper can cache the public pinned checkpoint; add `--offline` when it is already available.

## Run the earlier causal sequence from public inputs

The validated environment is Python 3.12.14 on Apple Silicon/MPS, with Float32/eager execution. Runtime versions are pinned in [requirements-recorded.txt](boolean_coupling/requirements-recorded.txt). The original machine had 24 GB of RAM. Allow several GB for checkpoint weights and generated activations. Other backends and versions have not been validated.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r research/boolean_coupling/requirements-recorded.txt
python research/boolean_coupling/reproduce_tail.py --run --out runs/my-reproduction
```

This runs the unchanged 225b parent, its qualification and intervention audits, then the unchanged 226b and 227b model runners and their separate auditors. It finishes with the checkpoint-based finite norm/projection checks and the explicitly posthoc product-algebra check. The wrapper finds the pinned `model.safetensors` in the local Hugging Face cache after the parent run; `--checkpoint` can supply its snapshot path explicitly.

The first parent run may download the public checkpoint. Use `--offline` when it is already cached. These experiments require no API key and make no paid model API calls. They do execute local model computations.

Reuse a completed 225b output to avoid rerunning the parent:

```sh
python research/boolean_coupling/reproduce_tail.py --run --offline \
  --base runs/my-existing-225b --out runs/my-new-continuation
```

`--out` must be new. Each phase writes its log, and `pipeline.json` records completed steps. A failing phase stops the sequence and preserves its outputs. No later step silently substitutes a native state or widens a tolerance.

## What the computation includes

225b performs 64 ordinary model computations and 704 finite suffix continuations. It includes primitive derivative qualification at its declared local scope.

226b performs 640 reference-block replays, 160 additive-input blocks, 160 additional local MLP calls, 480 intervention MLP calls, 1,920 continuation blocks and 1,120 finite heads. 227b performs 128 reference blocks, 96 local MLP calls, 320 direct down projections, 1,280 continuation blocks and 320 finite heads. These are different compute units, not interchangeable full-model calls. The follow-ups use finite computations and do not recertify the disputed original whole-block derivatives.

The checkpoint-based validator adds 608 finite-forward comparisons, without a full model call. All 32 cases in each follow-up are previously observed and dependent. Numerical validation does not make them fresh examples.

## Code map

The latest study uses [alignment228.py](boolean_coupling/alignment228.py), [smooth228.py](boolean_coupling/smooth228.py), original [audit228.py](boolean_coupling/audit228.py), corrected final [audit228b.py](boolean_coupling/audit228b.py), and [reproduce_alignment.py](boolean_coupling/reproduce_alignment.py). Its [diagnostic](boolean_coupling/diagnose228.py) and [plotter](boolean_coupling/plot228.py) use saved arrays/CSVs and make no model calls.

- [core225b.py](boolean_coupling/core225b.py), [run225b.py](boolean_coupling/run225b.py), [audit225b.py](boolean_coupling/audit225b.py): unchanged parent computation and audit.
- [tail226b.py](boolean_coupling/tail226b.py), [audit226b.py](boolean_coupling/audit226b.py): finite sublayer mixing, transport and causal removals.
- [tail227b.py](boolean_coupling/tail227b.py), [audit227b.py](boolean_coupling/audit227b.py): ordered primitive contributions and direct product edits.
- [validate227b.py](boolean_coupling/validate227b.py): independent finite RMSNorm/gate/up reconstruction from checkpoint tensors.
- [describe227b.py](boolean_coupling/describe227b.py): explicitly posthoc bilinear identity and arithmetic ledger.
- [reproduce_tail.py](boolean_coupling/reproduce_tail.py): dependency-free verification and pipeline orchestration; it does not alter the sealed runners.
- [plot226_227.py](boolean_coupling/plot226_227.py): redraws the published figure from the recorded CSVs.

The unsuffixed 225, 226 and 227 entry points preserve failed attempts. Use the `b` versions above for reproduction. The [225b guide](boolean_coupling/README.md) and both follow-up result guides retain individual phase commands and qualification details.
