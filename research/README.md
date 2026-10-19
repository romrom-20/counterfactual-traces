# Reproduce the current research

The latest study is **232**: source-conditioned normalization/product forecasts pass the complete attention-2 receiving-displacement criterion in **32/32 fresh literal pairs**, with **1.33%** median relative error versus **5.90%** for the original False-context rule. Verdict-position accuracy is preserved. Its budget includes an exact edited first block and a source-only attention call; it does not forecast the final response or demonstrate a compact predictor. **231** passes only at the verdict, **230** rejects single-primitive accounts, and **228** still fails the whole-response criteria.

## Inspect without a model

From the repository root, with Python 3.12:

```sh
python research/boolean_coupling/reproduce_source_context.py --verify-only --history --plan
```

This uses only the standard library. It checks all active releases, their scientific source seals, and the publication-filtered historical archive. It also prints the actual reproduction commands. It does not load weights, rerun numerical audits against omitted activations, or constitute model replication.

The [experiment index](../docs/EXPERIMENTS.md) links every active protocol, report, audit, failed attempt and data summary. The [research notes](../docs/RESEARCH-NOTES.md) explain which hypotheses changed and why.

## Run the prospective local forecast

After the environment setup below:

```sh
python research/boolean_coupling/reproduce_source_context.py --run --out runs/my-source-context
```

This independently runs forecast → NumPy qualification/global seal → native True observation → unchanged scoring. Add `--offline` for a cached checkpoint. It reuses fixed original inputs, not a new confirmation. The [232 release guide](../results/source-conditioned-local-232/README.md) records the partial known-edit computation, all cases, both criteria and exact phase commands. No private activation archive is needed.

The [230 guide](../results/receiver-curvature-230/README.md) separately regenerates its 229 prerequisite from public inputs and runs the factorial/qualification. It retains registered failures separately from the post-observation composite analysis.

## Run the 229 diagnostic

After the environment setup below:

```sh
python research/boolean_coupling/reproduce_generator.py --run --out runs/my-generator-reproduction
```

The wrapper regenerates the preceding 228 run from its public fixed inputs, then executes 229 and its separate NumPy audit. Add `--offline` for a cached checkpoint. To reuse existing local 228 arrays:

```sh
python research/boolean_coupling/reproduce_generator.py --run --offline \
  --input-228 runs/my-existing-228 --out runs/my-new-generator
```

Every output directory must be new. This is a retrospective diagnosis with observed native parents; replication does not become a fresh forecast test. [229's release guide](../results/first-attention-generator-229/README.md) explains the signed vectors, direct primitive interventions, all-ineligible downstream rule, bootstrap corrections and budget. The model run makes 32 ordinary False calls and uses saved True states; zero *new* ordinary True calls does not make it False-only.

## Run the earlier whole-response test (228)

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

232 uses [local232.py](boolean_coupling/local232.py), separate [audit232.py](boolean_coupling/audit232.py), fixed [cases232.json](boolean_coupling/cases232.json), two-anchor derivative qualification, and [reproduce_source_context.py](boolean_coupling/reproduce_source_context.py). [describe232.py](boolean_coupling/describe232.py) explicitly recombines sealed components after observation; [plot232.py](boolean_coupling/plot232.py) renders the recorded results. These reporting steps make no model calls.

230 uses [run230.py](boolean_coupling/run230.py), [curvature230.py](boolean_coupling/curvature230.py) and separate [audit230.py](boolean_coupling/audit230.py). [describe230.py](boolean_coupling/describe230.py) is explicitly post-observation. 231 uses [local231.py](boolean_coupling/local231.py), [audit231.py](boolean_coupling/audit231.py), fixed [cases231.json](boolean_coupling/cases231.json) and [reproduce_local.py](boolean_coupling/reproduce_local.py). [plot_local.py](boolean_coupling/plot_local.py) renders both studies from saved records without model calls.

229 uses [generator229b.py](boolean_coupling/generator229b.py), [attention229.py](boolean_coupling/attention229.py), separate [audit229.py](boolean_coupling/audit229.py), position-resolved [describe229.py](boolean_coupling/describe229.py), [plot229.py](boolean_coupling/plot229.py) and [reproduce_generator.py](boolean_coupling/reproduce_generator.py). Original generator229.py and the intermediate bootstrap source remain as failure evidence; use generator229b.py.

228 uses [alignment228.py](boolean_coupling/alignment228.py), [smooth228.py](boolean_coupling/smooth228.py), original [audit228.py](boolean_coupling/audit228.py), corrected final [audit228b.py](boolean_coupling/audit228b.py), and [reproduce_alignment.py](boolean_coupling/reproduce_alignment.py). Its [diagnostic](boolean_coupling/diagnose228.py) and [plotter](boolean_coupling/plot228.py) use saved arrays/CSVs and make no model calls.

- [core225b.py](boolean_coupling/core225b.py), [run225b.py](boolean_coupling/run225b.py), [audit225b.py](boolean_coupling/audit225b.py): unchanged parent computation and audit.
- [tail226b.py](boolean_coupling/tail226b.py), [audit226b.py](boolean_coupling/audit226b.py): finite sublayer mixing, transport and causal removals.
- [tail227b.py](boolean_coupling/tail227b.py), [audit227b.py](boolean_coupling/audit227b.py): ordered primitive contributions and direct product edits.
- [validate227b.py](boolean_coupling/validate227b.py): independent finite RMSNorm/gate/up reconstruction from checkpoint tensors.
- [describe227b.py](boolean_coupling/describe227b.py): explicitly posthoc bilinear identity and arithmetic ledger.
- [reproduce_tail.py](boolean_coupling/reproduce_tail.py): dependency-free verification and pipeline orchestration; it does not alter the sealed runners.
- [plot226_227.py](boolean_coupling/plot226_227.py): redraws the published figure from the recorded CSVs.

The unsuffixed 225, 226 and 227 entry points preserve failed attempts. Use the `b` versions above for reproduction. The [225b guide](boolean_coupling/README.md) and both follow-up result guides retain individual phase commands and qualification details.
