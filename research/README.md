# Reproduce the current research

The latest studies return to the failed whole-response forecast. **235** provides a complete write-error ledger and finite input-state/local-approximation split. **236** recomputes MLP20/21 in the actual forecast and still passes **0/32** full-response cases. A native reset before MLP20 improves the result but still leaves **0/32**, isolating further failure in attention 21–24. The fully finite oracle control replays exactly. These are retrospective diagnostics; Question B remains open. Earlier 234 local attention-2 success remains valid within its restricted budget.

## Latest: source versus receiver SiLU

243 tests a frozen regional factorial on the existing 242b contexts. Source-only and receiver-only each fail the full contrast-recovery rule. Their interaction is substantial; native residual errors improve but every arm remains at **0/32** joint accuracy passes. [Report](../docs/analysis/243-source-receiver-silu-result.md).

```sh
python -S research/boolean_coupling/reproduce_regional.py --verify-only --history --plan
python research/boolean_coupling/reproduce_regional.py --run --offline \
  --input-242 runs/existing-242b --out runs/my-regional-test
```

Omit the input option to regenerate 242b first. The regional test adds 32 zero-edit controls and 128 modified forwards, without new native True measurements. QQ/FF replay 242b100/101 exactly; receiver edits leave source states unchanged. The corrected 243b scorer preserves the original metadata-key failure and all numerical rules. Raw arrays are inventoried locally, and exact same-environment endpoint replay is required. These are retrospective diagnostics, not new predictive confirmation.

## Latest: fresh uniform primitive test

242b applies one fixed primitive-family policy across all blocks, generating its own source and receiver states. All forecasts were pushed before native measurements. The primary passes **0/32**, at **50.3%** whole-displacement and **1.44 nats** margin error; finite SiLU reduces the former to **17.4%**, still **0/32**. [Report and numerical scope](../docs/analysis/242b-uniform-primitive-families-result.md).

```sh
python -S research/boolean_coupling/reproduce_uniform.py --verify-only --history --plan
python research/boolean_coupling/reproduce_uniform.py --run --offline --out runs/my-uniform-test
```

This test needs no earlier activation archive or prerequisite experiment. It needs the pinned MPS environment below. The wrapper records replication mode before forecasts and creates a local seal; it does not push or alter Git. Its already observed literals are not fresh confirmation. Original 242's implementation failure is preserved separately; 242b changes the unsupported nested SiLU AD expression, without changing tolerances or hypotheses.

## Latest: complete upstream vectors and transport geometry

241 traces all29 native-input remainder vectors to the missing input before MLP16. Four greedy sites pass0/32; the first sufficient prefix uses16 sites. Explicitly posthoc metric checks distinguish reduced individual energy from weakened cancellation. [Report](../docs/analysis/241-before-mlp16-vectors-result.md).

```sh
python -S research/boolean_coupling/reproduce_early.py --verify-only --history --plan
python research/boolean_coupling/reproduce_early.py --run --offline \
  --input-238 runs/existing-238 --input-240 runs/existing-240 --out runs/my-early-test
```

Omit both input options to regenerate the fixed chain. 241 itself uses CPU64 checkpoint/component arithmetic, not new ordinary model forwards; rebuilding its prerequisites needs MPS. The241 protocol requires the original input-array hashes. Replication in a different numerical environment may stop on those hashes; this release does not silently substitute different regenerated inputs. All used tangents, finite-difference endpoint witnesses, terminal vectors and original numerical bounds are retained. The three descriptive follow-ups are explicitly posthoc and add no causal interventions.

## Latest: target-blind SiLU selection and finite primitive intervention

240's forecast-derived 1% activation mask recovers the whole missing MLP16 contrast in32/32 old cases. Conditional final gaps shrink, but the materiality guard admits only3/32 native-input and0/32 forecast-input cases. Full finite continuation from the forecast input still misses the native response by25.2% and1.09nats. [Report](../docs/analysis/240-target-blind-silu-result.md).

```sh
python -S research/boolean_coupling/reproduce_silu.py --verify-only --history --plan
python research/boolean_coupling/reproduce_silu.py --run --offline \
  --input-239 runs/existing-239 --out runs/my-silu-test
```

Omit `--input-239` to regenerate the fixed prerequisite chain. 240 needs the MPS environment below. It seals all masks before intervention outcomes, runs384 direct activation edits with finite continuation, and independently audits all cases. Recorded work adds32 False forwards,32 source-only blocks,32 mask MLPs,384 intervention MLPs,4,608 suffix blocks and384 heads; no new ordinary True forward. The post-run prerequisite-hash recheck binds the archived original inputs; regenerated arrays are evaluated numerically rather than required to match archived compressed-file bytes.

## Latest: upstream generators and their input-dependent curvature

238 traces every local remainder through the original affine prefix. The selected MLP16–19 repair improves margin5.15→1.34nats and whole error72.3%→31.6%, but has0/32 full-response passes.239's finite-normalization/quadratic-SiLU formula explains the local generator contrast at MLP17/18 and fails the complete four-site account. [238 report](../docs/analysis/238-upstream-generators-result.md) · [239 report](../docs/analysis/239-generator-curvature-result.md).

```sh
python research/boolean_coupling/reproduce_curvature.py --run --out runs/my-curvature-test
```

Reuse a local238 prerequisite with `--offline --input-238 runs/existing-238`. To run238 using an existing237b prerequisite:

```sh
python research/boolean_coupling/reproduce_upstream.py --run --offline \
  --input-237 runs/existing-237b --out runs/my-upstream-test
```

238 seals its audited selection before intervention outcomes and separately reconstructs all finite-difference endpoints.239 alone needs CPU calculations and the cached checkpoint; rebuilding earlier studies requires the MPS setup below. All outputs must be new. Old native input directions are supplied only as retrospective diagnostics. Source-only states have pre-observation computational provenance, clarified [here](../docs/analysis/238-information-budget.md).

## Latest: late-attention omissions and numerical diagnosis

237b crosses QK and softmax while keeping normalization/value products and remaining MLPs finite. Neither selective policy passes the joint final-response bound; QK restoration leaves10.8% median whole error under a native reset. Even complete late computation from the ordinary wrong incoming state leaves72.3%. [Full report](../docs/analysis/237-late-attention-result.md).

Qualification is partial. Original237 fails its final bridge; original237b fails one Float64 MLP bridge. The237c numerical diagnostic preserves every bound/failure and finishes25,120 checks,25,119passing. Independent Float32 MLPs and same-backend public replays pass. These are old-context finite-computation diagnostics, not a new False-only predictor.

```sh
python research/boolean_coupling/reproduce_late.py --run --out runs/my-late-diagnostic
```

Add`--offline --input-236 runs/existing-236` to reuse a qualified cached prerequisite. Add`--include-failed-237` to preserve a rerun of the original failed attempt separately. Original237b audit failure is retained before the complete diagnostic runs. Unexpected failures stop for review. [All cases, logs, exact commands and computation budget](../results/late-attention-237b/README.md).

## Inspect without a model

From the repository root, with Python 3.12:

```sh
python research/boolean_coupling/reproduce_curvature.py --verify-only --history --plan
```

This uses only the standard library. It checks all active releases, their scientific source seals, and the publication-filtered historical archive. It also prints the actual reproduction commands. It does not load weights, rerun numerical audits against omitted activations, or constitute model replication.

The [experiment index](../docs/EXPERIMENTS.md) links every active protocol, report, audit, failed attempt and data summary. The [research notes](../docs/RESEARCH-NOTES.md) explain which hypotheses changed and why.

## Run the global failure diagnostic

After the environment setup below:

```sh
python research/boolean_coupling/reproduce_failure.py --run --out runs/my-global-diagnostic
```

This regenerates fixed 228 inputs, executes 235's ledger/finite interventions and audit, then 236's actual-continuation factorial and audit. Add `--offline` for a cached checkpoint. Reuse an existing 228 run with `--input-228 runs/existing-228`, or a qualified 235 prerequisite with `--study 236 --input-235 runs/existing-235`. Use `--study 235` for the ledger study alone. Every output must be new; failures stop and preserve logs. These old-context diagnostics are not new prospective confirmations.

[235's guide](../results/global-error-budget-235/README.md) explains the oracle terminal vectors and conditional split. [236's guide](../results/upstream-factorial-236/README.md) distinguishes normal and native-reset input regimes, all branches, actual FD directions and finite final readout. The prerequisite's ordinary-model work is separate from each diagnostic's recorded component budget. No private activation archive is required.

## Earlier prospective local forecasts

The unchanged [transport wrapper](boolean_coupling/reproduce_transport.py) reproduces 233/234; [source-context wrapper](boolean_coupling/reproduce_source_context.py) reproduces 232; [local wrapper](boolean_coupling/reproduce_local.py) reproduces 231 and verifies 230. Each retains its original inputs, seals, criteria and scope. See the [experiment index](../docs/EXPERIMENTS.md). These local successes did not predict the final native response.

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

235 uses [ledger235.py](boolean_coupling/ledger235.py), [finite235.py](boolean_coupling/finite235.py) and separate [audit235.py](boolean_coupling/audit235.py). 236 uses [continue236.py](boolean_coupling/continue236.py) and [audit236.py](boolean_coupling/audit236.py). [reproduce_failure.py](boolean_coupling/reproduce_failure.py) orchestrates public-input regeneration; [plot_failure.py](boolean_coupling/plot_failure.py) renders saved CSVs without model calls. Earlier scientific sources are unchanged.

233 uses explicit NumPy [transport233.py](boolean_coupling/transport233.py) and separate Torch CPU64 [audit233.py](boolean_coupling/audit233.py) with every intervened derivative qualified by AD/FD. [describe233.py](boolean_coupling/describe233.py) is explicitly post-observation. 234 uses [local234.py](boolean_coupling/local234.py), separate [audit234.py](boolean_coupling/audit234.py), fixed [cases234.json](boolean_coupling/cases234.json), [reproduce_transport.py](boolean_coupling/reproduce_transport.py), and model-free [plot_transport.py](boolean_coupling/plot_transport.py). All earlier source files and seals remain unchanged.

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

## Native attention and first-mixing diagnostics (244–246b)

```sh
python -S research/boolean_coupling/reproduce_mixing.py --verify-only --history --plan
python research/boolean_coupling/reproduce_mixing.py --run --offline --out runs/my-mixing-study
```

The wrapper regenerates public242b/243 prerequisites, native-attention244,
first attention mixing245, and the selected-block origin test246b, with separate
auditors. Add --input-242, --input-243, or --input-244 to reuse appropriate
qualified local outputs; --input-244 requires its --input-242. Output must be
new. All are known-context retrospective diagnostics. FF uses an ordinary
native True replay and is never counted as a successful forecast. Primitive
removals use observed factorial components. Fixed numerical bounds and earlier
failures remain. Article drafts are not part of any release.

## Native context, carry and source tests (247–249)

```sh
python -S research/boolean_coupling/reproduce_native.py --verify-only --history --plan
python research/boolean_coupling/reproduce_native.py --run --offline --out runs/my-native-study
```

This regenerates242b prerequisites, native attention247, finite-suffix248, and
source249 forecasts/controls/audit. Reuse --input-242; matching --input-247 and
--input-248 can also be supplied (later inputs require their matching earlier
prerequisites). Every output directory must be new. Unexpected failures stop.

247/248 use observed native incoming states. 249's source-only algorithm computes
its source from False plus the known edit, and seals every forecast before
oracle-control reads. These are known-context tests, not fresh confirmation.
Dense source and suffix computation is substantial; no speedup or learned
circuit is claimed. Full native accuracy still fails. Code/source seals and all
controls are retained; omitted arrays are regenerated from public pinned inputs.

## Native communication250b

Verify complete evidence and history without model imports:

```sh
python -S research/boolean_coupling/reproduce_communication.py --verify-only --history --plan
```

Run the finite factorial and separate audit in the pinned environment:

```sh
python research/boolean_coupling/reproduce_communication.py --run --offline --input-242 runs/existing-242b --out runs/new-communication-reproduction
```

Omit `--input-242` to regenerate the prerequisite via the existing242b pipeline. Each output must be new. The250b producer wrapper explicitly retains the original250 complete-data witness-count failure, checks its narrow preconditions, then seals corrected accounting; any earlier/different failure stops. All32 witnesses at blocks10/19/28 are independently audited. This requires352 full forwards,608 additional suffix blocks,32 additional full heads and the documented primitive recomputations; no speedup. [Frozen correction](../docs/experiments/250b-witness-completion-correction.md) · [Result](../docs/analysis/250b-native-source-communication-result.md).
