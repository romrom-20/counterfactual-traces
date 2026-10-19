# 237b/237c: late attention omissions and complete numerical diagnosis

Selective QK/softmax policies all pass the full native-response criterion in
0/32, even with a native reset. QK restoration helps but leaves10.8% median
whole-displacement error. The complete finite attention control is exact with
the native reset; from the ordinary wrong input it still leaves72.3% error.
Fixed-input signed effects shift from mainlyQK at attention21 to mainlysoftmax
at23. They are defined operation effects, not learned features or a circuit.

**Qualification is partial.** The original237b audit failed one Float64 MLP
bridge.237c completes25,120 checks:25,119pass, the original failure remains.
All primitive graph/derivative checks and1536Float32MLP reconstructions pass;
1920publicMLP replays are bit-exact. No audit pass is substituted for the
original failure; conditional outcome counts are not a new predictor.

[Full report](../../docs/analysis/237-late-attention-result.md)
· [Frozen237b protocol](../../docs/experiments/237b-late-attention-removals.md)
· [237c numerical diagnostic](../../docs/experiments/237c-complete-numerical-diagnostic.md)
· [Preserved failed237](../late-attention-237-failed/README.md)

All32 cases/12arms are in`outcomes.csv`; all fixed-input vectors and evolving
factorials in`local-effects.csv`/`pipeline-effects.csv`; frozen criteria in
`decisions.csv`. The separate`margin-factorial.csv` and`posthoc-description.json`
are explicitly posthoc, with no new model calls. Original logs, failed audit,
complete checks, exact public replays, source snapshots and array hashes remain.

Verify retained evidence without importing model libraries:

```sh
python -S research/boolean_coupling/reproduce_late.py --verify-only --history --plan
```

Regenerate from fixed public228 inputs through235/236:

```sh
python research/boolean_coupling/reproduce_late.py --run --out runs/my-late-diagnostic
```

Reuse a qualified local236 prerequisite and cached checkpoint:

```sh
python research/boolean_coupling/reproduce_late.py --run --offline \
  --input-236 runs/existing-236 --out runs/my-late-diagnostic
```

Add`--include-failed-237` to rerun the original failed attempt separately.
The pipeline preserves the original237b numerical audit failure before running
the complete unchanged-bound diagnostic and public replay. An unexpected failure
or unexpected original-audit pass stops for investigation. Completing that
pipeline does not mean all numerical checks passed. This is replication of old
contexts, not fresh confirmation. Every output must be new.

Regenerate the scientific figure:

```sh
PYTHONPATH=research python -m boolean_coupling.plot_late \
  --run runs/my-late-diagnostic/experiment237b --out docs/figures/my-late-diagnostic
```

Use the [recorded MPS environment](../../research/README.md). No private data or
API key is required.33local arrays total1,459,313,107bytes; weights and activations
are omitted and inventoried, never included in this release. The prerequisite
MLP weights remain bound by235's inventory. Unpublished article drafts remain offGitHub.
