# 242b: fresh uniform primitive-family test

**Primary 100 and every secondary pass 0/32 full-response cases.** Four affine
normalization arms are unstable in every case. Four finite normalization arms
complete, but none reaches the declared residual and margin criteria.

[Report](../../docs/analysis/242b-uniform-primitive-families-result.md) ·
[Protocol](../../docs/experiments/242b-uniform-primitive-families.md) ·
[All outcomes](outcomes.csv) · [Final audit](audit.json) ·
[Pre-native seal](forecast-seal.json) · [Remote proof](measurement-proof.json)

The seal and all forecasts were pushed before native measurements at commit
`dd 81a 983cd 66f 488aecfd 0a 553670071e 2a 0275a`. The pre-publication manifest binds
those original files separately from this final release. Each forecast record
retains numerical checks, normalized derivative witnesses, stability outcomes
and softmax statistics. Raw arrays remain local; their complete SHA256/size
inventory is published. Article drafts are excluded.

Verify with Python 3.12 and no model imports:

```sh
python -S research/boolean_coupling/reproduce_uniform.py --verify-only --history --plan
```

After the [environment setup](../../research/README.md), regenerate this test
without the earlier prerequisite chain:

```sh
python research/boolean_coupling/reproduce_uniform.py --run --offline --out runs/my-uniform-test
```

The new output directory is required. This wrapper explicitly records replication
mode before forecasts, uses a local seal, does not publish or change Git, and
then measures all native references. Repeating these now-observed literals is
retrospective numerical replication, not fresh confirmation. The scientific
thresholds and numerical tolerances are unchanged. Unexpected qualification
failures stop; declared instability terminates only that arm and counts as a
failed prediction. Original investigation used the stronger private remote seal.

`layer-profiles.csv`, `policy-interactions.csv`, `raw-softmax-statistics.csv` and
`description.json` are explicitly posthoc descriptions with no new model calls.
They can be regenerated after the final audit using
`python -m boolean_coupling.describe242b --run RUN --figures FIGURE_DIR` with
`PYTHONPATH=research`. They do not select a replacement primary.

Original 242's unsupported nested SiLU AD failure remains
[separate](../uniform-primitive-242-failed/README.md). The corrected independent
checker does not recertify older Float32 block derivatives. The full-response
mechanism, learned semantics and competent-verifier claims remain unestablished.
