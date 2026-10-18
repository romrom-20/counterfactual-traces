# Counterfactual Traces

**Can we predict a transformer's response to an input edit from one observed computation?**

This project investigates **Question B**: why changing an input Boolean generates a particular residual-stream displacement, and whether identifying the operations that generate and propagate it lets us forecast the native response from the **False state and known edit alone**.

The starting puzzle is small: an explicitly irrelevant Boolean in a supplied completion prefix changes a later verdict's conditional preference. The preferred answer can stay the same while its score margin moves substantially. We follow that dependency through the computation, keeping failed explanations alongside the ones supported by interventions.

**Status: open.** There are qualified local causal results. A general False-only forecast of the native response has not been demonstrated.

[Latest findings](docs/analysis/228-false-only-alignment-result.md) · [Reproduce](research/README.md) · [Experiments](docs/EXPERIMENTS.md) · [Research notes](docs/RESEARCH-NOTES.md) · [Open question](docs/QUESTION-B.md)

## Latest test: a useful hypothesis fails

Experiment 228 sealed 96 forecasts before measuring 32 new native True targets. Finite receiving MLP22/23 updates improve every case against both controls, but achieve **0/32** passes on the signed gate/up alignment criterion and **0/32** on the stronger final-response criterion. Median parent relative error remains **0.750**, against a **0.25** bound.

The first-block receiving displacement is exact in all 32 cases. The next attention update already has **14.3%** median relative error. This narrows the first forecast discrepancy without identifying its responsible primitive or solving the full native response. Read the [qualified negative result](docs/analysis/228-false-only-alignment-result.md).

## What the earlier causal tests established

1. **The local interaction controls the tested sign reversal.** At MLP23, experiment 225b separates a gate–up interaction K from a baseline shift B. K reverses the sign in 32/32 old contexts; B alone does so in 0/32. The stricter quantitative K-only account still fails overall.
2. **The next MLP generates a consequential magnitude interaction.** In 226b, removing newly generated mixing at MLP24 lowers the descriptive median margin gap from **0.246 to 0.065 nats** and passes both margin and vector criteria in **16/16** descriptive contexts. Final normalization alone is insufficient.
3. **Explaining the margin is easier than explaining the vector.** In 227b, removing the gate/up cross-product halves the margin gap in **16/16** descriptive cases but meets the combined vector/margin criterion in **0/16**. Only removing the combined mixed term passes both. Some normalization and later MLP contributions partly counteract the gap.

![Recorded downstream interaction and primitive-removal results](docs/figures/226b-227b-downstream-interaction.png)

The figure shows every descriptive case; bars are medians. These are previously observed, dependent synthetic examples of one pinned Qwen2.5-1.5B-Instruct checkpoint. The tested primitive decomposition is order-dependent. It does not identify learned features or a circuit.

**Numerical limits remain visible.** In 227b, 31/32 raw final-coordinate comparisons to a differently placed intervention fail the original bound, although the two-score bridge passes. Two raw ideal-write comparisons also fail. Arithmetic-accounted closure does not turn those failures into passes. Read the [full report](docs/analysis/227b-mlp24-mixed-primitives-result.md) before generalizing the result.

## Quick start

Verify every current release and the retained historical evidence without a model:

```sh
python research/boolean_coupling/reproduce_alignment.py --verify-only --history --plan
```

For an actual run, follow the [environment setup](research/README.md), then:

```sh
python research/boolean_coupling/reproduce_alignment.py --run --out runs/my-reproduction
```

This runs the 228 preflight, forecasts and qualification seal before native True measurement, then scores the unchanged criteria. The [earlier causal sequence](research/README.md#run-the-earlier-causal-sequence-from-public-inputs) remains separately reproducible. It requires the recorded Apple Silicon/MPS environment and a pinned public checkpoint; no API key or private activation archive is needed. Every output directory must be new. Logs, source seals, controls and failed checks stay with the run.

GitHub checks cover evidence integrity, Python syntax and lint. They do **not** execute the model experiments or independently replicate the scientific findings.

## Find your way around

- **[Research guide](research/README.md)** — setup, complete pipeline, code map and computation budget.
- **[Experiment index](docs/EXPERIMENTS.md)** — protocols, reports, audits and preserved failed attempts.
- **[Research notes](docs/RESEARCH-NOTES.md)** — working explanations, what changed, and what remains puzzling.
- **[Results](results/README.md)** — every case, numerical limits, original logs and release manifests.
- **[Historical evidence](evidence/HISTORY.md)** — earlier experiments, superseded claims and qualified diagnostics.
- **[Contributing](CONTRIBUTING.md)** — scientific standards, checks and how to publish a new result.

Developed with substantial AI assistance. The separate auditors are code paths in this investigation, not an independent research team. Grounding controls failed, so these results do not demonstrate disabling a competent verifier. Article drafts and large activation arrays are excluded. Code is released under the [MIT license](LICENSE); upstream checkpoints keep their own terms.
