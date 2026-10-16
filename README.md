# Counterfactual Traces

**Can we predict how a transformer will respond to an input change from one observed computation?**

This repository investigates **Question B**: why changing an input Boolean generates a particular state-dependent residual displacement, and whether the nonlinear operations that create and propagate it let us predict the resulting verdict change from the **False state and the known edit alone**.

The starting puzzle is small: an explicitly irrelevant Boolean in a supplied completion prefix changes a later verdict's conditional preference. The project follows that dependency through the computation, recording explanations that survived direct tests and guesses that failed.

**Status: open.** There is a qualified local causal result, but no successful general False-only forecast of the native response.

[Latest experiment](docs/analysis/225b-baseline-interaction-result.md) · [Run it](research/boolean_coupling/README.md) · [The open question](docs/QUESTION-B.md)

## What the latest test establishes

At MLP23 in a pinned Qwen2.5-1.5B-Instruct checkpoint, the effect of the same SiLU correction changes sign across two computational backgrounds. Experiment 225b separates a local gate–up interaction **K** from the baseline shift **B** presented to the downstream network:

- **K alone reverses the sign in 32/32 existing contexts; B alone does so in 0/32.** Removing K while retaining B restores the positive effect.
- **The stricter quantitative K-only account fails overall:** 16/32 cases pass, all generic. Every descriptive-schema case fails; the baseline still matters for the magnitude.
- **The native origin is unresolved.** These are retrospective interventions on old, dependent templates, not fresh-context prediction, a learned-circuit identification, or evidence that a competent verifier can be disabled.

The [report](docs/analysis/225b-baseline-interaction-result.md) retains the numerical qualifications, matched controls and failed criteria.

## Reproduce or inspect

Check the published evidence without loading a model:

```sh
python research/boolean_coupling/reproduce.py --verify-only --history --plan
```

Run the model experiment in a new output directory, after following the [environment instructions](research/boolean_coupling/README.md):

```sh
python research/boolean_coupling/reproduce.py --run --out runs/my-reproduction
```

The validated model backend is Apple Silicon/MPS with Float32/eager execution. The run uses exact public prompt pairs and a pinned checkpoint; it regenerates its inputs without private activation archives. A complete run includes 64 ordinary model computations and 704 finite suffix continuations.

GitHub checks validate file integrity, Python syntax and lint. They do **not** rerun the MPS experiment or turn a failed scientific hypothesis into a passing one.

## Find your way around

- **[research/boolean_coupling](research/boolean_coupling/README.md)** — the standalone experiment, exact prompts, pinned settings and reproduction entry point.
- **[docs](docs/QUESTION-B.md)** — the research question, registered protocols, result report and figures.
- **[results/baseline-interaction-225b](results/baseline-interaction-225b/README.md)** — every case, numerical audit, original log, failed preflight and release manifest.
- **[evidence](evidence/HISTORY.md)** — the earlier research record, including superseded claims and the preserved tree before this cleanup.

Unrelated probe, sentiment and benchmark pipelines have been retired from the active tree. Their code and results remain in the [historical snapshot](evidence/HISTORY.md). New work should help answer Question B; see [contributing](CONTRIBUTING.md).

Developed with substantial AI assistance. The separate auditors are code paths within this investigation, not an independent research team. Article drafts are kept locally and are excluded from this repository.
