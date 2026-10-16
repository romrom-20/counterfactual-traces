# Context-dependent Boolean coupling

An explicitly irrelevant Boolean in a supplied output prefix changes a later verdict's conditional preference. This research follows that dependency into a small transformer's residual computation.

The concrete mechanistic result is conditional: at MLP23, the same finite-minus-affine SiLU correction raises the verdict margin with an affine product and lowers it with a finite product, across32 old contexts and both normalization settings. MLP22 behaves differently. A follow-up finds that the local interaction alone reverses the sign in 32/32 contexts, while the baseline alone reverses it in 0/32. The stricter quantitative interaction-only account fails overall:16/32 pass, all generic; the baseline still matters for descriptive-schema magnitudes.

**Question B remains open:** predicting the native residual and behavioral response from one observed False computation has not been established. These are retrospective interventions on one small checkpoint and dependent synthetic prompts, with numerical and grounding limitations preserved.

[Read the post draft](docs/drafts/lesswrong-context-dependent-corrections.md) · [Latest result](docs/analysis/225b-baseline-interaction-result.md) · [Run from scratch](research/boolean_coupling/README.md) · [Historical evidence](evidence/README.md)

## Start here

- The **post draft** explains the observation, the causal distinction and the measured boundary in ordinary language.
- The **standalone 225b implementation** contains readable model operations, published prompts, a pinned checkpoint, exact tolerances, separate qualification/intervention audits and matched controls. It needs no historical activation archive or API key.
- The **result report and JSON audits** retain every case and failed criterion. Counts over old templates are not independent replications.
- The **compressed evidence archive** preserves 2,868 earlier files, including failures and superseded hypotheses, without putting thousands of historical files in the main reading path. It excludes multi-gigabyte raw activations and model weights.

## Verify without a model

```sh
python research/boolean_coupling/reproduce.py --verify-only --history --plan
```

## Execute the standalone experiment

The recorded runtime is Python 3.12.14 on Apple Silicon/MPS, Float32/eager. See the [compute and dependency instructions](research/boolean_coupling/README.md).

```sh
python research/boolean_coupling/reproduce.py --run --out runs/my-reproduction
```

The command regenerates states from the public prompt pairs, audits and seals the four comparison conditions, runs the separated interventions, audits every outcome and draws the figures. It refuses to overwrite an existing run. Add `--offline` when the pinned checkpoint is cached. The total is 64 ordinary model calls and 704 finite suffix continuations; these are different units of computation.

## Research boundaries

The study does not identify a learned semantic circuit, establish novelty priority, or demonstrate that a competent verifier can be disabled. The original whole-block derivative discrepancy and failed 224 coordinate bridge remain documented. The new experiment uses local primitive checks and finite suffixes; those do not retrospectively recertify old operators.

Developed with an AI coding assistant. Separate auditors are independent code paths within this project, not a second research team. The LessWrong text is a draft; it has not been posted to LessWrong.

Earlier activation-probe and sentiment studies remain in Git history and the existing results folders. The [previous landing page](https://github.com/romrom-20/Training-Jev/blob/6ae7418a0a7ee9aed07cc5d689d855393c96d6b9/README.md) records that earlier project trajectory.
