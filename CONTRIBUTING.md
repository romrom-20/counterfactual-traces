# Contributing to Question B

The active question is whether an identified nonlinear computation can predict the native residual and verdict response to a known input edit, using one observed False-state computation. Useful contributions test a specific part of that claim: a qualified primitive intervention, a falsifying control, or a frozen forecast evaluated on fresh contexts.

Start with [Question B](docs/QUESTION-B.md), the [latest diagnostic](docs/analysis/261-264-mlp-normalization-origin.md), the [research notes](docs/RESEARCH-NOTES.md), and the [reproduction guide](research/README.md).

## Preserve the record

The published 225/225b and 226/227/228/229/230/231/232/233/234/235/236 scientific sources, protocols and numerical evidence are byte-sealed. Do not reformat them, overwrite failed attempts, or update their manifest to accommodate a changed computation. Put a scientific correction in a separately named experiment, freeze it before execution, retain the original failure, and keep the numerical bounds unchanged unless a new protocol explicitly justifies different bounds. Publication-only presentation revisions must list the original and current hashes separately.

Narrow lint exceptions preserve existing style in these sealed files. Syntax and undefined-name checks still apply. New code receives the full lint rules.

## Check a change

```sh
python -m pip install ruff==0.16.8
python -m ruff check .
python -S research/boolean_coupling/reproduce_curvature.py --verify-only --history --plan
python -m compileall -q research/boolean_coupling results/baseline-interaction-225b/frozen-225/research/boolean_coupling results/baseline-interaction-225b/frozen-225b/research/boolean_coupling
```

These are repository checks, not a model replication. Follow the experiment guide for an actual MPS model run and choose a fresh output directory.

## Publish each completed finding

Keep the repository current as research progresses. A completed study should include its code, protocol, exact inputs or regeneration path, runtime, source seals, every case, numerical audits, failures, and a result report. State which question it answers and which claims remain unsupported.

Update the [experiment index](docs/EXPERIMENTS.md), [research notes](docs/RESEARCH-NOTES.md), [results index](results/README.md), and relevant reproduction instructions. Update the homepage when the current scientific conclusion changes. Include figures generated from the recorded data; preserve the plotting code and raw-array inventory.

Verify the release and inspect the staged files before committing. Keep article drafts, checkpoint weights and activation arrays excluded. Push the reviewed update and check GitHub CI. Successful CI establishes repository integrity and syntax/lint checks, not a successful scientific hypothesis or independent model replication.

The clean code repository is public. Older PR records remain in a separate private archive. Updating research evidence does not authorize publishing article drafts or pushing legacy unpublished history.

## Make a scientific claim reviewable

State what is observed, what is supplied by an oracle, and what is predicted. Record the checkpoint revision, exact inputs, runtime, tolerances, computation counts and all failures. Keep final readouts finite when the intervention protocol requires it. Separate changes to a diagnostic from changes to the native residual or verdict.

Fresh-context predictions must be sealed before observing the native True trajectories. Explain the computation budget: executing the full counterfactual network is a reference computation rather than the compact account sought here. A local sign regime or product identity does not identify a learned feature or circuit.

Large activation arrays, model weights and private data stay out of Git. Preserve their sizes and hashes, and provide a way to regenerate public-input experiments. The project license does not relicense upstream checkpoints or datasets.

## Keep unpublished writing local

Do not commit article drafts or package them inside evidence archives. Keep them under the ignored `.context/` directory or `docs/drafts/`. Scientific code, reports, figures and audits can be shared independently of the article.
