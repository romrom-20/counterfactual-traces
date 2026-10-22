# Counterfactual Traces

**Can we predict a transformer's response to an input edit from one observed computation?**

This project investigates **Question B**: why a Boolean input edit generates a
particular residual-stream displacement, and whether the operations generating
and propagating it let us forecast native residual and verdict from the **False
computation, known edit and model weights**.

**Current answer: a bounded predictive success; the broader mechanism is open.**
A frozen rule using cubic receiver-query normalization and finite SiLU at a
ranked **1%** of receiving activation coordinates passes the full-response
criterion on **32/32 new literal pairs**. The remaining activation coordinates
use a quadratic approximation. Substantial source/K/V/MLP computation stays
finite; this is not a learned circuit or a demonstrated speedup.

[Start here](docs/READING-GUIDE.md) · [Latest diagnostic](docs/analysis/261-264-mlp-normalization-origin.md) ·
[Question B](docs/QUESTION-B.md) · [Reproduce](research/README.md) ·
[Experiment index](docs/EXPERIMENTS.md) · [Research notes](docs/RESEARCH-NOTES.md)

## New to this project?

The [reading guide](docs/READING-GUIDE.md) defines the task, residual displacement,
query feedback, finite versus approximate computation, and how to inspect a claim.
Follow its six-test path before the full chronological experiment index.

![How the receiving computation changes across layers](docs/figures/reader/changing-reader.png)

## What nonlinear MLP normalization adds

A deeper retrospective intervention finds that replacing MLP input
normalization by its False tangent raises whole-displacement error from 2.06%
to 32.08% on four old contexts. Restoring normalization to only the gate branch
or only the up branch is insufficient. A simple energy correction gets the
vectors within tolerance but misses the verdict margin in two cases. The
smaller normalization term has a different final vector effect on different
backgrounds. This narrows the generation mechanism; it does not establish a
learned circuit or solve general Question B. Original numerical failures remain
failed, with the public Float32 scope explicitly qualified.
[All interventions, signed effects and limitations](docs/analysis/261-264-mlp-normalization-origin.md).

## What the latest fresh forecast establishes

An earlier single-site SiLU repair worked locally but could not predict the
native response from its inaccurate incoming state. After obtaining an accurate
query-feedback rule, we apply the same target-blind activation rule at every
receiving MLP, using only each forecast's own evolving state.

The primary ranked 1% rule has **2.13% median whole-displacement error**,
**1.96% verdict-displacement error** and **.00678-nat margin error**. It passes
8/8 cases in every schema/order stratum. Quadratic SiLU everywhere leaves
**40.79%** median whole error and passes 0/32. A row-count-matched random 1%
restoration leaves **39.53%** and passes 0/32. The finite-SiLU reference passes 32/32
at 1.05% whole error. Every forecast was publicly sealed before native True
measurement. [All cases, method and budget](docs/analysis/258-receiver-mlp-result.md).

![All32 fresh literal pairs, including controls](docs/figures/258/258-receiver-mlp.png)

The mask ranks an analytical SiLU remainder bound times current up magnitude
and down-column norm. It uses no native True gates, final readout or observed
remainder. A separate **retrospective** comparison finds that dropping the
bound weakens local capture; dropping column norm changes little at the 24
saved witness inputs. Cubic gate-displacement magnitude also performs similarly
locally, so the full bound is not shown uniquely necessary. These simpler rules
have not been validated as fresh global forecasts.
[Score-factor diagnostic](docs/analysis/259-silu-score-factors-result.md).
A subsequent [energy diagnostic](docs/analysis/260-silu-energy-concentration-result.md)
finds the original mask contains at least 90% of individual omitted-write energy
in 20/24 whole-input witnesses, before cross-neuron cancellation. Verdict-local
capture is less uniform (16/24). This does not predict the upstream gate changes.

## The numerical boundary matters

The original 258 CPU64 whole-down-projection audit **failed**. Before any native
measurement, a separately frozen 258b qualification required exact public
Float32 replay at the original full-sequence shape, while retaining every
failed CPU64 check at its original tolerance. All other component and behavioral
criteria stayed fixed. The qualified final audit passes 20,016 checks; raw
whole-down and projection/rotary/QK coordinate failures remain published as
failures. This is not independent whole-model real-arithmetic reconstruction,
and it does not resolve older whole-block derivative disagreements.
[Numerical revision](docs/experiments/258b-matched-projection-qualification.md).

## How the evidence led here

- **An unchanged answer can hide dependence.** An explicitly irrelevant Boolean
  in a supplied prefix changes the later Boolean's conditional score margin.
  Failed grounding controls prevent a competent-verifier-shutdown claim.
- **Local accounts did not compose.** Several primitive interventions explained
  a local vector or scalar effect but failed final residual and verdict tests.
  Accurate incoming states matter. [Local SiLU boundary](docs/analysis/240-target-blind-silu-result.md).
- **Continued source reads matter.** Defined native source-value clamps remove
  most signed verdict response, but static receiving attention still fails to
  predict it. [Source communication](docs/analysis/250b-native-source-communication-result.md).
- **Receiver query feedback is consequential.** Predicting its normalization
  curvature first closes a declared full-response forecast with correction
  passes, then with one local cubic-rule pass. [Single-pass result](docs/analysis/256-local-query-curvature-result.md).
- **A small adaptive beyond-quadratic SiLU budget can compose with that rule.**
  The latest test moves beyond a conditional one-site repair, while retaining
  the other declared finite operations. [258](docs/analysis/258-receiver-mlp-result.md).

These experiments use one pinned small Qwen checkpoint and dependent synthetic
templates. New literals are useful held-out cases, not independent tasks or
cross-model replication. The learned source/receiving computation, necessity
of the remaining finite operations, generalization and useful compute savings
remain open. No novelty-priority or learned-feature claim is established.

## Quick start

Verify every release, source seal and retained historical record without a model:

```sh
python -S research/boolean_coupling/reproduce_receiver_mlp.py --verify-only --history --plan
python -S research/boolean_coupling/verify_reader_guide.py
```

For an actual numerical reproduction, follow the [environment setup](research/README.md), then:

```sh
python research/boolean_coupling/reproduce_receiver_mlp.py --run --offline --out runs/my-258-reproduction
```

The wrapper executes the published258 rule, the explicit258b qualification,
pre-native sealing, native/control measurements and separate final audit. It
never pushes Git. Repeating published literals is numerical replication, not
fresh confirmation. It requires the pinned Apple Silicon/MPS environment;
use a new output directory. Individual scientific stages ran in this release;
a separate full wrapper replay has not been executed. GitHub CI checks evidence
integrity, syntax and lint, **not** the model experiments.

[Research guide](research/README.md) gives exact commands, component budgets,
retrospective259 prerequisites, previous wrappers and numerical qualifications.

## Repository guide

- [Question B](docs/QUESTION-B.md): current answer and remaining gaps.
- [Experiment index](docs/EXPERIMENTS.md): protocols, reports and preserved failures.
- [Research notes](docs/RESEARCH-NOTES.md): explanations that changed and why.
- [Results](results/README.md): every case, controls, audits, logs and array inventories.
- [Historical evidence](evidence/HISTORY.md): qualified older diagnostics and invalidated claims.
- [Contributing](CONTRIBUTING.md): numerical and publication standards.

Developed with substantial AI assistance. Separate auditors are code paths in
this investigation, not an independent research team. Unpublished article drafts,
large activation arrays and private working context are excluded. Older PR
discussions remain in a separate private archive. Code has an [MIT license](LICENSE);
upstream checkpoints retain their own terms.
