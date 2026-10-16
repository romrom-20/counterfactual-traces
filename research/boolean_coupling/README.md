# Boolean coupling: separate a local interaction from its downstream baseline

This is the standalone entry point for experiment 225b. It regenerates the False/True states from the published 32 synthetic prompt pairs and a pinned checkpoint. It needs no private activation archive or API key.

The question is specific: when changing the product computation reverses the verdict effect of finite SiLU, does that reversal come from the extra local interaction K, the product-induced baseline shift B, or their combination? The original four conditions are reproduced and audited before the two missing causal conditions and matched random controls execute. All later public blocks and the final head remain finite. This is a retrospective native-input diagnostic, not a False-only forecast or a learned-circuit claim.

Read the [result and numerical boundaries](../../docs/analysis/225b-baseline-interaction-result.md), and [registered protocol](../../docs/experiments/225b-baseline-interaction-separation.md).

## Read the code

- [core225b.py](core225b.py): real model loading, captures, explicit receiving-node surgeries, local SiLU/product derivative checks, and finite suffix execution.
- [run225b.py](run225b.py): qualification and intervention phases, protected source/prefix rows, deterministic controls and exact execution counts.
- [audit225b.py](audit225b.py): separate NumPy equations, cast/projection/head checks, qualification seal and frozen action criteria.
- [cases.json](cases.json): exact False/True token sequences, readable messages, changed-token position, prompt-condition labels and original metadata hashes.
- [settings.json](settings.json): checkpoint revision, token IDs, sites and tolerances.
- [reproduce.py](reproduce.py): public verification or complete model sequence; [present.py](present.py) draws the actual audited outcomes.

`core.py`, `run.py` and `audit.py` preserve the original 225 preflight. It stopped before any forward because it checked bare-word tokens;895/830 actually encode leading-space ` false`/` true`. Use the 225b entry points. That correction changes neither prompts nor intervention criteria. The unsuccessful preflight remains documented rather than silently overwritten.

## Verify the published snapshot without a model

From the repository root:

```sh
python research/boolean_coupling/reproduce.py --verify-only --history --plan
```

This checks published file hashes and all 2,867 retained historical preservation files (one unpublished draft is withheld) inside the compressed evidence bundle. It does not load a checkpoint, check omitted raw activations, or count as model replication. The complete historical archive includes disqualified attempts and their corrections; see [evidence notes](../../evidence/README.md) before interpreting them.

## Run the experiment from scratch

The validated backend is an Apple Silicon Mac with MPS, Float32/eager execution and Python 3.12.14. The original environment is recorded in [requirements-recorded.txt](requirements-recorded.txt). Cross-version/backend agreement has not been demonstrated. The 1.5B model and activations require several GB of memory and disk; the original work used a 24 GB Mac.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r research/boolean_coupling/requirements-recorded.txt
python research/boolean_coupling/reproduce.py --run --out runs/my-225b-reproduction
```

The first run may download the pinned public checkpoint. Add `--offline` if that exact revision is already cached. Output directories must be new; the runner refuses to overwrite previous evidence. This command executes:

1. 64 ordinary False/True computations,512 primitive-direction comparisons, and 256 corner continuations.
2. A separate qualification audit and seal.
3. 448 new-cell/control continuations, with no ordinary model observations.
4. A separate intervention audit and descriptive plots.

The total is 704 finite suffix continuations,3,872 remaining-block evaluations and 704 finite heads. Suffix calls are not full-model calls. Qualification comparisons include repeated directions and are not independent replications.

The new directory contains raw NPZ arrays, JSON checks, runtime/design hashes, qualification seal, every case and gate, and figures. Once the model runs, individual phases can be inspected using `PYTHONPATH=research python -m boolean_coupling.audit225b --help`. The public default needs no previous run. The original execution additionally used `--reference-root` to compare all 256 corners to the historical224b arrays; that reference is optional for a new standalone reproduction and is never described as newly executed when omitted.

## Scope

The contexts are32 previously observed, dependent synthetic examples. Random controls match product-space norms, not residual/readout norms. Native receiving inputs are supplied as an oracle; no forecast is sealed before new target states in this experiment. Earlier grounding failures and whole-block derivative disagreements remain unresolved. Original 224's failed coordinate bridge is preserved;225b does not repair it. The result concerns these defined interventions, not training origin, semantic features, deployment prevalence or a general solution to Question B.
