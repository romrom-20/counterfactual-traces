# Experiment 225b: baseline versus local interaction

The local product interaction reverses the SiLU margin effect in 32/32 old MLP23 contexts; the baseline shift alone reverses it in 0/32. The stricter quantitative interaction-only account fails overall: 16/32 pass, all generic, and every descriptive case fails. This is a qualified retrospective diagnostic, not a False-only forecast or a learned-circuit identification.

Read the [result report](../../docs/analysis/225b-baseline-interaction-result.md), [first-person post draft](../../docs/drafts/lesswrong-context-dependent-corrections.md), and [reproduction guide](../../research/boolean_coupling/README.md).

- `audit.json`, `qualification-audit.json` and `qualification-seal.json` record numerical qualification, frozen decisions and the pre-intervention seal.
- `cases.csv` retains the 64 site/context rows, including failed account criteria. `behavior.csv` records 32 native paired preferences.
- `qualification/` and `interventions/` retain per-case metadata and controls; `logs/` contains the original execution logs.
- `frozen-225/`, `frozen-225b/` and `failed-225/` preserve the original failed token-validation preflight and separately named correction.
- `local-raw-inventory.json` records sizes and hashes of 65 local raw NPZ files. They and checkpoint weights are excluded from Git; the standalone runner regenerates arrays from the public prompts and pinned model.
- `manifest.json` seals the published study files. It excludes itself and the repository landing page; Git versions those separately.

All 32 contexts were previously observed and share synthetic templates. Passing numerical qualification does not mean the scientific account passed. Question B and the original whole-block derivative disagreement remain unresolved. Historical artifacts are preserved separately in [the evidence bundle](../../evidence/README.md), including disqualified conclusions that this release does not endorse.

Hash verification loads no model and is not a model replication:

```sh
python research/boolean_coupling/reproduce.py --verify-only --history --plan
```
