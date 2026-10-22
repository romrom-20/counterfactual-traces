# MLP normalization origin: all evidence, including failed accounts

Start with the [scientific report](../../docs/analysis/261-264-mlp-normalization-origin.md).
These studies use four old 258 contexts, all 28 layers. No fresh examples,
learned feature, speedup or complete Question B solution is claimed.

- `mlp-input-origin-261/`: original **invalid** metadata-path attempt, restored
  original-record hashes and preserved erroneous records. Its source/auditor
  remain available for inspection. **Never execute `input261.py`.**
- `mlp-input-origin-261b/`: corrected 28-forward run, exact reference controls,
  complete metadata and strict projection-audit failure.
- `mlp-input-replay-261c/`: all 1568 publicFloat32 projection replays are exact;
  27 CPU64 comparisons still fail their original bound.
- `mlp-input-qualified-261c/`: preserved failed absolute displacement audit.
- `mlp-input-bounded-261d/`: separately bounded public interpretation; all
  20,792 declared checks, 27 primal and 8 displacement raw failures retained.
  Tangent0/4, cubic2/4, random0/4, finite reference4/4.
- `gate-curvature-factors-262/`: all local factors and 8848 checks, zero model
  calls. Primary baseline-only curvature account fails its fixed90% criterion;
  secondary energy approximation is not promoted to primary success.
- `normalization-branches-energy-263/`: all 44 forwards, 32 outcome rows,
  gate/up interaction, rounding controls and 35,480 declared checks. Primary
  energy arm2/4, gate-only0/4, up-only0/4, full-factorization4/4. All2464 public
  projections replay exactly;41 raw CPU64 failures remain failed.
- `normalization-component-interaction-264/`: original final contrasts and
  incomplete/mislabeled layerwise output, **not the all-layer result**.
- `normalization-component-interaction-264b/`: count/index correction, all28
  blocks,1344 signed-vector-statistic rows,448 squared-error records,4 finite
  margin contrasts and452 checks. No model calls or new native measurements.

`raw-array-inventory.json` records hashes/sizes and regeneration paths for
omitted activations, weights and signed vectors. Source files and protocols
are sealed by `manifest.json`; scientific code bytes were frozen separately
before their respective computations. The provenance in the figure directory
binds its plotting source and every input CSV. Private drafts and local handoff
notes are excluded.

Run `python -S research/boolean_coupling/verify_normalization.py` from the repo
root for standard-library release verification. Follow
[model reproduction](../../research/README.md#mlp-normalization-origin-and-branch-interventions)
for tensor/model work. Integrity checks are not independent replication.
