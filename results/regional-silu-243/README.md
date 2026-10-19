# 243: source versus receiving SiLU

The primary source-only account and the secondary receiver-only account each
pass **0/32** contrast-recovery cases. The negligible-interaction account also
fails. Median native receiving errors are **50.30% QQ**, **30.75% FQ**,
**41.79% QF** and **17.38% FF**. All arms still pass 0/32 native-response cases.

[Report](../../docs/analysis/243-source-receiver-silu-result.md) ·
[Frozen protocol](../../docs/experiments/243-source-receiver-silu.md) ·
[Native outcomes](native-outcomes.csv) · [Contrast recovery](contrast-recovery.csv) ·
[Signed effects at every block](signed-effects.csv) · [Audit](audit.json)

Code and protocol were frozen before intervention work. The global seal binds
every result before the separate saved-array audit. These examples were already
observed in 242b: this is retrospective, not a fresh test. All 160 forwards
complete, including 32 zero controls and 128 policy interventions. No new native
True forwards occur. Source trajectories do not depend on receiver edits; old
endpoints and all zero controls replay exactly.

Original243's missing zero-count metadata key stopped its scorer. The complete
corrected 243b audit retains every numerical operation/threshold and records
both scorer hashes. [Correction](../../docs/experiments/243b-reporting-correction.md),
[original failure](failure-audit.json) and logs remain available. No numerical
failure was reclassified by widening a tolerance.

With Python 3.12, verify source/input/output seals without model imports:

```sh
python -S research/boolean_coupling/reproduce_regional.py --verify-only --history --plan
```

After the [pinned MPS setup](../../research/README.md), reuse an existing 242b run:

```sh
python research/boolean_coupling/reproduce_regional.py --run --offline \
  --input-242 runs/existing-242b --out runs/my-regional-test
```

Omit `--input-242` to regenerate 242b first. All output directories must be new.
The wrapper uses the corrected scorer. Same-environment endpoint replays are
required exactly; different regenerated compressed NPZ bytes are bound to that
replication's own metadata rather than required to equal archived compression.
Rebuilding known literals is numerical replication, not new confirmation.

Large arrays remain local and are inventoried by size/SHA256. The original
False anchors and checkpoint are not duplicated. Article drafts stay local.
`description.json` and the figure are posthoc displays of audited outcomes,
without additional model work. No learned feature or solution to Question B
is established.
