# Carried-state response: Experiment 283

**Qualified retrospective result: the fixed P rule passes all five joint criteria
in 32/32 old contexts and all 16 dependent schema pairs.** It predicts attention's
modulation of a specified missing cubic-product correction at B's own carried
input. It does not identify the complete native source generator or solve
general Question B.

[Readable report](../../docs/analysis/283-carried-state-response.md) ·
[Frozen protocol](../../docs/experiments/283-carried-state-response.md) ·
[Reproduce](../../research/README.md#carried-state-response-283284) ·
[Method](../../research/boolean_coupling/carried283.py).

![Every-case result and the paired local target](../../docs/figures/carried-response/283/carried-response.png)

## What the result does and does not discriminate

P passes the conditional C1-K local-increment test 32/32; gate-only H, up-only J
and all three norm-matched random increments pass 0/32. P's median increment
error is 12.29%, with cosine .99251. Its paired local schema contrast passes 16/16.
H also passes that paired contrast, so the contrast alone does not establish the
need for both terms. The earlier carry-only primary remains failed.

A posthoc same-schema cross-literal donor's saved P-K forecast vector also passes
the old conditional increment tolerance 32/32 without rescaling (17.62% median
error, cosine .98438); cross-schema donors pass 0/32. The analogous fresh comparison
passes 30/32, while the own rule remains more precise. These donors hold assignment,
order and target fixed. Matched random failure therefore does not establish
case-specific necessity. This is a zero-forward saved-vector comparison, not a
new primary, fresh prediction, donor installation or downstream native test.
[Transport summary and every comparison](transport-posthoc/summary.json) preserves
128 donor rows that reuse 32 recipients/16 dependent pairs, not 128 independent cases.
[Posthoc reproduction commands](../../research/README.md#posthoc-carried-write-portability)
write new destinations and preserve published records.

Every one of K/P/H/J/G0/G1/G2 already passes the original native-response criterion
32/32. Thus native accuracy alone does not select P's local account. The strict
incremental continued-effect and removal accounts pass 0/32: none of the C1-K
margin effects reaches the frozen .02-nat materiality guard, including full
calibration. Ineligible cases remain failures; no native necessity is claimed.
Native T has a different first-source computation and is not a positive control
for the conditional C1-K increment; exact C1/U are.

## Evidence files

- `design.json`, `status.json`, `runtime.json` and `weight-hashes.json` record the
  frozen design, completed producer, runtime and fixed weights.
- `audit.json`, `checks.csv`, `coefficients.csv`, `derivatives.csv`, `raw.csv` and
  `heads.csv` retain numerical decisions and original-bound raw failures.
- `native.csv`, `local.csv`, `effect.csv`, `incremental.csv`,
  `incremental_effect.csv`, `paired.csv`, `joint.csv` and `removal.csv` preserve
  every arm and denominator. Paired targets concern source-write coordinates;
  unequal-length whole-token receiving contrasts remain unavailable.
- `ledger.csv` records signed finite gate/up intervention effects,
  installation/nonadditivity and prediction remainders. These are vector
  projections, not causal fractions or identified learned features.
- `transport-posthoc/` records the later donor comparison, its own-rule reference,
  original thresholds and provenance, separately from the frozen primary.
- `cases/` contains metadata and logical-array hashes. `raw-array-inventory.json`
  records the excluded arrays and their exact bytes/hashes. Identical-array
  aliases are lossless and bind their historical dependencies.
- `execution-notes.json` and `log-provenance.json` preserve execution scope,
  the initial failed unsealed CPU assertion and the passing replacement fixture.
  Published logs retain scientific progress/results and identify their source
  hashes; they are not complete private terminal transcripts.

The original audit passes **5,408 required checks**, including all 32 staged
derivative anchors with independent Double automatic differentiation and two
finite differences. **Nine of 224 raw up comparisons still fail their original
1e-4 tolerance.** They remain failures under the predeclared public Float32 and
dimension-derived rounding-envelope scope. This does not certify rounded
execution's Jacobian, old whole-block derivatives or the GPU reduction algorithm.

The source/protocol freeze is `463ece70233d28e2f3bc6d2ae6d08e4e62880817`.
Production ran 448 full forwards, 32 extra finite corners, 64 local down maps,
32 normalizer JVPs and 64 CPU64 gate/up matrix-vector products. All 26 later
source blocks and all receiving operations remain finite. New raw arrays total
31,051,716 bytes; audits and synthetic fixtures are additional work. No efficiency
improvement over the causal-cache baseline is established.

Repository-only verification needs neither model packages nor activation arrays:

```sh
python -S research/boolean_coupling/verify_carried_results.py --experiment 283
```

Integrity verification is not an independent model replication. These old
dependent contexts use one small checkpoint. Fresh support is a separate
[separately sealed and audited Experiment 284](../../docs/analysis/284-carried-state-response.md).
Historical grounding failures, numerical caveats and novelty limits remain.
Weights, large arrays, private articles and local working context are excluded.
