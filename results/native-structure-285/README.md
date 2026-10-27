# Native source prediction across prompt structures: Experiment 285

**Native prediction transfers: C passes 32/32 joint H1 cases, eight per layout.
The stronger native-repair claim fails, 15/32.** All forecasts and numerical
qualification were public before native measurements. The immutable
[pre-native seal](pre-native/forecast-seal.json) and [completed native audit](native-phase/native-audit.json)
are both retained.

[Readable result and limits](../../docs/analysis/285-native-structure.md) ·
[Frozen protocol](../../docs/experiments/285-native-structure.md) ·
[Reproduce](../../research/README.md#structural-native-source-test-285)

The test changes the causal prefix into table, user-rule, dialogue and XML layouts,
keeping facts fixed. It compares the unchanged False-coefficient cubic first-source
rule with two historical forecast-direction controls. Only the first-source SiLU
is approximated; attention, normalization, projections, product and all 27 later
source blocks/receivers remain finite. This is a substantial computation budget,
not a compact predictor or identified learned circuit.

![Native first-source transfer and the failed stronger account](../../docs/figures/native-structure-285/native-structure.png)

C's median native source-write error is 1.61%, versus roughly 7.4% for the donors.
Both donors pass all 32 local, receiving and verdict vector bounds. Their six H1
failures are solely margin errors above .10 nat, with dialogue only 4/8 versus
six required. Their 26/32 aggregate therefore does not meet the full transfer rule.
Matching C's forecast magnitude does not repair those six confidence errors.

H2 requires source-write and continued-effect repair beyond matched donor M,
including a .02-nat actual margin guard. Only 19/32 native effects are eligible;
exact T calibration passes only those 19. C predicts all 19 continued effects,
but four eligible cases fail its local repair bound, leaving 15/32 joint passes.
All thirteen ineligible cases remain failures. No threshold or denominator changed.

Within the recorded True/False candidates, even the False-write control Z agrees
with T's True preference in all 32 cases while failing H1 throughout. These are
not full-vocabulary decoded answers or correctness tests. The result concerns
residual/margin fidelity; it does not establish verifier competence or necessity
of a unique case-specific source direction.

The forecast audit passes 4,288 required checks and all 320 raw 1e-4 diagnostics.
The native audit passes 992 required checks and includes 32 unhooked ordinary
True replays. Scientific freeze: `813f1619b5175f875e836df05ff56a04f057d2ba`.
Public forecast commit: `17eafe8847b69568c467a36cec01bdb9d934bba9`.
Actual production: 224 forecast forwards in 88.86 s, 32 native in 22.98 s;
qualification and native audit took 15.41 s and 19.14 s. Audits and historical
bank generation are additional work. Original earlier numerical failures remain
unchanged; no complete Question B, efficiency or novelty claim follows.

`pre-native/` contains all forecast metadata, scores, numerical checks and bank
index. `native-phase/` contains every native case, endpoint metric, H1/H2 decision
and eligibility failure. Omitted-array inventories bind exact hashes/sizes;
raw activations and weights remain local. `publication-notes.json` preserves a
publication-only device-string check correction. The final manifest binds both
phases and the report/figure/helper provenance.

```sh
python -S research/boolean_coupling/verify285.py --phase forecast
python -S research/boolean_coupling/verify285.py --phase complete
```

These verify recorded evidence and frozen decisions without model imports. They
are not independent checkpoint replication. Private articles remain excluded.
