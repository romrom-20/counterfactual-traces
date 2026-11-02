# 302: native first-source alias geometry

This is a retrospective conditional diagnostic on the eight originally fixed 301
witness task pairs (four layouts, two alias-mention orders, reverse=target=0).
Both native Boolean endpoints have already been observed. No fresh or single-False
prediction, learned circuit, semantic feature or priority claim is made.

## Fixed target and interventions

At first-block input, the Boolean query and self K/V are identical between the
two explicitly equivalent aliases for each fixed bit. Only two name-token prefix
K/V rows differ. Execute both False/True at both aliases for each arm:

- N: ordinary native calibration; first replay every saved 301 receiving/source
  trace and both scores exactly before any new interventions.
- SELF: reinstall identical first-source attention inputs.
- K/V/KV: replace only the two name-token keys, values, or both with the other
  alias's native first-layer prefix entries. Only the source attention output is
  installed. Its MLP, all later source/receiving operations and full head remain finite.
- H/S: replace keys by the native alias key difference projected into the joint
  nullspace of ALL False source-query heads sharing each KV bank, or its complement.
  The SVD rank cutoff is 1e-10 of the largest singular value. Shared-query grouping
  is required. The same prefix change is used for both bits.
- RH/RK/RV: seeded random hidden-key/full-key/value changes, norms matched per
  KV head and edited row to H/K/V respectively. RH also lies in the joint nullspace.
- P: keep the A/B key center fixed, rotate only the native key difference so its
  edited token's relative source phase becomes zero. This is a defined off-manifold
  difference-orientation intervention, not removal of all RoPE or a natural input.
- G: apply the same one-position rotary rotation to the source query and ALL
  source-visible keys, including self. Orthogonal common-rotation invariance is
  a numerical control, not a new transformer phenomenon.

K/V comparisons are the primary content test. Hidden/seen and phase tests are
separately fixed secondary hypotheses. Rotary phase is part of key geometry,
not a third mutually exclusive causal category. 301 already matched positions;
therefore changed position alone cannot explain its effect. A long-context
out-of-distribution account is not being asserted for these short contexts.

Let D[a,r] be an arm's True-minus-False displacement in recipient context r.
For each recipient separately, the conditional target is

    E_A = D[KV,A] - D[N,A]
    E_B = D[N,B] - D[KV,B].

The source post-block1 and next emitted-value targets equal the native alias
contrast, checked by exact donor restoration. Final receiving/verdict targets
are the effects of this ONE source read intervention, not the entire native
Boolean or whole-alias contrast. Signed effect predictions replace KV by each
candidate arm in the same formula. Both references are required; False hybrid
baselines are measured rather than silently replaced by native False.

## Frozen decisions and stopping rules

Each case passes a candidate only if BOTH reference contexts satisfy:
source post-block1, next emission, final receiving and verdict vector norms>=.001,
relative error<=.25, cosine>=.95; actual finite coupling effect>=.02 nat and
error<=min(.10 nat,.25*effect). All eight cases stay denominator, including
below-materiality cases. Also report source/emission-only success separately.

Population requires>=6/8,>=1/2 in each layout,>=3/4 in each mention order.
H_key additionally requires V and RK each<=2; H_value requires K and RV each<=2.
H_hidden requires S and RH each<=2. H_phase requires the fixed-center P source
alias-contrast norm<=25% of native in the same population, with native norm>=.001
and G numerical controls passed. H_phase is local necessity under a defined
orientation manipulation, never a claim about full native behavior.

Report signed attention/MLP source-vector ledgers and all target/error norms.
Projection fractions are not unique causal shares. No layer, alias, rank,
frequency-band or phase sweep follows a failure. Do not lower a materiality
guard or revise a sufficient-account threshold after the result.

## Numerical gate before attribution

First perform all 32 ordinary native calibrations and an independent CPU64 finite
audit. Only after that audit passes execute the other 352 forwards. Stop and retain
all output if any required calibration/qualification check fails.

All exact native replays, SELF replays, donor first-source/next-emission and
protected non-source attention outputs use zero tolerance. CPU probability/PV
and finite-head replay use 1e-4. Source-output projection uses max1e-3 AND relative
L2 1e-5. The first MLP and next-value maps use max1e-4; first MLP additionally
requires relativeL2 1e-5. Explicit Float32 residual additions must match exactly.
G source/final replay uses max1e-3 AND relativeL2 1e-5, scores1e-3. H/RH False
probabilities1e-4 and first-source states1e-4 qualify numerical invisibility.
CPU geometry identities use1e-9; matched-norm relative error1e-6. Original 1e-4
raw QK/output coordinate diagnostics are saved separately even if failed.

The CPU audit reconstructs finite QK/softmax/PV/output, RMSNorm/SiLU/gate-up/down,
next emission and final head. Its complex rotary construction and pinv projector
check the production split-half rotation and SVD projector. This is selected
primitive arithmetic in the same investigation, not an independent whole model
or derivative proof. 301's 96/576 raw failures and older disputed derivatives
remain unchanged. No derivatives are used in 302.

Unequal-GQA tiny fixtures use six query heads, three KV heads, two edited rows,
two sequence lengths and unequal hidden/intermediate widths. Coherent wrong-key
and wrong-value full continuations, single-query-only nullspaces, phase-norm
corruption and known K-only/V-only/joint effect tables must be rejected/qualified
before checkpoint execution. All scientific sources are committed before new
checkpoint execution and retained byte-for-byte in the run.

## Budget and limitations

384 full checkpoint forwards total:32 deliberate native calibrations and352
conditional continuations. The latter add352 full first-attention QK/softmax/PV
calls. Every call saves a source QK witness via384 further full QK products:
736 extra QK products overall,352 extra softmax/PV. Host audits and tiny fixture
forwards are additional. No efficiency advance is claimed.

Reuse the shared offline checkpoint and original 301 witness arrays. No model
copy, re-download or regeneration of pruned old runs. Reserve8GiB, cap all new
activation archives at128MiB, and do not save another MLP/checkpoint tensor copy.

Nullspace invisibility concerns what the False source query reads. Complete
False K/V observations retain those coordinates. Synthetic intervention
invariance is not an impossibility result for prediction with the known full
model/cache/edit. A successful local account would still need a frozen,
restricted-information/budget predictor evaluated before fresh native True
measurements to address the open prospective part of Question B.
