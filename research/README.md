# Reproduce the current research

<a id="joint-gate-292"></a>

## Full-native joint-gate forecast (292)

[Result and limits](../docs/analysis/292-joint-gate-forecast.md) ·
[Frozen protocol](../docs/experiments/292-joint-gate-forecast.md) ·
[All evidence](../results/joint-gate-292/README.md)

From the repository root, verify public records without model packages:

```sh
python -S research/boolean_coupling/verify292.py
```

For actual replication, use the pinned model/runtime setup described below,
the original 102 source files and a fresh output directory. The original
production run used Float32 MPS eager execution. Preserve historical records.

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research
python -m boolean_coupling.preflight292 --out runs/tiny-292
python -m boolean_coupling.run292 forecast --run runs/replication-292
python -m boolean_coupling.audit292 forecast --run runs/replication-292
python -m boolean_coupling.run292 seal --run runs/replication-292
```

The gate intentionally stops before native True until the NEW global seal is
public. Publish the replication's exact seal at the frozen `PUBLIC_SEAL` path
on a dedicated replication branch/fork; do not merge it over this historical
292 record. Commit/push the exact forecast evidence and obtain that branch's
full commit SHA. The native gate verifies local HEAD, remote HEAD and public
seal bytes. Then, with that exact receipt:

```sh
python -m boolean_coupling.run292 native --run runs/replication-292 \
  --public-commit FULL_PUSHED_REPLICATION_SHA --public-branch REPLICATION_BRANCH
python -m boolean_coupling.audit292 native --run runs/replication-292
```

The public forecast case JSON and omitted-array inventory specify shapes,
dtypes and archive hashes; arrays/weights are local, not downloaded by CI.
Independent replication must regenerate them. Original source parameters and
bounds remain frozen. `resident292` is a resource-only execution path for the
recorded disk stops; its first-case replay requires the preserved original
archive. Normal replication with adequate headroom uses the commands above.
The CPU preflight's `--help` lists its fresh-output arguments; its synthetic
native phase cannot authorize checkpoint measurement.

<a id="attention-and-receiving-response-289290"></a>

## Attention and receiving response (289–290)

[Failed attention account](../docs/analysis/289-attention-interface.md) ·
[Receiving gated MLP result](../docs/analysis/290-receiving-nonlinearity.md) ·
[289 protocol](../docs/experiments/289-attention-interface.md) ·
[290 protocol](../docs/experiments/290-receiving-nonlinearity.md)

These are retrospective conditional response tests on all 32 old contexts.
The attention interface alone fails 0/32. With that interface supplied,
receiving gated MLP response passes 32/32 while normalizer response alone
passes 0/32. Only 16 remaining finite contrasts are material. Edited source,
attention and head states are supplied by finite forecast replay; this is not
a compact native predictor or a learned circuit.

Check the public records without model packages, from the repository root:

```sh
python -S research/boolean_coupling/verify289.py
python -S research/boolean_coupling/verify290.py
```

For actual model replication, first regenerate the qualified 285–288 references
and their tiny CPU fixtures using the preceding studies below. The 287 head
archive and its hash receipt are reused. Keep the original failed 287 audit
and its separately named qualification. Use fresh output directories:

```sh
RESPONSE_REFERENCE=/path/to/qualified/native-structure-285
RESPONSE_PREVIOUS=/path/to/qualified/tangent-routing-288
RESPONSE_HEAD=/path/to/qualified/source-receiver-readout-287
RESPONSE_CPU_REFERENCE=/path/to/synthetic-285/complete
RESPONSE_CPU_PREVIOUS=/path/to/synthetic-288/complete
RESPONSE_CPU_HEAD=/path/to/synthetic-287/complete
RESPONSE_OUTPUTS=/path/to/new/output-parent
python -m research.boolean_coupling.preflight289 --out "$RESPONSE_OUTPUTS/cpu-289" --reference-285-fixture "$RESPONSE_CPU_REFERENCE" --previous-288-fixture "$RESPONSE_CPU_PREVIOUS" --head-287-fixture "$RESPONSE_CPU_HEAD"
python -m research.boolean_coupling.run289 --reference-285 "$RESPONSE_REFERENCE" --previous-288 "$RESPONSE_PREVIOUS" --head-287 "$RESPONSE_HEAD" --out "$RESPONSE_OUTPUTS/interface-289"
python -m research.boolean_coupling.audit289 --run "$RESPONSE_OUTPUTS/interface-289" --reference-285 "$RESPONSE_REFERENCE" --previous-288 "$RESPONSE_PREVIOUS" --head-287 "$RESPONSE_HEAD"
python -m research.boolean_coupling.preflight290 --out "$RESPONSE_OUTPUTS/cpu-290" --reference-285-fixture "$RESPONSE_CPU_REFERENCE" --previous-289-fixture "$RESPONSE_OUTPUTS/cpu-289/complete" --head-287-fixture "$RESPONSE_CPU_HEAD"
python -m research.boolean_coupling.run290 --reference-285 "$RESPONSE_REFERENCE" --previous-289 "$RESPONSE_OUTPUTS/interface-289" --head-287 "$RESPONSE_HEAD" --out "$RESPONSE_OUTPUTS/receiving-290"
python -m research.boolean_coupling.audit290 --run "$RESPONSE_OUTPUTS/receiving-290" --reference-285 "$RESPONSE_REFERENCE" --previous-289 "$RESPONSE_OUTPUTS/interface-289" --head-287 "$RESPONSE_HEAD"
```

All enumerated sources must be committed before checkpoint execution. Installed
source and checkpoint hashes must match the pinned references. Both producers
require 384MiB initial and 256MiB per-case free space, with a 96MiB archive cap.
No ordinary True call or True gradient is made. All finite source/receiving
work, reference passes, full-vocabulary heads, primitive and block VJPs count.
See each status file and report for the full diagnostic budget.

Required gates pass 21,216/9,280 records. Eight new narrow coordinate failures
remain failed under the declared cached replay bounds; earlier failures remain.
FDs qualify actual finite modified functions. Analytic norm/attention probes
cover selected native cached-shape derivatives, not an independent full Jacobian.
The gated MLP family is not a separate validation of every SiLU/product coordinate.

Posthoc `publish289`/`publish290` export scalar records and omitted-array
inventories, excluding weights, activation vectors and private articles.
`describe289`/`describe290 --out /new/figure-directory` regenerate all-case plots
and refuse existing directories. Their `publish... --seal` commands bind the
final reports, figures, helper code and scientific sources after presentation.

## Source and receiving response maps (288)

[Result](../docs/analysis/288-source-receiver-tangent-routing.md) ·
[Protocol](../docs/experiments/288-source-receiver-tangent-routing.md) ·
[Evidence](../results/tangent-routing-288/README.md)

This retrospective eight-corner diagnostic distinguishes source recurrence,
receiving/transmission and final-head response maps. Every arm is an actual
finite recentered function with the same edited zero-displacement trajectory.
Both primary sufficient accounts fail 0/32, as do the three secondary accounts.
The required gate passes4,384 checks; seven narrow finite-coordinate diagnostics
remain failed at1e-4 under the separately declared .002 cached replay bound.
The original287 failed gate is unchanged. General Question B remains open.

Dependency-free public record check, from the repository root:

```sh
python -S research/boolean_coupling/verify288.py
```

For model replication, first regenerate the qualified285 forecasts and the
287/287b boundary-qualified references using the sections below. Retain the
original287 audit failure; the287b status must be `qualified_boundary_only`.
The small CPU references likewise come from the preceding preflights. New
output directories are required. The pinned packages/checkpoint and installed
source hashes must match. These commands run from the repository root:

```sh
ROUTING_REFERENCE=/path/to/qualified/native-structure-285
ROUTING_PREVIOUS=/path/to/source-receiver-readout-287
ROUTING_QUALIFICATION=/path/to/boundary-qualification-287b
ROUTING_CPU_REFERENCE=/path/to/synthetic-285/complete
ROUTING_CPU_PREVIOUS=/path/to/synthetic-287/complete
ROUTING_CPU_QUALIFICATION=/path/to/synthetic-boundary-287b
ROUTING_OUTPUTS=/path/to/new/output-parent
python -m research.boolean_coupling.preflight288 --out "$ROUTING_OUTPUTS/cpu-288" --reference-285-fixture "$ROUTING_CPU_REFERENCE" --previous-287-fixture "$ROUTING_CPU_PREVIOUS" --qualification-287b-fixture "$ROUTING_CPU_QUALIFICATION"
python -m research.boolean_coupling.run288 --reference-285 "$ROUTING_REFERENCE" --previous-287 "$ROUTING_PREVIOUS" --qualification-287b "$ROUTING_QUALIFICATION" --out "$ROUTING_OUTPUTS/routing-288"
python -m research.boolean_coupling.audit288 --run "$ROUTING_OUTPUTS/routing-288" --reference-285 "$ROUTING_REFERENCE" --previous-287 "$ROUTING_PREVIOUS" --qualification-287b "$ROUTING_QUALIFICATION"
```

The source files must be committed before checkpoint execution; the producer
checks Git bytes and checkpoint hashes. Initial/per-case reserve guards are
640/384MiB, with a128MiB case-array cap. One False observation per case supplies
prefix KV; no True gradient or new True call is made. All finite source and
receiving work, backward passes and full-vocabulary heads count. See status.json
and the report for the substantial diagnostic budget; there is no speedup claim.

`publish288` exports scalar evidence and an omitted-array inventory, never
weights, activations or private articles. `describe288 --out /new/figure-dir`
regenerates all-case plots from those records and refuses to overwrite an
existing directory. `publish288 --seal` binds the final report, plot provenance,
export inventory and all78 scientific sources after presentation is complete.

## Source and receiving initial states (287)

[Result](../docs/analysis/287-source-receiver-readout.md) ·
[Original protocol](../docs/experiments/287-source-receiver-readout.md) ·
[Posthoc boundary qualification](../docs/experiments/287b-boundary-qualification.md) ·
[Evidence](../results/source-receiver-readout-287/README.md)

This is a retrospective factorial on all 32 old 285/286 contexts, with source and
other incoming rows independently swapped after block 1. All later source and
receiving operations remain finite and adaptive. Source-only recreates the defined
sensitivity shift; it does not recover complete native residual fidelity.

The original audit exits nonzero and preserves one failed intermediate-adjoint
check before writing any scientific table. The separately named 287b analysis
qualifies ONLY the actual boundary gradients at the original tolerance, retains
that failed gate and compares both gradient estimates. On a repeat, inspect the
original audit before invoking the qualification: it may reject other failures.
Do not assume a later result or use it to weaken the original check.

Repository-only verification requires the standard library and full Git history:

```sh
python -S research/boolean_coupling/verify287.py
```

First regenerate the qualified 285, 286 and 286b references using the preceding
experiment instructions below. A rerun needs its own forecast receipt when that
stage requires one; the original receipt cannot certify new calls. Use pinned
Python 3.12, Float32 MPS/eager, cached checkpoint and NEW output directories.
The original 70-source freeze is ff01b72, narrower 73-source freeze 98f58d5.

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research
FACTOR_OUTPUTS=/path/to/new/outputs
FACTOR_REFERENCE=/path/to/qualified/native-structure-285
FACTOR_READOUT=/path/to/qualified/readout-286
FACTOR_FULL=/path/to/qualified/full-shape-286b
FACTOR_CPU_REFERENCE=/path/to/synthetic-285/complete
FACTOR_CPU_READOUT=/path/to/synthetic-286/complete
FACTOR_CPU_FULL=/path/to/synthetic-286b/complete
mkdir -p "$FACTOR_OUTPUTS"
python -m boolean_coupling.preflight287 --out "$FACTOR_OUTPUTS/cpu-287" --reference-285-fixture "$FACTOR_CPU_REFERENCE" --readout-286-fixture "$FACTOR_CPU_READOUT" --qualifier-286b-fixture "$FACTOR_CPU_FULL"
python -m boolean_coupling.run287 --reference-285 "$FACTOR_REFERENCE" --readout-286 "$FACTOR_READOUT" --qualifier-286b "$FACTOR_FULL" --out "$FACTOR_OUTPUTS/factorial-287"
python -m boolean_coupling.audit287 --run "$FACTOR_OUTPUTS/factorial-287" --reference-285 "$FACTOR_REFERENCE" --readout-286 "$FACTOR_READOUT" --qualifier-286b "$FACTOR_FULL"
```

After inspecting and preserving that audit, run the separate boundary check:

```sh
python -m boolean_coupling.preflight287b --fixture "$FACTOR_OUTPUTS/cpu-287/complete" --out "$FACTOR_OUTPUTS/cpu-287b"
python -m boolean_coupling.qualify287b --run "$FACTOR_OUTPUTS/factorial-287" --reference-285 "$FACTOR_REFERENCE" --out "$FACTOR_OUTPUTS/boundary-287b"
```

287 uses 32 ordinary False captures, 32 first blocks, 1,792 cached suffix forwards,
128 cached backwards, 3,456 full-shape block forwards, 3,456 reverse block
recomputations/VJPs and 128 head VJPs. 287b makes no model calls. Retain the original
768 MiB initial / 512 MiB per-case guards and 64 MiB case-array cap. This is not
an efficiency result. The initial zero-case disk stop and unchanged retry are
preserved alongside all numerical failures.

`publish287` exports scalar records and inventories; it never exports vectors or
article drafts. `describe287 --out /path/to/new/figures` plots the exported scalar
package with no model calls and refuses to overwrite prior renderings. `publish287
--seal` binds final records, original scientific sources and separate posthoc
helpers. CI recomputes scalar/vector-statistic arithmetic, not omitted arrays.

## State-dependent readout (286)

[Explanation](../docs/analysis/286-state-dependent-readout.md) ·
[Frozen rule](../docs/experiments/286-false-readout.md) ·
[Full-shape qualifier](../docs/experiments/286b-full-shape-adjoints.md) ·
[All records](../results/false-readout-286/README.md)

This is a retrospective scalar diagnostic on the existing 32 structural inputs.
The original False source gradient misses all six large donor margin errors;
the gradient at the False-derived M forecast predicts them. This does not reduce
the finite network budget or solve general Question B. No native True gradient
is computed. Old native arrays are replay references, not prospective information.

Repository-only verification uses the standard library and full Git history:

```sh
python -S research/boolean_coupling/verify286.py
```

For a numerical repeat, first regenerate the qualified 285 arrays below. The
original public pre-native receipt belongs to that original execution; a rerun
needs its own receipt. Use pinned Python3.12, Float32 MPS/eager and the cached
checkpoint, NEW output directories and the unchanged sources. The tiny CPU
fixtures come from `preflight285`; 286 uses four unequal-length sentinels. The
actual initial development run reused an already qualified fixture after its
regeneration hit the inherited disk guard. That failure remains recorded.

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research
READOUT_OUTPUTS=/path/to/new/outputs
READOUT_REFERENCE=/path/to/qualified/native-structure-285
READOUT_CPU_REFERENCE=/path/to/synthetic-285/complete
mkdir -p "$READOUT_OUTPUTS"
python -m boolean_coupling.preflight286 --out "$READOUT_OUTPUTS/cpu-286" --reference-285-fixture "$READOUT_CPU_REFERENCE"
python -m boolean_coupling.run286 --reference-285 "$READOUT_REFERENCE" --out "$READOUT_OUTPUTS/readout-286"
python -m boolean_coupling.audit286 --run "$READOUT_OUTPUTS/readout-286" --reference-285 "$READOUT_REFERENCE"
python -m boolean_coupling.preflight286b --out "$READOUT_OUTPUTS/cpu-286b" --reference-285-fixture "$READOUT_CPU_REFERENCE" --readout-286-fixture "$READOUT_OUTPUTS/cpu-286/complete"
python -m boolean_coupling.qualify286b --reference-285 "$READOUT_REFERENCE" --readout-286 "$READOUT_OUTPUTS/readout-286" --out "$READOUT_OUTPUTS/full-shape-286b"
python -m boolean_coupling.audit286b --run "$READOUT_OUTPUTS/full-shape-286b" --reference-285 "$READOUT_REFERENCE" --readout-286 "$READOUT_OUTPUTS/readout-286"
```

286 spends32 ordinary False observations,32 first blocks,1,664 suffix forwards
and64 backwards. The separate numerical control adds32 False observations,
32 first blocks,1,728 full-shape block forwards,1,728 reverse block recomputations/
VJPs and64 head VJPs. Both stages require768MiB initial/512MiB per-case headroom
and cap their own case arrays at32MiB. They do not relax older experiment guards.

All192 original cached-coordinate1e-4 diagnostics remain failed; the predeclared
.002 cached replay bound passes. The full-shape control independently passes
its declared1e-4 source-adjoint comparison, using the same library primitives.
It is not a complete Jacobian qualification. Public inventories bind omitted
arrays/head weights; Git contains only scalar scores, metadata, checks and figures.

The posthoc `describe286` plotter takes `--run`, `--qualifier` and a new `--out`;
it never calls a model. `publish286` prepares a new scalar-only package and seals
the finished records/presentation with `--seal`. A numerical repetition on these
known inputs is not fresh scientific confirmation.

## Structural native source test (285)

[Result and failed stronger account](../docs/analysis/285-native-structure.md) ·
[Frozen protocol](../docs/experiments/285-native-structure.md) ·
[Evidence and original public receipt](../results/native-structure-285/README.md)

The unchanged first-source rule passes 32/32 native/source-write cases across four
new causal-prefix layouts. The stronger C-minus-matched-donor repair claim fails
15/32; every ineligible case remains in the denominator. Donor vectors still pass
all local/vector bounds, and six confidence errors determine their native failures.
This does not replace the 27 later source blocks or finite receiving computation.

Repository-only verification requires Python stdlib and Git history, not model
packages or raw arrays:

```sh
python -S research/boolean_coupling/verify285.py --phase forecast
python -S research/boolean_coupling/verify285.py --phase complete
```

Use a full-history clone for the complete verifier: it checks exact seal bytes in
the original pre-native public commit. For actual model repetition, first regenerate
279's qualified forecast arrays using the later source-activation instructions.
The frozen donor bank reads only their False/C forecast writes. Use pinned Python
3.12/Float32 MPS/eager, the cached checkpoint, and NEW output directories. Keep
host headroom beyond the frozen 1-GiB initial/512-MiB per-case guards; large arrays
and the pinned checkpoint are not included in Git.

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research
STRUCTURE_BANK=/path/to/regenerated/source-activation-forecast-279
STRUCTURE_OUTPUTS=/path/to/new/outputs
mkdir -p "$STRUCTURE_OUTPUTS"
python -m boolean_coupling.preflight285 --out "$STRUCTURE_OUTPUTS/synthetic-285"
python -m boolean_coupling.run285 forecast --run "$STRUCTURE_OUTPUTS/structural-repeat-285" --reference-279 "$STRUCTURE_BANK"
python -m boolean_coupling.audit285 forecast --run "$STRUCTURE_OUTPUTS/structural-repeat-285" --reference-279 "$STRUCTURE_BANK"
```

These stages actually ran. The synthetic pipeline includes unequal lengths,
variable source positions, all endpoints, unhooked True equivalence and six
premature/corrupted-receipt rejections before any forward. Tiny scientific outcomes
need not pass; all software/numerical checks must pass. The real producer used
224 full forwards; forecast audit 4,288 required checks plus 320 raw diagnostics
all passed. Historical bank production and qualification are additional work.

The original production CLI deliberately enforces a public forecast receipt for
all real runs. A repetition of published inputs is **not fresh confirmation** and
must label itself accordingly. Its new seal has new execution times/hashes, so
the original receipt cannot authorize it. Publish the repetition's exact seal at
`results/native-structure-285/pre-native/forecast-seal.json` on a separate research
branch before invoking native measurement; use a clean receipt destination rather
than overwriting this repository's original sealed records. The publication helper
is write-once and refuses different existing bytes. An independent worktree rooted
at scientific freeze `813f161` can hold a repetition's own receipt, with the
posthoc delivery helpers copied from the completed release. There is currently no
real-checkpoint local-only bypass. Exact native commands, once that new receipt
is public, are:

```sh
STRUCTURE_PUBLIC_COMMIT=YOUR_PUBLISHED_REPETITION_COMMIT
STRUCTURE_PUBLIC_BRANCH=YOUR_REPETITION_BRANCH
python -m boolean_coupling.run285 native --run "$STRUCTURE_OUTPUTS/structural-repeat-285" --reference-279 "$STRUCTURE_BANK" --public-commit "$STRUCTURE_PUBLIC_COMMIT" --public-branch "$STRUCTURE_PUBLIC_BRANCH"
python -m boolean_coupling.audit285 native --run "$STRUCTURE_OUTPUTS/structural-repeat-285" --reference-279 "$STRUCTURE_BANK"
```

Original native production was 32 full forwards, plus 32 ordinary unhooked True
audit replays. All 992 native numerical checks pass. The audit retains the failed
H2 materiality and local-repair cases; successful numerical validation does not
mean both scientific hypotheses passed. No source/Jacobian tolerance was changed.
`describe285.py --run NEW_RUN --out NEW_FIGURE_DIR` creates the posthoc all-case
figure from qualified records; `publish285.py` packages phase records and omitted
array inventories. Every destination is new; private articles stay local.


## Carried-state response (283/284)

[283 qualified retrospective result](../docs/analysis/283-carried-state-response.md) ·
[284 sealed fresh result](../docs/analysis/284-carried-state-response.md) ·
[283 protocol](../docs/experiments/283-carried-state-response.md) ·
[284 protocol](../docs/experiments/284-sealed-carried-response.md) ·
[284b cache-only continuation](../docs/experiments/284b-native-cache-continuation.md)

Experiment 283 predicts a specified missing-product response at B's own carried input. Its
joint primary passes 32/32 old contexts; 282's carry-only primary remains failed.
Experiment 284 applies the unchanged rule to 32 new literals with every forecast and its
qualification public before any native True. Its unchanged primary passes 32/32,
all strata 8/8; original native audit and 284b continuation wrapper pass. The
original 17-case stop and both zero-forward retries remain preserved. Strict
incremental/removal behavioral accounts remain 0/32 under the frozen .02-nat guard.
Repository-only verification needs no model imports or arrays:

```sh
python -S research/boolean_coupling/verify_carried_forecasts.py
python -S research/boolean_coupling/verify_carried_results.py --experiment 283
python -S research/boolean_coupling/verify_carried_results.py --experiment 284
```

For Experiment 283, first regenerate qualified 282c/281d arrays and the unequal-token 282c CPU
fixture in the following sections. Use pinned Python 3.12/Float32 MPS/eager and
the cached checkpoint. Choose NEW destinations under an existing parent. Allow
model/process headroom beyond the frozen 640 MiB initial/512 MiB per-case reserves;
these guards do not guarantee sufficient memory or prevent host disk pressure.
Never delete or recompress old evidence to make a run proceed.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
CARRIED_ORIGIN=/path/to/regenerated/source2-origin-repeat-282
CARRIED_PRODUCTS=/path/to/regenerated/source2-products-repeat-281d
CARRIED_CPU=/path/to/synthetic-unequal-282c
CARRIED_OUTPUTS=/path/to/new/outputs
mkdir -p "$CARRIED_OUTPUTS"
python -m boolean_coupling.preflight283 --fixture "$CARRIED_CPU" --out "$CARRIED_OUTPUTS/synthetic-carried-283"
python -m boolean_coupling.run283 --reference-282 "$CARRIED_ORIGIN" --reference-281d "$CARRIED_PRODUCTS" --out "$CARRIED_OUTPUTS/carried-state-repeat-283"
python -m boolean_coupling.audit283 --reference-282 "$CARRIED_ORIGIN" --reference-281d "$CARRIED_PRODUCTS" --run "$CARRIED_OUTPUTS/carried-state-repeat-283"
```

Those original scientific stages actually ran: 448 full forwards, 32 extra finite
corners, 64 local down maps, 32 normalizer JVPs and 64 CPU64 gate/up matrix-vector
products; 5,408 required checks pass. Nine of 224 raw up comparisons retain their
failed original 1e-4 bounds under the predeclared public Float32/rounding-envelope
scope. All 26 later source and all receiving operations remain finite. New arrays
occupy 31.1 MB plus resolved historical dependencies. No speedup over 272 or old
whole-block-Jacobian recertification is established.

Experiment 284 regenerates directly from published cases/checkpoint, without 283's activation
archive. Run the complete unequal-token CPU fixture first. It uses a clearly
synthetic receipt and tests premature-native and wrong-seal rejection. Published
literals require explicit repetition mode: numerical repetition is not fresh
confirmation and must not fabricate a public receipt.

```sh
python -m boolean_coupling.preflight284 --fixture "$CARRIED_CPU" --out "$CARRIED_OUTPUTS/synthetic-carried-284"
python -m boolean_coupling.run284 forecast --repeat-known-literals --run "$CARRIED_OUTPUTS/carried-state-repeat-284"
python -m boolean_coupling.audit284 forecast --run "$CARRIED_OUTPUTS/carried-state-repeat-284"
python -m boolean_coupling.run284 native --repeat-known-literals --run "$CARRIED_OUTPUTS/carried-state-repeat-284"
python -m boolean_coupling.audit284 native --run "$CARRIED_OUTPUTS/carried-state-repeat-284"
```

Require the forecast audit and global seal before native measurement. Original
forecasts 320 and 4,672 required checks completed; 11/224 raw up comparisons remain
failed at 1e-4. Their exact seal was public at
`c2439037904cf6d6c9eb4cb91c38301f4f12f6f5` before C1/U/N/T. The commands
above deliberately record a local repetition instead of that original public
receipt. Original forecast arrays occupy 54.7 MB; combined new-array cap 96 MiB.
The original native stage stopped after 68 forwards/17 cases at its 512 MiB reserve.
A complete local native repetition needs 128 producer and 32 ordinary True audit
forwards; no second full real-checkpoint pipeline has been executed here.

### Qualified resource continuation (284b)

This path applies to the **actual retained 17-case stop**, preserving its original
seal, first-native public receipt, failure, log and arrays. Do not manufacture a
partial run to invoke it. The unequal-token CPU continuation must pass first:

```sh
python -m boolean_coupling.preflight284b --fixture "$CARRIED_OUTPUTS/synthetic-carried-284" --checkpoint "$CARRIED_CPU/tiny-checkpoint" --out "$CARRIED_OUTPUTS/synthetic-carried-284b"
```

The original CPU fixture passed 17 replay + 15 remaining cases, all 32 NPZ and JSON
files exactly equal, and the unchanged original native audit. This is not a test
of real MPS cache release. The separately frozen real continuation command is:

```sh
CARRIED_PARTIAL=/path/to/original-284-with-17-sealed-native-cases
CARRIED_STOP_LOG=/path/to/that-original-native-failure.log
python -m boolean_coupling.continue284b --original "$CARRIED_PARTIAL" --failure-log "$CARRIED_STOP_LOG" --out "$CARRIED_OUTPUTS/carried-state-continued-284b"
python -m boolean_coupling.audit284b --run "$CARRIED_OUTPUTS/carried-state-continued-284b"
```

The original non-synthetic path also verifies the actual published receipt and
remote source binding; it is not a substitute for explicit local repetition.
It releases only unused MPS allocator cache after each forward and before each
case. All 17 old NPZ **and JSON hashes** must replay exactly before any of the 15
new measurements. Any mismatch stops continuation. The wrapper then calls
unchanged `audit284.native`, including 32 ordinary True replays without editing
hooks. Predictions, endpoint code, numerical scope, 512 MiB reserve and 96 MiB cap
are unchanged.

The first two real 284b attempts stopped after model loading, before **any** replay
or new forward; they add no measurements. A third attempt completed all 17 exact
NPZ+JSON replays before 15 new cases in 40.80 seconds. The unchanged original native audit
and 284b wrapper then passed in 16.92 seconds: 1,024 required checks, 128 finite head maps
and 32 ordinary True unhooked replay forwards. Actual native producer work is
68 original + 68 replay + 60 new = 196 calls for 32 unique cases, including 49
ordinary True producer calls; 32 additional unhooked audit forwards are separate.
All original failures and the first-native public receipt remain preserved.

[283 every-case figure](../docs/figures/carried-response/283/carried-response.png) ·
[284 every-case fresh figure](../docs/figures/carried-response/284/carried-response.png).
Private articles, weights, large arrays and local context remain excluded.
Successful correction prediction would not establish learned features, material
margin necessity, novelty priority, efficiency or complete Question B closure.

### Posthoc carried-write portability

The posthoc same-schema cross-literal donor passes the local tolerance in 32/32
old and 30/32 fresh cases without rescaling, at 17.62% and 21.33% median error.
The own fresh rule is more precise at 12.00%. Cross-schema donors pass 0/32 in
both studies. Donors hold assignment, order and target fixed; case-specific
vectors are not shown necessary. These saved-vector comparisons use zero model
forwards, make no causal installation or native-response comparison, and do not
change the frozen primary. Each condition reuses 32 recipients/16 dependent
pairs; the 128 donor rows are not independent examples.
[Old transport records](../results/carried-state-response-283/transport-posthoc/summary.json) ·
[Fresh transport records](../results/carried-state-forecast-284/transport-posthoc/summary.json).

After regenerating and auditing the arrays above, run the
[saved-array diagnostic](boolean_coupling/diagnose_carried_transport.py) into NEW
directories. It performs both raw and recipient-forecast-norm-matched comparisons;
no native target norm supplies the rescaling. A complete ordinary 284 repetition
is accepted. If the run instead carries a 284b continuation, that additional
qualification is also verified; no artificial runtime failure is required.

```sh
python -m boolean_coupling.diagnose_carried_transport --experiment 283 --run "$CARRIED_OUTPUTS/carried-state-repeat-283" --reference "$CARRIED_ORIGIN" --upstream "$CARRIED_PRODUCTS" --out "$CARRIED_OUTPUTS/carried-transport-283"
python -m boolean_coupling.diagnose_carried_transport --experiment 284 --run "$CARRIED_OUTPUTS/carried-state-repeat-284" --out "$CARRIED_OUTPUTS/carried-transport-284"
```

Both diagnostics actually ran with new output destinations. Every summary and
donor/own-reference CSV byte matched the initial diagnostic; provenance records
bind the executed helper and input hashes. These are posthoc tests on existing
arrays, not additional model replication or a new frozen primary.

## Source2 origin (282)

[Result and limitations](../docs/analysis/282-source2-gate-origin.md) ·
[Original protocol](../docs/experiments/282-source2-gate-origin.md) ·
[282b numerical scope](../docs/experiments/282b-public-up-qualification.md) ·
[282c paired accounting](../docs/experiments/282c-paired-secondary-accounting.md) ·
[All outcomes and failures](../results/source2-origin-282/README.md)

Repository verification uses neither model packages nor arrays:

```sh
python -S research/boolean_coupling/verify_source_origin.py
```

First regenerate281d and its qualified281e audit using the next section. Use
the pinned Python3.12/Float32 MPS/eager environment and cached checkpoint. Choose
new outputs under an existing parent and allow at least1GiB free after inputs,
plus model/process headroom. Original guards remain500MiB initial/448MiB per case;
they are not a guarantee against system disk changes. New arrays occupy33.7MB,
losslessly aliasing byte-identical input/local observations. Never delete or
recompress old evidence. All three synthetic checks below must pass first.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
ORIGIN_INPUTS=/path/to/regenerated/source2-products-repeat-281d
ORIGIN_OUTPUTS=/path/to/new/outputs
mkdir -p "$ORIGIN_OUTPUTS"
python -m boolean_coupling.preflight282 --out "$ORIGIN_OUTPUTS/synthetic-origin-282"
python -m boolean_coupling.preflight282b --fixture "$ORIGIN_OUTPUTS/synthetic-origin-282" --out "$ORIGIN_OUTPUTS/synthetic-audit-282b"
python -m boolean_coupling.preflight282c --out "$ORIGIN_OUTPUTS/synthetic-unequal-282c"
python -m boolean_coupling.run282 --reference-281d "$ORIGIN_INPUTS" --out "$ORIGIN_OUTPUTS/source2-origin-repeat-282"
python -m boolean_coupling.audit282 --reference-281d "$ORIGIN_INPUTS" --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282"
```

On the recorded environment the final command fails at CPU64_U_up's original
1e-4 absolute bound. Preserve it. Do not forge a failure to activate a qualifier,
or continue if the failure differs. The helpers bind actual failures and source
hashes; they do not silently repair the original audit or raise its tolerance.
After inspecting that expected failure:

```sh
python -m boolean_coupling.diagnose_input_bridges282 --reference-281d "$ORIGIN_INPUTS" --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282" --out "$ORIGIN_OUTPUTS/origin-input-bridges"
python -m boolean_coupling.qualify282b --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282" --diagnostic "$ORIGIN_OUTPUTS/origin-input-bridges"
python -m boolean_coupling.audit282b --reference-281d "$ORIGIN_INPUTS" --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282"
```

282b's expected failure is the unequal-length secondary receiving comparison,
after all primitive checks. Preserve and inspect it before the separate282c
accounting qualification. Any other failure needs investigation.282c marks the
original whole-state secondary unavailable; it changes no primary criterion.

```sh
python -m boolean_coupling.qualify282c --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282"
python -m boolean_coupling.audit282c --reference-281d "$ORIGIN_INPUTS" --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282"
python -m boolean_coupling.diagnose_origin_geometry --reference-281d "$ORIGIN_INPUTS" --run "$ORIGIN_OUTPUTS/source2-origin-repeat-282" --out "$ORIGIN_OUTPUTS/origin-geometry-posthoc"
```

These CLIs actually ran. A cold retry of the frozen producer completed all32 original
predictions without changing the original producer, runtime policy or guards;
all23 saved arrays and metadata from the interrupted attempt reproduced exactly.
There is no282b/c model producer.480 completed-study full forwards plus345 failed-
attempt forwards total825. The study also uses128 finite local corner maps and
256 extra local down maps. Audits and CPU fixtures are additional. The initial
unsealed282b CPU name-shadowing failure and282b's shape failure remain recorded;
the final unequal-length fixture exercises the original producer and full audit.

Repeating published literals is a numerical repetition, not fresh confirmation.
Finite26 later source/all receiving operations and additional finite corner maps
remain. The posthoc geometry reads fixed weights/saved full calibration with no
new model forwards. At the time of282, a False-derived response around the carried
state was proposed rather than tested. The separately frozen283 result above now
tests that rule; it does not change282's failed primary. Article drafts,
weights, arrays and local continuity remain excluded.

## Source2 products (281d)

[Result and all limits](../docs/analysis/281-source2-products.md) ·
[Original protocol](../docs/experiments/281-source2-products.md) ·
[Qualified audit](../docs/experiments/281e-random-control-order.md) ·
[Every case and failed attempt](../results/source2-products-281d/README.md)

Repository-only verification requires neither model imports nor local arrays:

```sh
python -S research/boolean_coupling/verify_source2_products.py
```

For model repetition, first regenerate279 and280b using the following sections.
Use the pinned Python3.12/Float32 MPS/eager environment and cached checkpoint.
Choose a NEW destination under an existing parent. Allow at least700MiB free
after generating the input archives, in addition to weights/process headroom.
The lean new arrays occupy71.4MB; this does not reduce input-regeneration cost.
No original evidence should be deleted or recompressed.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
PRODUCT_INPUTS=/path/to/regenerated/source-composition-repeat-280b
PRODUCT_OUTPUTS=/path/to/new/outputs
mkdir -p "$PRODUCT_OUTPUTS"
python -m boolean_coupling.preflight281d --out "$PRODUCT_OUTPUTS/synthetic-producer-281d"
python -m boolean_coupling.run281d --reference-280b "$PRODUCT_INPUTS" --out "$PRODUCT_OUTPUTS/source2-products-repeat-281d"
python -m boolean_coupling.audit281 --run "$PRODUCT_OUTPUTS/source2-products-repeat-281d"
```

Require the synthetic preflight to pass before the model producer. The original
audit is expected to reproduce its retained exact random-write operation-order
failure. It must fail specifically at that check, with a zero bound and error
below1e-15; any other failure needs investigation. The helper below checks those
conditions, preserves the failure and seals the published qualification sources
before rerunning. It cannot overwrite an existing qualification or successful
audit. No tolerance is relaxed.

```sh
python -m boolean_coupling.qualify281e --run "$PRODUCT_OUTPUTS/source2-products-repeat-281d"
python -m boolean_coupling.audit281e --run "$PRODUCT_OUTPUTS/source2-products-repeat-281d"
python -m boolean_coupling.diagnose_product_geometry --run "$PRODUCT_OUTPUTS/source2-products-repeat-281d" --out "$PRODUCT_OUTPUTS/product-geometry-posthoc"
```

Omitting `--partial-281b` regenerates all32 known cases. The published execution
instead reused30 original sealed cases and required the orphan31st array to
regenerate exactly before case32; both cold32 and resumed30+2 paths passed the
CPU synthetic fixture with identical NPZ hashes. A separate full cold checkpoint
repetition has not run. The qualification helper passed a metadata-only fixture;
that is not another model replication. Fresh support requires new protocol,
contexts and globally sealed forecasts/qualification before native measurement.

The original281 bookkeeping failure,281b storage-cap failure,281c binding failure
and missed preflight gate remain public.281d changes only the binding/storage
continuation;281e changes only exact RNG reconstruction order.576 completed-study
forwards plus38 failed-attempt forwards ran, then public primitive/head audits.
All later26 source and all receiving operations remain finite. This is not a
whole-model speedup or learned circuit. Geometry/rectifier ledgers use saved
measured targets with zero model forwards; their concentration rankings are not
forecasts. Private writing, model weights and large arrays remain excluded.

## Source composition (280b)

[Report and limits](../docs/analysis/280-source-composition.md) ·
[Original protocol](../docs/experiments/280-source-composition.md) ·
[Storage qualification](../docs/experiments/280b-storage-continuation.md) ·
[Every case and failure](../results/source-composition-280b/README.md)

```sh
python -S research/boolean_coupling/verify_source_composition.py
```

For model repetition, first regenerate279 forecasts and run its published-literal
native helper in the next section. Use the pinned Python3.12/Float32 MPS/eager
checkpoint/environment. No private activation archive is downloaded. Choose a NEW
output and existing parent; allow at least2.3GiB free in addition to the regenerated
279 inputs, cached weights and process headroom. Never delete original evidence.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
COMPOSITION_INPUTS=/path/to/regenerated/source-activation-repeat-279
COMPOSITION_OUTPUTS=/path/to/new/outputs
mkdir -p "$COMPOSITION_OUTPUTS"
python -m boolean_coupling.run280b --reference-279 "$COMPOSITION_INPUTS" --out "$COMPOSITION_OUTPUTS/source-composition-repeat-280b"
python -m boolean_coupling.audit280 --run "$COMPOSITION_OUTPUTS/source-composition-repeat-280b"
```

Omitting `--partial-280` is an explicitly supported cold repetition of the known
32 literals; provenance fields record no reused partial arrays. It accepts279's
`repetition-observations` format as well as original native observations. The
original completed study instead supplied `--partial-280` to preserve28 completed
arrays and run only the last four. That continuation actually ran, followed by
the ORIGINAL unchanged audit280 on all32. Cold-regeneration CLI support has not
been separately executed as another full model repetition; it is not fresh support.

The original280 producer stopped at its768MiB disk reserve after28 cases. Its
1200MiB raw cap was also too small.280b freezes a storage-only1536MiB cap and512MiB
per-new-case reserve, reusing all28 arrays unchanged. The first280b launch rejected
its initial guard before model/output creation; only disposable workspace bytecode
was cleared before retry. Original failures and source seals remain public.

Total352 full forwards and1792 extra local product/down maps (896 also finite
SiLU), then explicit audits. All receiving/attention/norm/projections/products
remain finite. The all-source cubic primary and one local removal both fail0/32;
this is an old-context source-composition diagnostic, not a speedup or circuit.
Raw arrays1,422,656,845bytes stay local with public hashes. A storage cap is not a
guarantee of sufficient initial free space; allow raw output plus reserve/headroom.

The posthoc packager/plotter and `diagnose_source_gates.py` use audited data only.
The gate ledger additionally reads pinned checkpoint weights, with zero model
forwards, and uses native targets explicitly. It is not a prospective selector
or neuron intervention. Private writing and all weights/arrays remain excluded.


## Source activation forecast (279)

[Result and limits](../docs/analysis/279-source-activation-forecast.md) ·
[Protocol](../docs/experiments/279-sealed-source-activation.md) ·
[Every case and original public receipt](../results/source-activation-forecast-279/README.md)

Repository-only verification uses the standard library, without weights/arrays:

```sh
python -S research/boolean_coupling/verify_source_activation.py
```

For published-literal model repetition, use the pinned Python 3.12/Torch/
Transformers/NumPy, Float32 MPS/eager environment and cached checkpoint documented
below. All destination parents must exist and output directories must be NEW.
The 278 dependency here is public metadata only; no old private arrays are needed.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
ACTIVATION_OUTPUTS=/path/to/new/outputs
mkdir -p "$ACTIVATION_OUTPUTS"
python -m boolean_coupling.run279 forecast --reference-278 results/source-mlp-primitives-278 --run "$ACTIVATION_OUTPUTS/source-activation-repeat-279"
python -m boolean_coupling.audit279 forecast --run "$ACTIVATION_OUTPUTS/source-activation-repeat-279"
python -m boolean_coupling.repeat_source_activation --run "$ACTIVATION_OUTPUTS/source-activation-repeat-279"
```

The last helper creates a new local receipt, measures native True, and recomputes
original-native/local/effect/paired joint endpoints. It uses `repetition-*` outputs,
sets `fresh_confirmation=false`, and does not recertify the stronger sparse
correction account. It was tested by reusing the original forecasts and recomputing
32 native references; all native/joint counts reproduced. That helper test was
not a full forecast regeneration or new independent confirmation.

The original study ran `run279 forecast`, `audit279 forecast`, `run279 receipt`,
committed/pushed receipt 519f4e8, then `run279 native --proof ...` and
`audit279 native`, in that order. The immutable public pre-native directory cannot
be overwritten. `native` verifies the pushed receipt and its source/array seals.
For genuinely new support, freeze a separately named protocol/code/case set and
receipt location before checkpoint execution; these literals are now known.

The cubic primary passes 29/32 joint; all three below-materiality cases stay in
32. Quadratic also misses their tight effect margin bounds. Local-write and
original-native criteria pass 32/32 for both. Stronger sparse correction fails
0/32. First source attention/norm/gate/up/product/down and every later block remain
finite. Before the seal: 32 full False plus 288 forecast forwards, 32 extra first
attention calls, 32 source norm/gate/up maps, 128 padded masked SiLU maps and 224
source product/down maps; 32 native references follow. No whole-model speedup.

Forecast arrays total 81,675,881 bytes, native arrays 16,003,156 bytes. Reserve at
least 1280 MiB initially and 768 MiB per case, plus model/process headroom and audit
outputs. Original evidence is never overwritten or deleted. All 96 derivative
coefficient vectors, 672 original-bound raw comparisons and separate native
operand checks pass here; no old derivative/raw-logit failure is recertified.

Posthoc `package_source_activation.py` and `describe_source_activation.py` preserve
the metadata and figure provenance. The figure/eligibility calculation makes zero
model calls. NPZ, weights and unpublished article text remain excluded.

## Source MLP primitives (278)

[Intervention result](../docs/analysis/278-source-mlp-primitives.md) ·
[Protocol](../docs/experiments/278-source-mlp-primitives.md) ·
[Every corner and signed ledger](../results/source-mlp-primitives-278/README.md)

```sh
python -S research/boolean_coupling/verify_source_primitives.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
PRIMITIVE_INPUTS=/path/to/regenerated/runs
PRIMITIVE_OUTPUTS=/path/to/new/outputs
mkdir -p "$PRIMITIVE_OUTPUTS"
python -m boolean_coupling.run278 --reference-276 "$PRIMITIVE_INPUTS/first-source-mlp-gain-276" --out "$PRIMITIVE_OUTPUTS/source-mlp-primitives-278"
python -m boolean_coupling.audit278 --reference-276 "$PRIMITIVE_INPUTS/first-source-mlp-gain-276" --run "$PRIMITIVE_OUTPUTS/source-mlp-primitives-278"
```

Regenerate 276 using the next section and its upstream instructions. Both
scientific commands actually ran after freezing their sources. All 32 cases were
old; this is retrospective. The normalization-only primary and all nontrivial
sufficient accounts fail 0/32. A posthoc edge comparison of the frozen factorial
shows normalization helps only with finite multiplication. It is not a changed
primary, a learned feature or a new architectural principle.

736 full forwards and 768 extra full-shaped local MLP maps; all other source/
receiving operations and final heads remain finite. The 448 primitive directional
checks and 4,608 raw comparisons pass original bounds. Raw arrays total
375,665,311 bytes; reserve 1536 MiB initially and 768 MiB per case, plus regenerated
upstream arrays/model headroom. Public inventories bind all local arrays.

Posthoc `package_source_primitives.py` and `describe_source_primitives.py` publish
all-case evidence and scientific figures without new model calls. Additional
margin/paired ledgers and conditional edge comparisons are explicitly posthoc.
Neither this packaging nor the standard-library verifier is a model replication.


## First-source MLP (276/277)

[Result and boundaries](../docs/analysis/276-277-first-source-mlp.md) ·
[276 protocol](../docs/experiments/276-first-source-mlp-gain.md) ·
[277 protocol](../docs/experiments/277-sealed-first-source-mlp.md) ·
[Old records](../results/first-source-mlp-gain-276/README.md) ·
[Fresh records and receipt](../results/first-source-mlp-forecast-277/README.md)

Repository-only verification needs neither model packages nor activation arrays:

```sh
python -S research/boolean_coupling/verify_first_mlp.py
```

For model repetition use the pinned Python3.12/Torch/Transformers/NumPy,
Float32 MPS/eager setup and cached checkpoint from the earlier instructions.
Regenerate275 arrays with the next section; no private archive is downloaded.
All output parents must exist and destinations must be NEW. From repo root:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
MLP_INPUTS=/path/to/regenerated/runs
MLP_OUTPUTS=/path/to/new/outputs
mkdir -p "$MLP_OUTPUTS"
python -m boolean_coupling.run276 --reference-275 "$MLP_INPUTS/source-attention-reading-275" --out "$MLP_OUTPUTS/first-source-mlp-gain-276"
python -m boolean_coupling.audit276 --reference-275 "$MLP_INPUTS/source-attention-reading-275" --run "$MLP_OUTPUTS/first-source-mlp-gain-276"
python -m boolean_coupling.run277 forecast --reference-276 "$MLP_OUTPUTS/first-source-mlp-gain-276" --run "$MLP_OUTPUTS/first-source-mlp-repeat-277"
python -m boolean_coupling.audit277 forecast --run "$MLP_OUTPUTS/first-source-mlp-repeat-277"
python -m boolean_coupling.repeat_first_mlp --run "$MLP_OUTPUTS/first-source-mlp-repeat-277"
```

The last helper is explicitly a **published-literal numerical repetition**. It
makes a local all-case receipt after the pre-native audit, then measures ordinary
True and recomputes original-native/paired/joint endpoints. Its output is named
`repetition-*.json/csv`; `fresh_confirmation` is false. It does not rerun the
stronger causal-account audit or claim independent fresh support. This helper
was executed with a new local receipt of the original forecasts and recomputed
32 native references, reproducing every original/joint count. The original full
forecasts were reused for that helper check, not generated again as new evidence. It avoids
altering or overwriting the immutable original public pre-native receipt.

The original fresh277 study actually ran `run277 forecast`, `audit277 forecast`,
`run277 receipt`, committed/pushed that receipt, then `run277 native --proof ...`
and `audit277 native`, in that order. `receipt` deliberately refuses to overwrite
`results/first-source-mlp-forecast-277/pre-native`, which now exists in the release.
The original native CLI requires proof containing `public_commit` and
`forecast_seal_sha256`; it verifies the pushed branch before measuring True.
For genuinely new literals, freeze a separately named protocol/code/case set and
new receipt location before checkpoint computation; rerunning known literals is
not new confirmation. Do not edit frozen277 scientific files to change this history.

276 uses448 full forwards,128 extra first-MLP maps and64 CPU64 tangents.277 uses
320 full False/forecast forwards,32 extra first-attention calls,64 corner PV and
output projections,96 extra first-MLP maps and64 tangents before the seal;32 native
references follow. Every other model operation remains finite. There is no
speedup or compact whole-network rule here;272 supplies the causal-cache baseline.

Raw276 arrays total156,745,136bytes.277 forecasts total151,012,790bytes and native
arrays21,017,778bytes. Initial producer reserves1536MiB/1280MiB respectively;
per-case reserve768MiB; old evidence is never deleted. Additional model/process
headroom and upstream regenerated arrays are needed. Public inventories bind
all omitted arrays. Both stronger gain explanations remain failed,8/32 and6/32.
The128 local smooth derivative anchors pass separate JVP/FD/reverse checks; no
old whole-block derivative/raw-logit failure is retroactively passed.

Posthoc `package_first_mlp.py` and `describe_first_mlp.py` publish metadata and
generate figures from all-case CSVs, with zero new model calls. Original model
sources and article drafts are kept separate; no NPZ/checkpoint/article is committed.

## Source attention reading (275)

[Protocol](../docs/experiments/275-source-attention-reading.md) ·
[Result](../docs/analysis/275-source-attention-reading.md) ·
[All32 old cases](../results/source-attention-reading-275/README.md)

```sh
python -S research/boolean_coupling/verify_source_reading.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research python -m boolean_coupling.run275 --reference-273 /path/to/regenerated/native-value-interchange-273 --out runs/my-source-reading
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research python -m boolean_coupling.audit275 --reference-273 /path/to/regenerated/native-value-interchange-273 --run runs/my-source-reading
```

Both scientific CLIs actually ran after the source freeze. Regenerate273 arrays
using its section below; no private activation archive is required. Use the pinned
MPS/Float32/eager environment and cached checkpoint; output must be NEW. This
is an old-context native primitive diagnostic, not fresh confirmation. Only the
source query row is clamped; every MLP and receiving operation remains finite.
Full arms use False observation plus known edit and weights but execute whole
finite forwards. Conditional value-only arms additionally retain native source K.

416 full forwards, the four extra PV/output-projection categories and all
primitive audits are recorded. The32 NPZ files total1,015,184,122bytes, below the
1000MiB cap. Require2GiB initial and768MiB per-case free space, plus model/process
headroom. No old evidence is deleted or overwritten. Original CPU64 raw bounds
remain1e-4 (all26,880 pass); older failed numerical checks are not recertified.

Posthoc `package_source_reading.py`, `describe_first_source.py` (saved-array first
block analysis, zero model calls) and `describe_source_reading.py` preserve evidence,
figure provenance and the distinction from the frozen primary. They ran for this
release. `verify_source_reading.py` checks source/evidence integrity and independently
recomputes every published CSV rule using only the standard library. This does
not constitute a model replication. Private article text and all NPZ/weights stay
excluded; public inventories bind regeneration outputs to the original run.

## Source residual reanchoring (274)

[Protocol](../docs/experiments/274-source-residual-reanchoring.md) ·
[Result](../docs/analysis/274-source-residual-reanchoring.md) ·
[All32 old cases](../results/source-residual-reanchoring-274/README.md)

```sh
python -S research/boolean_coupling/verify_reanchoring.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research python -m boolean_coupling.run274 --reference-273 /path/to/regenerated/native-value-interchange-273 --out runs/my-source-reanchoring
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research python -m boolean_coupling.audit274 --reference-273 /path/to/regenerated/native-value-interchange-273 --run runs/my-source-reanchoring
```

The two scientific commands actually ran, after freezing all sources. They
require regenerated273 arrays from the next section, including its old False
and generated source trajectories; no private activation archive is downloaded.
Use the pinned MPS environment, cached checkpoint and a new output directory.
All32 cases are old. Donor/reference steps are calibration, and U/D/J additionally
use own-True information as declared. Native source keys remain finite in the
continuations. This is not a cheap complete False-only forecast or fresh test.

384 full forwards,8,960 extra PV calls and10,752 source RMS/V calls each are
recorded. The32 NPZ files total497,897,030bytes; public hashes bind the local
raw arrays. Reserve at least1.2GiB initially,768MiB per case, plus model/process
headroom. No old evidence is overwritten. The original1e-4 raw checks remain
unchanged; all21,504 pass here, without recertifying older failed derivatives.
Posthoc `package_reanchoring.py` / `describe_reanchoring.py` produce the public
records and figure; `verify_reanchoring.py` independently recomputes CSV rules
without model dependencies. Private writing and arrays are excluded.

## Native source-value interchange 273

This old-context native primitive study rejects the portable-signal primary 5/32.
Direction and magnitude controls remain separate from the strict native target.
[Read the result and budget](../docs/analysis/273-native-source-value-portability.md).

```sh
python -S research/boolean_coupling/verify_native_values.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research python -m boolean_coupling.run273 --out runs/my-native-value-interchange
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research python -m boolean_coupling.audit273 --run runs/my-native-value-interchange
```

Use the pinned environment described below and the locally cached checkpoint.
The runner requires only public 269 inputs and model weights. It does not need
an old activation archive. The recorded original run additionally supplied
`--reference-272 runs/causal-prefix-budget-272` to check exact old F/T replay;
omitting it skips that archive-only check, without changing the intervention.
Choose a new output directory; original evidence is never overwritten.

The source generator executes 896 full-shaped blocks, carrying only the edited
source and resetting every receiver input to observed False. The receiving
continuations remain fully finite, with 288 full forwards and the separately
recorded diagnostic projections/PV calls. This is an expensive causal test,
not a speedup. Reserve at least 2 GiB free disk for up to 600MiB new raw arrays
and process overhead. One paired donor adds another False/source observation.
No native True receiving state is supplied, but these cases were already
observed historically: repeating them is not fresh confirmation. Publication
packaging/verifiers certify the recorded release, not a new independent run.

Earlier studies returned to the failed whole-response forecast. **235** provides a complete write-error ledger and finite input-state/local-approximation split. **236** recomputes MLP20/21 in the actual forecast and still passes **0/32** full-response cases. A native reset before MLP20 improves the result but still leaves **0/32**, isolating further failure in attention 21–24. The fully finite oracle control replays exactly. These are retrospective diagnostics; Question B remains open. Earlier 234 local attention-2 success remains valid within its restricted budget.

For task definitions, a causal diagram and the shortest evidence path, begin
with the [reader guide](../docs/READING-GUIDE.md).

<a id="signed-amplitudes-and-directions-268269"></a>

## Signed amplitudes and directions 268–269

269's amplitude-only rule passes 32/32 fresh literal pairs under its explicit
L-cache/finite-computation budget.268's stronger old interaction account
fails 2/4 and stays failed. [Report and every qualification](../docs/analysis/268-269-amplitudes-directions-and-prediction.md).

```sh
python -S research/boolean_coupling/verify_amplitude.py
python -S research/boolean_coupling/reproduce_amplitude.py --plan --out runs/my-amplitude-replay
python research/boolean_coupling/reproduce_amplitude.py --run --out runs/my-amplitude-replay
```

The replay needs the pinned MPS environment/cached checkpoint and the public
fixtures; it needs **no earlier activation archive** and never pushes Git.
It repeats published269 literals, with a local seal explicitly marked
numerical replication. It is not another fresh confirmation and does not
bypass or change the original fresh CLI's required public-receipt guard.
Choose a new destination and keep several GiB free. It executes288forecast
forwards, a primitive preaudit,32 native measurements and a separate final audit.
All original scientific stages ran directly; this newly supplied wrapper has
plan/import/syntax checks, without a second full batch replay.

For268, first regenerate258/263/267 with the earlier pipelines, then use new
destinations with the exact scientific CLIs:

```sh
export PYTHONPATH=research OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
QB_258=runs/my-258-reproduction/experiment258
QB_263=runs/my-normalization-reproduction/normalization-branches-energy-263
QB_267=runs/my-field-state-267
QB_268=runs/my-amplitude-direction-268
python -m boolean_coupling.amplitude268
python -m boolean_coupling.rollout268 --original "$QB_258" --reference "$QB_263" --previous "$QB_267" --out "$QB_268"
python -m boolean_coupling.audit268 --original "$QB_258" --reference "$QB_263" --previous "$QB_267" --run "$QB_268"
python -m boolean_coupling.geometry268 --run "$QB_268" --reference "$QB_263" --out runs/my-directional-geometry-268
```

The last command is explicitly posthoc, zero-forward input geometry, not a
causal plane intervention. The original 269 fresh study used frozen
`forecast269 forecast`, `audit269 pre`, `forecast269 receipt`, a public
commit/push/proof, then `forecast269 native` and `audit269 final`. Its receipt
is retained unchanged. The local replay above serves numerical reproduction;
fresh research requires new fixtures and a newly frozen/sealed protocol.
Original raw CPU64 failures stay failed; all numerical scope limitations apply.
CI verifies record/source integrity, not independent model replication.

## Finite correction-field factorial 267

267 separates changing normalization correction vectors from their changing
local MLP operating point, with finite SiLU everywhere. The generator-only
crossing preserves the final interaction 4/4; the opposite crossing, both
frozen and matched random controls each pass 0/4. These are **four old contexts**,
not a new held-out forecast. [Report](../docs/analysis/267-correction-field-feedback.md).

```sh
python -S research/boolean_coupling/verify_fields.py
```

For actual numerical reproduction, first regenerate258 and263 with the
published pipelines below. Use the pinned MPS environment and a new destination:

```sh
export PYTHONPATH=research OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
QB_ORIGINAL=runs/my-258-reproduction/experiment258
QB_REF=runs/my-normalization-reproduction/normalization-branches-energy-263
QB_FIELDS=runs/my-field-state-267
python -m boolean_coupling.fields267
python -m boolean_coupling.rollout267 --original "$QB_ORIGINAL" --reference "$QB_REF" --out "$QB_FIELDS"
python -m boolean_coupling.audit267 --original "$QB_ORIGINAL" --reference "$QB_REF" --run "$QB_FIELDS"
```

The synthetic preflight reads no study inputs. Producer80 forwards include an
extra L forecast and 4,816 full-shape public MLP evaluations; the audit adds
primitive evaluations and smooth RMS derivative checks, not more full forwards.
All individual CLIs ran directly. There is no compute-speedup or circuit claim.
Keep several GiB free for model/swap headroom as well as the 260 MiB raw cap and
512 MiB reserve. The original storage-aborted20-forward attempt remains in the
release; its unchanged-source retry is the retained 80-forward experiment.
Do not delete evidence or repeat a failed destination. Every original raw
CPU64 failure stays failed; matched public Float32 checks qualify the stated
component scope without certifying whole-block derivatives. CI verifies the
published record, not model replication. Regenerate the posthoc figure with
`python research/boolean_coupling/describe_fields.py`.

## Local product mixing and finite activation control

265 performs a same-input MLP factorial, then removes its pointwise interaction
during complete continuation.266 checks the sparse local contrast against
finite SiLU. All examples are old; primary sufficiency/fidelity hypotheses fail.
[Results and numerical qualifications](../docs/analysis/265-266-local-mixing-and-state-feedback.md).

The standard-library release verifier adds no model calls:

```sh
python -S research/boolean_coupling/verify_transmission.py
```

For actual tensor/model reproduction, use the pinned MPS environment below and
first regenerate the263 arrays with the normalization pipeline. Then choose
new output directories; preserve every input and prior failure:

```sh
export PYTHONPATH=research OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
QB_REF=runs/my-normalization-reproduction/normalization-branches-energy-263
QB_TX=runs/my-transmission-reproduction
python -m boolean_coupling.local265 --reference "$QB_REF" --out "$QB_TX/local-mixing-265"
python -m boolean_coupling.rollout265 --reference "$QB_REF" --local "$QB_TX/local-mixing-265" --out "$QB_TX/local-mixing-continuation-265"
python -m boolean_coupling.audit265 --reference "$QB_REF" --local "$QB_TX/local-mixing-265" --run "$QB_TX/local-mixing-continuation-265"
python -m boolean_coupling.finite266 --reference "$QB_REF" --local "$QB_TX/local-mixing-265" --out "$QB_TX/finite-silu-interaction-266"
```

The first stage reconstructs224 saved input points with the pinned cached
weights. The second performs28 complete forwards with extra corner tensor work;
the others add primitive evaluations/audits only. Preserve all raw CPU64 failures;
exact matched-shape Float32 checks qualify the declared component scope without
certifying older whole-block derivatives. All four CLI stages ran directly;
CI verifies evidence integrity and does not independently replicate the model.
Regenerate the posthoc figure with
`python research/boolean_coupling/describe_transmission.py`.

## MLP normalization origin and branch interventions

261–264 test whether the MLP gate/up displacement needs finite input
normalization, then directly intervene on its branches and two remainder terms.
These are four **old** contexts. The energy-only primary fails2/4 joint cases;
it is not a new successful fresh predictor. See the
[report and numerical boundary](../docs/analysis/261-264-mlp-normalization-origin.md).

```sh
python -S research/boolean_coupling/reproduce_normalization.py --verify-only
python research/boolean_coupling/reproduce_normalization.py --plan \
  --inputs runs/my-258-reproduction/experiment258 \
  --out runs/my-normalization-reproduction
```

The verifier needs only the standard library and published metadata. To perform
the model/tensor experiment, first regenerate258's qualified final arrays using
the pipeline below and install the pinned `requirements-recorded.txt` environment
on an MPS machine. Then change `--plan` to `--execute`, using a **new, disjoint
output parent**. The wrapper rejects existing output paths, never executes the
preserved invalid original261 producer, and stops on unexpected failures.

It executes corrected261b, the two strict numerical audits (both expected to
fail on their recorded comparisons), full-shape261c replay, bounded261d,
local262 factors, the44-forward263 intervention and corrected264b analysis.
The first model stage adds28 forwards:72 total, plus audit tensor work. Several
GB of free disk are needed for raw arrays. Logs and all failures remain in the
new output tree. The strict failures are retained as failures; qualification
does not change their tolerances or certify older whole-block derivatives.

All original scientific CLI stages ran locally. This new wrapper's plan,
argument safety and syntax are checked; the entire wrapper has **not** been
rerun as a second model replication. Figures are posthoc and can be regenerated
from the complete public CSVs with
`python research/boolean_coupling/normalization_figures.py`.

## Local energy diagnostic and reader-figure provenance

260 uses the existing258 forecast witnesses to separate individual omitted-write
concentration from cross-neuron cancellation. It adds no model forwards or
native measurements. [Result](../docs/analysis/260-silu-energy-concentration-result.md).

```sh
PYTHONPATH=research python -m boolean_coupling.energy260 \
  --run runs/my-258-reproduction/experiment258 --out runs/my-260-diagnostic
python -S research/boolean_coupling/verify_reader_guide.py
```

The first command needs pinned cached CPU weights and locally regenerated258
arrays. The output must be new. The second verifies the complete public260
records, CSV summaries, identities and reader-figure provenance with only the
standard library. It does not redo omitted tensor products or model experiments.
The existing full-history verifier below remains unchanged.

## Current:258/258b receiver-MLP forecasting and259 ranking diagnostic

258's fixed1% finite-SiLU rule, combined with cubic receiver queries, passes32/32
fresh full-response cases. Quadratic/random controls pass0/32.259 is a
retrospective comparison of all saved C/S MLP witness inputs; it is not a fresh
forecast. [258 report](../docs/analysis/258-receiver-mlp-result.md) ·
[259 report](../docs/analysis/259-silu-score-factors-result.md).

```sh
python -S research/boolean_coupling/reproduce_receiver_mlp.py --verify-only --history --plan
python research/boolean_coupling/reproduce_receiver_mlp.py --run --offline --out runs/my-258-reproduction
```

The first command uses only the standard library and checks the entire published
history plus new258/259 receipts and summaries. Integrity is not numerical model
replication. The second requires the pinned Apple Silicon environment below.
It runs forecast→258b qualification→pre-audit→local seal→local proof→native/control
measurement→final audit, using a new directory. It never publishes or pushes.
Source code and all individual scientific stages executed in the original study;
we have not run a separate end-to-end replay of this convenience wrapper.

258b is an explicit changed whole-primal qualification: full-sequence-shape
Float32 down projections must replay exactly; original CPU64 absolute errors
remain in separate failed-check CSVs. The published original pre-audit failure
is preserved. A new replication directly executes258b; it does not fabricate
another copy of the historical original failure. Every other component and
behavioral criterion remains fixed. Raw QK and older whole-block derivative
failures remain unresolved. No full independent real-arithmetic model audit.

The selected fraction is .999504% across all receiving coordinates; the rest
use quadratic SiLU. Full public SiLU still executes before hook replacement.
All finite source/K/V/RMS/projection/product/softmax work and extra audit
projections are declared in the report. This is not a speedup experiment.

After generating258 arrays, reproduce the fixed-input259 diagnostic separately:

```sh
PYTHONPATH=research python -m boolean_coupling.score259 \
  --run runs/my-258-reproduction/experiment258 --out runs/my-259-diagnostic
```

This loads pinned down-projection weights on CPU and the saved forecast witnesses,
without running a model forward. It checks original mask/component bridges and
reports all nine variants, both whole and verdict scopes. Inputs are published
literals and existing forecast states, so rerunning does not make it fresh
confirmation. Raw arrays are generated locally, never downloaded from a private
research archive.

## Earlier results and wrappers

## Earlier: source versus receiver SiLU

243 tests a frozen regional factorial on the existing 242b contexts. Source-only and receiver-only each fail the full contrast-recovery rule. Their interaction is substantial; native residual errors improve but every arm remains at **0/32** joint accuracy passes. [Report](../docs/analysis/243-source-receiver-silu-result.md).

```sh
python -S research/boolean_coupling/reproduce_regional.py --verify-only --history --plan
python research/boolean_coupling/reproduce_regional.py --run --offline \
  --input-242 runs/existing-242b --out runs/my-regional-test
```

Omit the input option to regenerate 242b first. The regional test adds 32 zero-edit controls and 128 modified forwards, without new native True measurements. QQ/FF replay 242b100/101 exactly; receiver edits leave source states unchanged. The corrected 243b scorer preserves the original metadata-key failure and all numerical rules. Raw arrays are inventoried locally, and exact same-environment endpoint replay is required. These are retrospective diagnostics, not new predictive confirmation.

## Earlier: fresh uniform primitive test

242b applies one fixed primitive-family policy across all blocks, generating its own source and receiver states. All forecasts were pushed before native measurements. The primary passes **0/32**, at **50.3%** whole-displacement and **1.44 nats** margin error; finite SiLU reduces the former to **17.4%**, still **0/32**. [Report and numerical scope](../docs/analysis/242b-uniform-primitive-families-result.md).

```sh
python -S research/boolean_coupling/reproduce_uniform.py --verify-only --history --plan
python research/boolean_coupling/reproduce_uniform.py --run --offline --out runs/my-uniform-test
```

This test needs no earlier activation archive or prerequisite experiment. It needs the pinned MPS environment below. The wrapper records replication mode before forecasts and creates a local seal; it does not push or alter Git. Its already observed literals are not fresh confirmation. Original 242's implementation failure is preserved separately; 242b changes the unsupported nested SiLU AD expression, without changing tolerances or hypotheses.

## Earlier: complete upstream vectors and transport geometry

241 traces all 29 native-input remainder vectors to the missing input before MLP16. Four greedy sites pass 0/32; the first sufficient prefix uses 16 sites. Explicitly posthoc metric checks distinguish reduced individual energy from weakened cancellation. [Report](../docs/analysis/241-before-mlp16-vectors-result.md).

```sh
python -S research/boolean_coupling/reproduce_early.py --verify-only --history --plan
python research/boolean_coupling/reproduce_early.py --run --offline \
  --input-238 runs/existing-238 --input-240 runs/existing-240 --out runs/my-early-test
```

Omit both input options to regenerate the fixed chain. 241 itself uses CPU64 checkpoint/component arithmetic, not new ordinary model forwards; rebuilding its prerequisites needs MPS. The 241 protocol requires the original input-array hashes. Replication in a different numerical environment may stop on those hashes; this release does not silently substitute different regenerated inputs. All used tangents, finite-difference endpoint witnesses, terminal vectors and original numerical bounds are retained. The three descriptive follow-ups are explicitly posthoc and add no causal interventions.

## Earlier: target-blind SiLU selection and finite primitive intervention

240's forecast-derived 1% activation mask recovers the whole missing MLP16 contrast in32/32 old cases. Conditional final gaps shrink, but the materiality guard admits only 3/32 native-input and 0/32 forecast-input cases. Full finite continuation from the forecast input still misses the native response by25.2% and 1.09 nats. [Report](../docs/analysis/240-target-blind-silu-result.md).

```sh
python -S research/boolean_coupling/reproduce_silu.py --verify-only --history --plan
python research/boolean_coupling/reproduce_silu.py --run --offline \
  --input-239 runs/existing-239 --out runs/my-silu-test
```

Omit `--input-239` to regenerate the fixed prerequisite chain. 240 needs the MPS environment below. It seals all masks before intervention outcomes, runs384 direct activation edits with finite continuation, and independently audits all cases. Recorded work adds32 False forwards,32 source-only blocks,32 mask MLPs,384 intervention MLPs,4,608 suffix blocks and 384 heads; no new ordinary True forward. The post-run prerequisite-hash recheck binds the archived original inputs; regenerated arrays are evaluated numerically rather than required to match archived compressed-file bytes.

## Earlier: upstream generators and their input-dependent curvature

238 traces every local remainder through the original affine prefix. The selected MLP16–19 repair improves margin5.15→1.34 nats and whole error 72.3%→31.6%, but has 0/32 full-response passes.239's finite-normalization/quadratic-SiLU formula explains the local generator contrast at MLP17/18 and fails the complete four-site account. [238 report](../docs/analysis/238-upstream-generators-result.md) · [239 report](../docs/analysis/239-generator-curvature-result.md).

```sh
python research/boolean_coupling/reproduce_curvature.py --run --out runs/my-curvature-test
```

Reuse a local238 prerequisite with `--offline --input-238 runs/existing-238`. To run238 using an existing237b prerequisite:

```sh
python research/boolean_coupling/reproduce_upstream.py --run --offline \
  --input-237 runs/existing-237b --out runs/my-upstream-test
```

238 seals its audited selection before intervention outcomes and separately reconstructs all finite-difference endpoints.239 alone needs CPU calculations and the cached checkpoint; rebuilding earlier studies requires the MPS setup below. All outputs must be new. Old native input directions are supplied only as retrospective diagnostics. Source-only states have pre-observation computational provenance, clarified [here](../docs/analysis/238-information-budget.md).

## Earlier: late-attention omissions and numerical diagnosis

237b crosses QK and softmax while keeping normalization/value products and remaining MLPs finite. Neither selective policy passes the joint final-response bound; QK restoration leaves10.8% median whole error under a native reset. Even complete late computation from the ordinary wrong incoming state leaves72.3%. [Full report](../docs/analysis/237-late-attention-result.md).

Qualification is partial. Original237 fails its final bridge; original237b fails one Float64 MLP bridge. The 237c numerical diagnostic preserves every bound/failure and finishes25,120 checks,25,119passing. Independent Float32 MLPs and same-backend public replays pass. These are old-context finite-computation diagnostics, not a new False-only predictor.

```sh
python research/boolean_coupling/reproduce_late.py --run --out runs/my-late-diagnostic
```

Add`--offline --input-236 runs/existing-236` to reuse a qualified cached prerequisite. Add`--include-failed-237` to preserve a rerun of the original failed attempt separately. Original237b audit failure is retained before the complete diagnostic runs. Unexpected failures stop for review. [All cases, logs, exact commands and computation budget](../results/late-attention-237b/README.md).

## Inspect without a model

From the repository root, with Python 3.12:

```sh
python research/boolean_coupling/reproduce_curvature.py --verify-only --history --plan
```

This uses only the standard library. It checks all active releases, their scientific source seals, and the publication-filtered historical archive. It also prints the actual reproduction commands. It does not load weights, rerun numerical audits against omitted activations, or constitute model replication.

The [experiment index](../docs/EXPERIMENTS.md) links every active protocol, report, audit, failed attempt and data summary. The [research notes](../docs/RESEARCH-NOTES.md) explain which hypotheses changed and why.

## Run the global failure diagnostic

After the environment setup below:

```sh
python research/boolean_coupling/reproduce_failure.py --run --out runs/my-global-diagnostic
```

This regenerates fixed 228 inputs, executes 235's ledger/finite interventions and audit, then 236's actual-continuation factorial and audit. Add `--offline` for a cached checkpoint. Reuse an existing 228 run with `--input-228 runs/existing-228`, or a qualified 235 prerequisite with `--study 236 --input-235 runs/existing-235`. Use `--study 235` for the ledger study alone. Every output must be new; failures stop and preserve logs. These old-context diagnostics are not new prospective confirmations.

[235's guide](../results/global-error-budget-235/README.md) explains the oracle terminal vectors and conditional split. [236's guide](../results/upstream-factorial-236/README.md) distinguishes normal and native-reset input regimes, all branches, actual FD directions and finite final readout. The prerequisite's ordinary-model work is separate from each diagnostic's recorded component budget. No private activation archive is required.

## Earlier prospective local forecasts

The unchanged [transport wrapper](boolean_coupling/reproduce_transport.py) reproduces 233/234; [source-context wrapper](boolean_coupling/reproduce_source_context.py) reproduces 232; [local wrapper](boolean_coupling/reproduce_local.py) reproduces 231 and verifies 230. Each retains its original inputs, seals, criteria and scope. See the [experiment index](../docs/EXPERIMENTS.md). These local successes did not predict the final native response.

## Run the 229 diagnostic

After the environment setup below:

```sh
python research/boolean_coupling/reproduce_generator.py --run --out runs/my-generator-reproduction
```

The wrapper regenerates the preceding 228 run from its public fixed inputs, then executes 229 and its separate NumPy audit. Add `--offline` for a cached checkpoint. To reuse existing local 228 arrays:

```sh
python research/boolean_coupling/reproduce_generator.py --run --offline \
  --input-228 runs/my-existing-228 --out runs/my-new-generator
```

Every output directory must be new. This is a retrospective diagnosis with observed native parents; replication does not become a fresh forecast test. [229's release guide](../results/first-attention-generator-229/README.md) explains the signed vectors, direct primitive interventions, all-ineligible downstream rule, bootstrap corrections and budget. The model run makes 32 ordinary False calls and uses saved True states; zero *new* ordinary True calls does not make it False-only.

## Run the earlier whole-response test (228)

Follow the environment setup below, then run:

```sh
python research/boolean_coupling/reproduce_alignment.py --run --out runs/my-alignment-reproduction
```

This runs the unchanged 228 preflight and forecast machinery, seals and audits all forecasts, and only then measures the native True targets. Final scoring uses the separately named 228b metadata correction. See the [228 guide](../results/false-only-alignment-228/README.md) for ordering, budget, failed criteria, original failure record and fixed inputs. Replication is not a second fresh-context demonstration. The wrapper can cache the public pinned checkpoint; add `--offline` when it is already available.

## Run the earlier causal sequence from public inputs

The validated environment is Python 3.12.14 on Apple Silicon/MPS, with Float32/eager execution. Runtime versions are pinned in [requirements-recorded.txt](boolean_coupling/requirements-recorded.txt). The original machine had 24 GB of RAM. Allow several GB for checkpoint weights and generated activations. Other backends and versions have not been validated.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r research/boolean_coupling/requirements-recorded.txt
python research/boolean_coupling/reproduce_tail.py --run --out runs/my-reproduction
```

This runs the unchanged 225b parent, its qualification and intervention audits, then the unchanged 226b and 227b model runners and their separate auditors. It finishes with the checkpoint-based finite norm/projection checks and the explicitly posthoc product-algebra check. The wrapper finds the pinned `model.safetensors` in the local Hugging Face cache after the parent run; `--checkpoint` can supply its snapshot path explicitly.

The first parent run may download the public checkpoint. Use `--offline` when it is already cached. These experiments require no API key and make no paid model API calls. They do execute local model computations.

Reuse a completed 225b output to avoid rerunning the parent:

```sh
python research/boolean_coupling/reproduce_tail.py --run --offline \
  --base runs/my-existing-225b --out runs/my-new-continuation
```

`--out` must be new. Each phase writes its log, and `pipeline.json` records completed steps. A failing phase stops the sequence and preserves its outputs. No later step silently substitutes a native state or widens a tolerance.

## What the computation includes

225b performs 64 ordinary model computations and 704 finite suffix continuations. It includes primitive derivative qualification at its declared local scope.

226b performs 640 reference-block replays, 160 additive-input blocks, 160 additional local MLP calls, 480 intervention MLP calls, 1,920 continuation blocks and 1,120 finite heads. 227b performs 128 reference blocks, 96 local MLP calls, 320 direct down projections, 1,280 continuation blocks and 320 finite heads. These are different compute units, not interchangeable full-model calls. The follow-ups use finite computations and do not recertify the disputed original whole-block derivatives.

The checkpoint-based validator adds 608 finite-forward comparisons, without a full model call. All 32 cases in each follow-up are previously observed and dependent. Numerical validation does not make them fresh examples.

## Code map

235 uses [ledger235.py](boolean_coupling/ledger235.py), [finite235.py](boolean_coupling/finite235.py) and separate [audit235.py](boolean_coupling/audit235.py). 236 uses [continue236.py](boolean_coupling/continue236.py) and [audit236.py](boolean_coupling/audit236.py). [reproduce_failure.py](boolean_coupling/reproduce_failure.py) orchestrates public-input regeneration; [plot_failure.py](boolean_coupling/plot_failure.py) renders saved CSVs without model calls. Earlier scientific sources are unchanged.

233 uses explicit NumPy [transport233.py](boolean_coupling/transport233.py) and separate Torch CPU64 [audit233.py](boolean_coupling/audit233.py) with every intervened derivative qualified by AD/FD. [describe233.py](boolean_coupling/describe233.py) is explicitly post-observation. 234 uses [local234.py](boolean_coupling/local234.py), separate [audit234.py](boolean_coupling/audit234.py), fixed [cases234.json](boolean_coupling/cases234.json), [reproduce_transport.py](boolean_coupling/reproduce_transport.py), and model-free [plot_transport.py](boolean_coupling/plot_transport.py). All earlier source files and seals remain unchanged.

232 uses [local232.py](boolean_coupling/local232.py), separate [audit232.py](boolean_coupling/audit232.py), fixed [cases232.json](boolean_coupling/cases232.json), two-anchor derivative qualification, and [reproduce_source_context.py](boolean_coupling/reproduce_source_context.py). [describe232.py](boolean_coupling/describe232.py) explicitly recombines sealed components after observation; [plot232.py](boolean_coupling/plot232.py) renders the recorded results. These reporting steps make no model calls.

230 uses [run230.py](boolean_coupling/run230.py), [curvature230.py](boolean_coupling/curvature230.py) and separate [audit230.py](boolean_coupling/audit230.py). [describe230.py](boolean_coupling/describe230.py) is explicitly post-observation. 231 uses [local231.py](boolean_coupling/local231.py), [audit231.py](boolean_coupling/audit231.py), fixed [cases231.json](boolean_coupling/cases231.json) and [reproduce_local.py](boolean_coupling/reproduce_local.py). [plot_local.py](boolean_coupling/plot_local.py) renders both studies from saved records without model calls.

229 uses [generator229b.py](boolean_coupling/generator229b.py), [attention229.py](boolean_coupling/attention229.py), separate [audit229.py](boolean_coupling/audit229.py), position-resolved [describe229.py](boolean_coupling/describe229.py), [plot229.py](boolean_coupling/plot229.py) and [reproduce_generator.py](boolean_coupling/reproduce_generator.py). Original generator229.py and the intermediate bootstrap source remain as failure evidence; use generator229b.py.

228 uses [alignment228.py](boolean_coupling/alignment228.py), [smooth228.py](boolean_coupling/smooth228.py), original [audit228.py](boolean_coupling/audit228.py), corrected final [audit228b.py](boolean_coupling/audit228b.py), and [reproduce_alignment.py](boolean_coupling/reproduce_alignment.py). Its [diagnostic](boolean_coupling/diagnose228.py) and [plotter](boolean_coupling/plot228.py) use saved arrays/CSVs and make no model calls.

- [core225b.py](boolean_coupling/core225b.py), [run225b.py](boolean_coupling/run225b.py), [audit225b.py](boolean_coupling/audit225b.py): unchanged parent computation and audit.
- [tail226b.py](boolean_coupling/tail226b.py), [audit226b.py](boolean_coupling/audit226b.py): finite sublayer mixing, transport and causal removals.
- [tail227b.py](boolean_coupling/tail227b.py), [audit227b.py](boolean_coupling/audit227b.py): ordered primitive contributions and direct product edits.
- [validate227b.py](boolean_coupling/validate227b.py): independent finite RMSNorm/gate/up reconstruction from checkpoint tensors.
- [describe227b.py](boolean_coupling/describe227b.py): explicitly posthoc bilinear identity and arithmetic ledger.
- [reproduce_tail.py](boolean_coupling/reproduce_tail.py): dependency-free verification and pipeline orchestration; it does not alter the sealed runners.
- [plot226_227.py](boolean_coupling/plot226_227.py): redraws the published figure from the recorded CSVs.

The unsuffixed 225, 226 and 227 entry points preserve failed attempts. Use the `b` versions above for reproduction. The [225b guide](boolean_coupling/README.md) and both follow-up result guides retain individual phase commands and qualification details.

## Native attention and first-mixing diagnostics (244–246b)

```sh
python -S research/boolean_coupling/reproduce_mixing.py --verify-only --history --plan
python research/boolean_coupling/reproduce_mixing.py --run --offline --out runs/my-mixing-study
```

The wrapper regenerates public242b/243 prerequisites, native-attention244,
first attention mixing245, and the selected-block origin test246b, with separate
auditors. Add --input-242, --input-243, or --input-244 to reuse appropriate
qualified local outputs; --input-244 requires its --input-242. Output must be
new. All are known-context retrospective diagnostics. FF uses an ordinary
native True replay and is never counted as a successful forecast. Primitive
removals use observed factorial components. Fixed numerical bounds and earlier
failures remain. Article drafts are not part of any release.

## Native context, carry and source tests (247–249)

```sh
python -S research/boolean_coupling/reproduce_native.py --verify-only --history --plan
python research/boolean_coupling/reproduce_native.py --run --offline --out runs/my-native-study
```

This regenerates242b prerequisites, native attention247, finite-suffix248, and
source249 forecasts/controls/audit. Reuse --input-242; matching --input-247 and
--input-248 can also be supplied (later inputs require their matching earlier
prerequisites). Every output directory must be new. Unexpected failures stop.

247/248 use observed native incoming states. 249's source-only algorithm computes
its source from False plus the known edit, and seals every forecast before
oracle-control reads. These are known-context tests, not fresh confirmation.
Dense source and suffix computation is substantial; no speedup or learned
circuit is claimed. Full native accuracy still fails. Code/source seals and all
controls are retained; omitted arrays are regenerated from public pinned inputs.

## Native communication250b

Verify complete evidence and history without model imports:

```sh
python -S research/boolean_coupling/reproduce_communication.py --verify-only --history --plan
```

Run the finite factorial and separate audit in the pinned environment:

```sh
python research/boolean_coupling/reproduce_communication.py --run --offline --input-242 runs/existing-242b --out runs/new-communication-reproduction
```

Omit `--input-242` to regenerate the prerequisite via the existing242b pipeline. Each output must be new. The 250b producer wrapper explicitly retains the original250 complete-data witness-count failure, checks its narrow preconditions, then seals corrected accounting; any earlier/different failure stops. All 32 witnesses at blocks 10/19/28 are independently audited. This requires352 full forwards,608 additional suffix blocks,32 additional full heads and the documented primitive recomputations; no speedup. [Frozen correction](../docs/experiments/250b-witness-completion-correction.md) · [Result](../docs/analysis/250b-native-source-communication-result.md).

## Latest False-only forecasts251b–252

```sh
python -S research/boolean_coupling/reproduce_attention.py --verify-only --history --plan
python research/boolean_coupling/reproduce_attention.py --run --offline --out runs/new-attention-reproduction
```

Choose `--experiment 251` or `--experiment 252` to run one. No prior activation archive or242b model regeneration is required. The wrapper uses the pinned cached checkpoint, executes forecasts, pre-audit, local seal proof, native controls and final audit; published-literal repetition is not fresh confirmation. For251 it explicitly retains the expected original raw-QK audit failure before 251b's composed-output qualification. Unexpected failures stop. The raw absolute checks remain failed and are never called numerical passes.

Actual original studies used public remote forecast receipts before their native measurements. Each study's completed run uses 224 full forwards plus the documented extra primitive products and separate audits. The initial251 disk-guard attempt adds five aborted-case forwards; all evidence is retained. There is no speedup claim. [251 report/boundary](../docs/analysis/251-fixed-receiver-attention-result.md) · [252 report](../docs/analysis/252-source-driven-attention-result.md).

## Query generation 253 and predicted curvature 254

```sh
python -S research/boolean_coupling/reproduce_query.py --verify-only --history --plan
python research/boolean_coupling/reproduce_query.py --run --offline --experiment 254 --out runs/new-query-reproduction
```

Choose 253 or both to include the tangent/key factorial. Each run is direct from the pinned checkpoint/fixtures, with no earlier activation archive. The wrapper uses an explicit local proof between the global forecast seal and native measurement; repeating published literals is replication, not fresh confirmation. The original scientific studies used public remote receipts. Individual frozen CLI stages were executed and audited; this batch's new wrapper has integrity/plan/syntax checks, without a separate second full model replay. The earlier 251b/252 full local pipeline replay remains separately documented.

254's primary uses one False run, a tangent seed and two corrected passes; zero/random/native replay arms are additional controls. It costs more than ordinary edited inference. Exact budgets and the retained raw-coordinate/composed-output numerical boundary are in the [protocol](../docs/experiments/254-predicted-query-curvature.md) and [report](../docs/analysis/254-predicted-query-curvature-result.md).

## Single-pass polynomial query forecast 256

```sh
python -S research/boolean_coupling/reproduce_local_query.py --verify-only --history --plan
python research/boolean_coupling/reproduce_local_query.py --run --offline --out runs/new-local-query-reproduction
```

This runs 256 directly from the pinned checkpoint and published fixtures, with
an explicit local proof between the global forecast seal and native measurement.
It needs no 254 activation archive. Original individual CLI stages and separate
pre/final numerical audits were executed; the new wrapper has integrity/plan/
syntax checks, without a separate second full model replay. Repeating published
literals is replication, not fresh confirmation.

The primary algorithm uses one False pass plus one forecast pass. K/V, MLPs,
source computation and readout remain finite. No wall-clock speedup or compact
learned generator is established. Complete budgets, original failed coordinate
checks and passing composed-output scope are in the [report](../docs/analysis/256-local-query-curvature-result.md).

255 is model-free retrospective algebra and requires retained 254 arrays:

```sh
PYTHONPATH=research python -m boolean_coupling.curvature255 --run runs/YOUR-254 --out runs/NEW-255
```

Its source verifies the 254 input-array hashes; every destination must be new.
The optional posthoc command diagnoses local smooth Taylor radii without new
model calls. It does not prove whole-network stability:

```sh
PYTHONPATH=research python -m boolean_coupling.radius256 --run runs/YOUR-256
```

## Source/receiver radius diagnostic 257

```sh
python -S research/boolean_coupling/reproduce_source_radius.py --verify-only --history --plan
PYTHONPATH=research python -m boolean_coupling.source_radius257 --run runs/YOUR-256 --out runs/NEW-257
```

The diagnostic reads already measured native 256 arrays and two public embedding
rows; no model forward occurs. The outside-radius hypothesis is rejected.
The initial source embedding comparison is unique, not 32 independent facts.
A combined published-literal reproduction is available with
`reproduce_source_radius.py --run --offline --out runs/NEW`, which first invokes
256's locally sealed pipeline, then 257. Individual original stages ran; this
new composition has plan/integrity/syntax checks, without another full model
replay. It preserves failures and requires a new output directory.

## Backbone curvature and feedback (270/271)

These are retrospective diagnostics on four OLD263 contexts. Both primary
hypotheses fail; all76 forwards finish.270 tests the common backbone rather
than interaction between normalization corrections.271 uses an extra successful
B forecast as a controller to test amplitude/query feedback, not as a free
prediction input. Read the [result and numerical scope](../docs/analysis/270-271-backbone-curvature-and-feedback.md)
and protocols [270](../docs/experiments/270-backbone-curvature-feedback.md),
[271](../docs/experiments/271-amplitude-query-mediation.md) before interpreting them.

The code requires regenerated258,263 and268 arrays from the earlier sections,
the pinned Qwen checkpoint, the documented Python/Torch/Transformers/MPS
environment, and storage headroom. It never downloads a private activation
archive or deletes old evidence. Choose a NEW output parent and ensure it
exists. Each scientific writer refuses an existing destination. From repo root:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
INPUTS=/path/to/regenerated/runs
OUTBASE=/path/to/new/outputs
mkdir -p "$OUTBASE"
python -m boolean_coupling.rollout270 --original "$INPUTS/receiver-mlp-curvature-258" --reference "$INPUTS/normalization-branches-energy-263" --previous "$INPUTS/amplitude-direction-268" --out "$OUTBASE/backbone-curvature-feedback-270"
python -m boolean_coupling.audit270 --original "$INPUTS/receiver-mlp-curvature-258" --reference "$INPUTS/normalization-branches-energy-263" --previous "$INPUTS/amplitude-direction-268" --run "$OUTBASE/backbone-curvature-feedback-270"
python -m boolean_coupling.rollout271 --original "$INPUTS/receiver-mlp-curvature-258" --reference "$INPUTS/normalization-branches-energy-263" --previous "$INPUTS/amplitude-direction-268" --controller "$OUTBASE/backbone-curvature-feedback-270" --out "$OUTBASE/amplitude-query-mediation-271"
python -m boolean_coupling.audit271 --original "$INPUTS/receiver-mlp-curvature-258" --reference "$INPUTS/normalization-branches-energy-263" --previous "$INPUTS/amplitude-direction-268" --controller "$OUTBASE/backbone-curvature-feedback-270" --run "$OUTBASE/amplitude-query-mediation-271"
python -m boolean_coupling.clamps271 --run "$OUTBASE/amplitude-query-mediation-271" --out "$OUTBASE/controller-clamp-checks-271"
```

These five scientific/posthoc CLIs actually ran. The published cases use B/J
exact-replay controls; rerunning published literals is a numerical repetition,
not fresh confirmation. The measured arithmetic remainder remains in every
backbone-removal arm. Actual finite SiLU witnesses execute, and271 additionally
supplies B-derived controllers. No speedup is demonstrated.

Raw case arrays total64,187,261 bytes for270 and58,344,823 for271, plus signed
vector arrays inventoried publicly. Preserve original raw CPU64 projection
failures169/6720 and161/6384 at1e-4. Exact public Float32 replays and independent
smooth primitive derivative checks qualify their declared scope, not every
whole-model derivative or a learned circuit.

For repository-only verification, requiring neither model packages nor arrays:

```sh
python -S research/boolean_coupling/verify_backbone.py
```

The posthoc `package_backbone.py` and `describe_backbone.py` preserve publication
and plot provenance. Public logs select scientific progress/result/timing lines;
`log-provenance.json` records original hashes and that representation change.
Private articles, weights and activation arrays remain excluded.

<a id="receiving-coefficients-291"></a>
## Receiving MLP response coefficients (291)

This retrospective experiment separates the gate response, the up multiplier
of that response and the gate-activation coefficient on the up path. It holds
the edited source/attention/head computation supplied and receiving norm
responses at False. Eight finite corners and three shuffled coefficient
controls share the same zero-offset edited trajectory. Full 111 is calibration;
the six proper nonzero corners are the sufficient-account hypotheses.

Read the [frozen protocol](../docs/experiments/291-receiving-coefficients.md)
and [result and numerical scope](../docs/analysis/291-receiving-coefficients.md).
This tests a small common correction around an edited forecast, not the full
native Boolean displacement or a compact counterfactual predictor. It uses all
32 previously observed 285 contexts. No ordinary True forward or True gradient
is added. Every source/attention/MLP replay and derivative calculation counts.

Regenerate the 285 native reference, 287 head and 289/290 saved arrays using the
preceding sections. With the pinned MPS environment, an existing new output
parent and the frozen sources, execute from the repository root:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
python -m boolean_coupling.run291 --reference-285 /path/to/native-structure-285 --previous-290 /path/to/receiving-nonlinearity-290 --head-287 /path/to/source-receiver-readout-287 --out /path/to/NEW-receiving-coefficients-291
python -m boolean_coupling.audit291 --reference-285 /path/to/native-structure-285 --previous-290 /path/to/receiving-nonlinearity-290 --head-287 /path/to/source-receiver-readout-287 --run /path/to/NEW-receiving-coefficients-291
```

The destination must not exist. New guards require 512 MiB initial free space,
384 MiB before each case and at most 192 MiB of case archives. They do not alter
any historical guard. Preserve and report a stop; do not delete old evidence.
Checkpoint repetitions are numerical reproductions, not fresh confirmation.

For a model-package-free repository check:

```sh
python -S research/boolean_coupling/verify291.py
```

For tiny CPU qualification, first regenerate the earlier CPU fixtures. Run
`preflight291` with `--out` a fresh directory and
`--reference-285-fixture`, `--previous-290-fixture`, `--head-287-fixture` pointing
to their completed fixture directories. It checks four unequal-length cases,
all source coordinates, selected-layer primitive derivatives and coherent
bound/FD/primitive tamper rejection. It does not require random-model scientific
hypotheses to pass. Public verification checks source seals, records, scalar
finite differences and sufficient statistics; it cannot replicate omitted
vectors or the checkpoint independently. `publish291.py` and `describe291.py`
preserve the delivery and figure provenance separately from frozen science.

## Causal-prefix budget baseline (272)

[Protocol](../docs/experiments/272-causal-prefix-budget-baseline.md) ·
[Review/result](../docs/analysis/272-research-review-and-causal-baseline.md) ·
[All records](../results/causal-prefix-budget-272/README.md)

272 is a retrospective reference on all 32 OLD269 contexts. It retains a full
False KV observation, crops strictly before the edited token, supplies the known
edited suffix at global positions and evaluates all affected operations finitely.
It is not a Taylor approximation, native activation oracle or learned circuit.
The information budget is a per-layer prefix KV trace, not one final vector.

The scientific runner requires a completed269 reference directory to bind the
old F/T endpoints and input/runtime records exactly. Use the earlier269 pipeline
or documented local published-literal repetition to regenerate that reference;
activation archives stay out of Git. The cache algorithm itself (`cache272.py`)
requires only the pinned checkpoint, token IDs and one ordinary False KV run;
no old True activation archive supplies its forecast. The reference is used
solely for evaluation and replay binding.

Use the pinned local MPS environment and an existing new output parent, from
repo root. The two scientific CLIs below actually ran; destinations must not
exist. They preserve all prior arrays and retain a512MiB free reserve.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=research
python -m boolean_coupling.run272 --reference /path/to/signed-amplitude-forecast-269 --out /path/to/NEW-causal-prefix-budget-272
python -m boolean_coupling.audit272 --reference /path/to/signed-amplitude-forecast-269 --run /path/to/NEW-causal-prefix-budget-272
```

This executes64 profiled full and96 profiled suffix calls, then96 unprofiled
full/suffix timing pairs:160 full plus192 suffix calls in total. Cache cloning
is inside the suffix timer; common False observation/model loading/tokenization
are excluded from both marginal timings. Paired order alternates deterministically.
The separate audit reconstructs measured matrix counts from pinned config and
checks finite public heads, prefix identities and every case decision.

Original1e-4 changed-shape coordinate diagnostics fail299/2688; these stay failed.
Every same-executed-shape finite head and every timed output replay exactly.
Source/Qwen/cache/mask/checkpoint/runtime seals and raw prefix hashes are saved.
Raw trajectory arrays total140,726,694 bytes, plus metadata; full KV tensors
are not persisted and regenerate from the False run. No derivative is used or
recertified. A local repetition is not another fresh confirmation.

For repository-only stdlib verification:

```sh
python -S research/boolean_coupling/verify_cache.py
```

Posthoc scripts `review272.py`, `package_cache.py` and `describe_cache.py` bind
the completion inventory, all public evidence and figures. The inventory covers
published artifacts, not a new line-by-line review of every legacy runner.
No private article or unpublished legacy history is included.
