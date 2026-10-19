# 235: does the local research address the failed end-to-end forecast?

Frozen before this diagnostic on 19 October 2026. All 32 native targets and
forecasts in 228 are already observed. Use its best finite-MLP arm F. This is
retrospective diagnosis, not another fresh prediction or correction sweep.

## Complete error budget before MLP24

The residual stream uses one coordinate basis across updates. For each of the
24 attention and first 23 MLP updates, calculate

```
w_j = (native_output - native_input) - (forecast_output - forecast_input).
```

These are errors in the generated writes, including effects of inaccurate
incoming states. They are not newly generated nonlinear remainders alone.
Their sum plus the final Float32 installation error must exactly equal the
native-minus-F parent error at the input to MLP24. Receiving input error before
block1 is zero. Check complete-vector closure at1e-9; retain all48 terms,
including the cast term. Report signed projection, norm, cancellations and
position dependence for every site and every case.

Rank the47 sublayers by median signed projection onto the whole-receiving
terminal parent error; break ties by chronological index. Select the first4,
before any finite diagnostic/outcome calls. No ranking by final margin or by
which intervention succeeds. Publish all ranks and schema/order results.

This ledger is telescoping accounting. It does not establish upstream causal
origin, transmission or a learned circuit. A large write error might reflect
the operation's approximation or an inherited input error.

## Finite conditional split at the four selected sites

Reconstruct native and forecast full inputs using one repeated ordinary False
trajectory and the saved, verified native source row. Recompute each selected
operation finitely at both its native input and its forecast input. Separate:

```
generation = finite_write(forecast_input) - original_forecast_write
state     = finite_write(native_input) - finite_write(forecast_input)
replay    = saved_native_write - finite_write(native_input)
w_j = generation + state + replay.
```

Generation here is the local approximation discrepancy at the *forecast* state;
it includes the explicitly recorded Float32 input-installation convention.
State is the effect of the different receiving inputs through the same finite
operation. Both are conditional on this specified path; they are not unique
causal percentages. Recompute native outputs within the existing1e-4 bound,
and independently reconstruct both finite operations from checkpoint tensors
in NumPy at1e-4. Derivative-free: no new attribution through disputed Jacobians.

## Interventions and unchanged finite final readout

At the original F input to MLP24, add terminal receiving vectors from: TOP1,
TOP2, TOP4 write errors, their GEN4/STATE4/REPLAY4 splits, or the complete error.
Run MLP24 and blocks25–28 plus the final norm/head finitely. Controls: original
F replay and three independent row-norm-matched random orientations for TOP2
and TOP4. All14 arms, all32 cases retained. Final norm/head stay finite.

These are **terminal-vector interventions** testing the consequence of an
accounting component. They do not repair the selected upstream generator or
recompute the forecast trajectory. Oracle native writes are used explicitly;
success is not prospective prediction or a compact mechanism.

Primary descriptive TOP4 account: baseline parent and final-verdict vector
errors>=.001 and absolute margin error>=.10 nats; TOP4 halves all three and
strictly beats all three TOP4 random controls on each metric. Require24/32
and6/8 in each schema/order stratum. Report ineligible cases separately.
TOP1/TOP2 and GEN/STATE comparisons are secondary. Full repair must recover
saved native final coordinates within1e-4 and scores within.005; baseline must
replay original F under the same bounds. Independent finite-head checks.005.
No tolerance changes or silent replacement of failed checks.

Expected computation after model-free ledger:32 ordinary False, zero ordinary
True;256 local finite selected operations;32*(1+number_of_selected_MLPs)
source-only attention calls;448 finite MLP24 updates,1792 suffix blocks and448
heads. Additional NumPy finite reconstructions are recorded separately. All
sources, raw inputs, local split vectors, installed states, outcomes and failures
are hashed. Stop on failure and preserve it. Earlier numerical boundaries remain.

The purpose is to decide whether final failure is concentrated in a few write
errors and whether those errors mostly reflect local approximation or wrong
incoming states. A successful local attention rule in234 does not imply either.
Any new False-only update suggested here needs a separate frozen fresh test.
