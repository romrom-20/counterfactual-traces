# 251 — can False attention weights transmit the native response?

250b establishes a retrospective necessity result for continued source K/V
communication, with value interventions larger than source-key interventions.
It leaves receiver queries, other keys and attention weights adaptive. The new
hypothesis is that receiver attention-weight changes are unnecessary for an
accurate False-only native forecast if values, residual updates, normalization,
MLPs and relays remain finite.

Use32 tokenization-only new literal pairs,64 hashes absent from the published
historical catalog. They reuse existing dependent synthetic templates: new
literals do not establish task-family generalization or independent replication.
Pin the same Qwen checkpoint, decoder, Float32 MPS/eager and tolerances. Freeze
code, fixtures and this protocol before any model forward. No derivatives.

## Forecasts from one False computation and known edit

Cache native False attention probabilities at the source and all later query
rows, block states and scores. Run a zero-edit control with these cached weights
on all tail queries; it must replay every False tail coordinate and score exactly.

For the known True input edit run three independently evolving forecasts:

- **R, primary:** all receiver queries strictly after the source use the cached
  False probabilities at every block1–28. Source attention stays finite, so its
  trajectory is computed from the known edit and unchanged causal prefix.
- **S, secondary:** only the source query uses cached False probabilities;
  receiving attention stays finite at its own forecast inputs.
- **B, secondary:** source and all receiving queries use cached False probabilities.

Every value projection, RMSNorm, MLP, residual update and final full-vocabulary
head stays finite at the forecast's own state. Prefix queries remain untouched.
Recompute PV at the full original query shape, install only selected outputs,
and keep masked cached probabilities exactly. Original finite QK/softmax is
still computed by this implementation even when receiver outputs are unused;
declare all this work. No speedup or compact-circuit claim.

R tests `delta attention = P_False * delta V` across receivers while preserving
the source's finite computation. Relay values evolve too. It is NOT a linear
residual/Jacobian forecast. No unknown native True states or attention maps are
permitted in these three forecasts.

Primary R: native whole-receiving displacement AND verdict displacement each
have relative error<=.05, cosine>=.99, target norms>=.001, and absolute finite
margin error<=.10 nats. Require24/32 and6/8 in each schema/order stratum. S/B
use identical secondary criteria; they never replace the R primary. Report
signed vector projections, errors, norms, margins and source-row errors for
every arm/case/layer. No tuning, depth/operation search or outcome-based selection.
If unstable (nonfinite states or forecast tail norm>100 times False tail norm),
stop, preserve the failure and do not reinterpret missing forecasts as success.

## Seal, then native measurements and controls

Seal all32 forecasts globally, pass the separate pre-audit, then push the
forecast receipts/seal/source hashes to the public feature branch and verify
the remote commit before the first native True forward. A separately timestamped
proof records that commit. Reproduction can use an explicit local-replication
proof instead; it is not a fresh first test. Do not upload arrays or articles.

Only then run ordinary native True, capturing its probability maps, followed by
an **oracle control O** using native True cached tail probabilities. O must replay
every native True tail coordinate and score exactly. This is implementation
qualification, never a valid forecast. R's source row must match the now-measured
native True source exactly at every layer. All causal prefixes must replay False.

Use the native full finite target to examine T−R−S+B as a secondary finite
interaction, with both full-receiver and verdict scopes; a failed forecast does
not make this a learned circuit or identify a particular nonlinear primitive.

## Separate numerical audit and explicit budget

Save full tail trajectories/endpoint scores and False/source prefix hashes.
Save tail attention probability tables locally for F and T. For first case of
each schema/order tuple (exactly4), save Q/K/V/raw/mask/current/installed P/Y
witnesses at blocks1,14,28, all7 arms F/Z/R/S/B/T/O. Separately audit CPU64 QK,
softmax and PV with1e−4 tolerance and every finite endpoint head at.005. Also
audit cached probability installation and query grouping exactly. This is bounded
primitive witness coverage, not a full independent forward reconstruction.

Forecast budget:32 F +32 zero +96 R/S/B =160 full forwards and3,584 extra full
PV products (zero+R/S/B). Native measurement/control:32 T +32 oracle O =64 full
forwards and896 extra full PV products.84 witness extra QK products overall.
The source trajectory is included in each forecast's ordinary full computation;
there is no separate source-row solver and no leaked oracle source input.
Separate audits are additional work. Every failure, seal, source snapshot and
array hash is retained, and the publication includes all outcomes, not just
successful metrics. Article drafts remain local.
