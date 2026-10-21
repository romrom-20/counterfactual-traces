# 252 — source-driven routing versus receiver query/key feedback

251b's fixed receiver probabilities fail despite an exact known-edit source.
This does not localize the missing adaptation to receiver queries or keys:
the source key itself can change weights.250b tested source keys only after
block9, and cannot exclude important earlier source-key changes.

Freeze a new three-arm False-only forecast before any measurements on32 new
literal pairs. Same pinned checkpoint, dependent templates, matched lengths,
fixed decoder and final finite readout.64 hashes must be absent from the union
of published catalogs/cases, including251. No model forwards during fixture creation.

Cache False rotary-applied receiver queries and receiver keys at all28 blocks.
The known edited source and unchanged causal prefix retain finite computation
in every arm. Their states/keys/values require no oracle inputs. Receiver values,
RMSNorm, MLPs, residual updates and relays always evolve finitely at forecast states.

- **A, primary:** receiver queries AND receiver keys remain False. Source key
  stays live, computed from the known edit. Softmax stays finite, so routing
  can change from the source alone. This tests a source-driven routing account.
- **Q, secondary:** receiver queries live; receiver keys False. Source key live.
- **K, secondary:** receiver queries False; receiver keys live. Source key live.

Only attention outputs for queries strictly after the source are installed.
Prefix/source outputs remain native finite; all three source trajectories must
later match native True exactly. Recompute QK/softmax/PV at original full query
shape. Original attention computations also run and must be counted; no speedup.

Primary A: whole-receiver AND verdict displacement error<=.05, cosine>=.99,
target norms>=.001 and margin error<=.10 nats;24/32 overall and6/8 each
schema/order stratum. Q/K use the same secondary rule and cannot replace A.
Report every case/arm/layer and signed effects, without fitting or depth search.
Nonfinite states or norm>100 times False stop and retain the failure.

All forecasts globally sealed and separate pre-audit passed; push the forecast
receipts/seal to the public feature branch and verify its commit before native
True measurements. Then native T and an oracle/current-QK replay O are run.
O recomputes current QK/softmax/PV at full shape and must replay native T exactly.
Zero-edit recomputation must likewise replay False exactly. New source trajectories
are computed internally, not supplied from T. Reproduction uses explicit local proof.

Numerical qualification inherits251b's explicit boundary BEFORE this new run:
raw scaled-QK absolute1e−4 checks are recorded, including failures at large
first-block logits, but do not gate measurement. CPU64 composed QK→softmax→PV
probability/output checks remain1e−4; endpoint heads .005; cached Q/K installation
and replay controls exact. Retain every raw-check result. Witness sample is first
case of each(schema,reverse) tuple, blocks1/14/28, all7arms. This is a bounded
kernel audit, not an independent full-forward reconstruction or derivative check.

Budget:160 forecast-phase full forwards (F/Z/A/Q/K),64 native/control forwards
(T/O),4,480 extra QK/softmax/PV products EACH (Z/A/Q/K/O),84 extra witness QK.
Every raw array stays local with public hashes; complete code, logs, failures,
receipts and outcomes are published. No learned circuit/feature or fresh task-family
generalization follows. No article draft is uploaded.
