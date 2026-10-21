# 251b — qualify composed attention outputs; retain raw-logit failure

Before any native True forward,251's pre-auditor failed its absolute1e−4 raw
QK-logit coordinate check. Saved first-block native False logits reach about
19,575; their CPU64/MPSFloat32 differences reach .00662. This also occurs in
the identical zero control, not just forecast states. Do not declare that raw
check passed or silently widen its tolerance. Preserve the original auditor,
failure and all forecasts. An earlier disk-space-guard attempt also remains.

The new qualification evaluates the composed operation that generates attention
outputs. From the saved Q/K/V, independently compute CPU64 scaled QK, masked
softmax, selected cached-probability installation and PV. Require independently
computed current probability AND installed attention output to match actual
Float32 witnesses within the existing1e−4 output tolerance. Also retain the
original softmax-from-runtime-logits and PV-from-runtime-probabilities checks.
Actual cached probability installation remains exact; endpoint heads remain
at .005. Record **every raw QK check with its original1e−4 bound and failure
status** separately. The raw-logit criterion remains unresolved at that bound.

This revises the object being numerically qualified, not merely its threshold:
composed probability/output accuracy replaces universal raw-logit coordinate
agreement as the measurement gate. A probability/output mismatch still stops
the experiment. It is not a full independent forward reconstruction. The
first four-stratum witness sample and blocks1/14/28 remain fixed.

No forecast is recomputed, selected, excluded or tuned; no behavioral primary,
secondary, vector, margin or success threshold changes. The new auditor and
this boundary are committed before native True data exist. A qualification
receipt binds the old failure, design, forecast status and new source hashes;
the global seal and public pre-native receipt then bind this revised pre-audit.
Do not call this the original251 numerical protocol passing unchanged.
