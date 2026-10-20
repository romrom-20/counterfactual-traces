# 246b implementation correction

Original246 stops in its first case after the four regional replays, before
local-star calculations/interventions, because MPS does not support Float64.
Preserve its code, protocol, failure log and partial arrays. 246b performs the
already specified Float64 additive-state sum on CPU, casts to Float32 and
returns it to MPS. Same formulas, sites, primary, bounds and32 cases; no outcome
was scored from246. The separately named scorer accepts the corrected run ID.
No other scientific change. Freeze corrected sources before execution.
