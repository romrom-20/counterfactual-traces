# 237c: finish inspecting237b without converting failed checks into passes

237b's runner completed32 cases. Its frozen independent auditor stopped on
`numpy_finite_mlp`: error0.00011064888120415617 >1e-4. Preserve the original
runner, auditor, protocol, metadata and failure. No237b audit pass is claimed.

This is a posthoc numerical diagnostic of those unchanged saved arrays, not a
new experiment or prospective success. Run every original audit comparison,
retain its original bound, and record all failures rather than stopping at
the first finite (non-NaN) discrepancy. Nonfinite values still stop execution.
Record case and operation for every check. The final status must explicitly
indicate any failed checks, even if all other checks pass.

Also compare each finite MLP with an independent NumPy Float32 reconstruction,
using the same inputs/weights and same1e-4 bound. This tests whether the public
Float32 arithmetic is closer than the mathematical Float64 evaluation; it
cannot rehabilitate an original failed Float64 comparison. Report both domains
and all failures separately. No new tolerance, example exclusion, correction
or native intermediate substitution is permitted.

All original outcomes are retained as conditional diagnostic measurements.
Do not describe them as a fully qualified237b mechanism result while relevant
checks remain failed. The relevant attention graph/derivative checks may be
reported by their actual pass/fail counts, with the MLP bridge limitation visible.
If needed, separately replay public full-sequence MLP calls; label such a replay
as same-backend execution, not independent mathematical reconstruction.
