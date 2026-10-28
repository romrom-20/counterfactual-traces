# 292c: one resident model, unchanged predictor and native gate

292b stopped before its first archive because the host reserve was below the
unchanged 384 MiB guard after loading. Preserve that complete stopped attempt,
as well as original 292's first-case stop. Neither attempted native True.
This execution policy does not change the 102 predictor bytes, inputs, dtype,
weights, numerical bounds or any original disk guard.

Use the ORIGINAL loader once. After it returns, release only unused MPS cache
and unreachable/unused host allocator memory, then wait at most 180 seconds for
640 MiB free disk space. The 640 MiB floor is the original 384 MiB reserve plus
the maximum 256 MiB new archives. A timeout stops and preserves the attempt.
Waiting is not permission to report fictitious disk space or bypass a guard.

Keep this same eval-mode Float32 MPS instance resident through forecasting,
qualification, sealing, waiting for publication, native measurement and audit.
Only weight reuse changes; full forward counts and all computations stay those
of 292. Require stable parameter identities/versions and exactly one model load.
No modified or direct-device loader, quantization, lower precision or different
checkpoint is introduced.

292b's exact 42-array first-case replay executor remains mandatory before any
remaining case. Its mismatched replay fixture must remain rejected. Validate
the resident loader's one-factory-call identity, unchanged parameter objects,
actual headroom threshold and timeout behavior on tiny CPU before execution.
Freeze this policy/executor before checkpoint use.

After all forecasts qualify and seal, the process waits for an explicit receipt
file supplied only after the exact seal is committed and pushed. The ORIGINAL
native entry verifies the current Git head, remote branch head, source/array/
metadata hashes and public seal before the first True call. A resident model
never authorizes bypassing this global gate. The staged process stops if a
receipt does not arrive within an hour.

Before native measurement additionally require the original 384 MiB reserve
plus the remaining archive allowance. The original per-case guards still run.
Preserve every wait, failure and maintenance receipt; no original evidence,
array, weight or private article is removed. Report model residence and timings
explicitly; this is a resource continuation, not a new scientific candidate or
an independent replication.
