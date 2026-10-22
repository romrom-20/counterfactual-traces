# 261b — correct metadata destinations and the auditor import

The original261 producer completed28 model forwards and exact F/B/T replays,
but reused a variable holding an original input path when writing case metadata.
It temporarily overwrote four original258 native metadata JSONs and wrote no
new per-case JSONs in261's cases directory. Original arrays were not changed.
The original261 auditor separately fails to import a nonexistent CSV helper.
No new derivative or mechanism claim is qualified by that original attempt.

Preserve the original frozen sources, run/status, actual arrays, overwritten
metadata copies and errors. Restore the four original258 records byte-for-byte
from the immutable public release, verifying every original record SHA256 and
the unchanged original native arrays against the pre-existing status. All64
original forecast/native metadata records must again match their sealed hashes.
The restoration report is retained in261/failure-bookkeeping.json.

Before any261b model forward, freeze a separate producer and auditor:

1. Use distinct case_path and input_path variables. Validate original input
   metadata against its original record hash; never write to original inputs.
2. Every output JSON destination is checked to remain inside the new output
   directory. Each case array/metadata hash binds the actual new case file.
3. Define the CSV helper in the new auditor instead of importing the absent one.
4. Rerun the same four cases and identical seven primitive policies into a new
   directory. Require all original F/B/T and zero controls exactly as before.

No operation policy, model, cases, random seed, numerical tolerance, scientific
question or outcome criterion changes. Original261 remains a failed bookkeeping
attempt.261b is still retrospective and cannot supply a fresh confirmation or
resolve older numerical disputes by repairing these implementation errors.
