# 234: fresh source-key/value forecast

The joint source key/value rule passes the all-receiving attention-2 criterion
in 32/32 fresh pairs, 8/8 per stratum. Median relative error is 1.520%, versus
5.807% for old FNV. Removing the mixed product raises error to 2.472% and leaves
16/32 accurate. This is a local forecast, not final residual/margin prediction.

[Report](../../docs/analysis/234-source-key-value-result.md)
· [Frozen protocol](../../docs/experiments/234-source-key-value-forecast.md)

Every arm and case is retained in `cases.csv`. The complete numerical
qualification and all source/input/prediction hashes precede native True
measurement. The forecasts were pushed and remotely verified at
`ab7234305bbab591820360c65198e3a0a207ee69`; the global seal and original receipt
remain. Source snapshots, all forecast/observation JSONs, phase logs, witness
hashes and both raw-array inventories are preserved. There are 1,216 passing
pre-target checks and 384 derivative/FD comparisons, using 768 FD endpoints.

The budget includes an exact edited block 1 and one finite source-only attention
call per pair, plus one ordinary False and substantial local arithmetic. The
receiving transport uses False softmax operands; the public source-only primal
still has finite source-dependent probabilities. This is not a zero-counterfactual
computation method or a speedup. All native first blocks replay exactly.

Verify every release and retained history without model dependencies:

```sh
python -S research/boolean_coupling/reproduce_transport.py --verify-only --history --plan
```

After the [recorded MPS environment setup](../../research/README.md), reproduce
from fixed public inputs and the pinned checkpoint:

```sh
python research/boolean_coupling/reproduce_transport.py --run --out runs/my-234
```

Add `--offline` for a cached checkpoint. The wrapper executes forecast, separate
qualification/global seal, native observation and unchanged scoring. It requires
no private archive or GitHub push. Outputs must be new; failures stop the run
and preserve logs. Reusing the original fixed literals is not another fresh test.

Individual phases with `PYTHONPATH=research`:

```sh
python -m boolean_coupling.local234 --phase forecast --out runs/new-234 --offline
python -m boolean_coupling.audit234 --phase forecast --out runs/new-234
python -m boolean_coupling.local234 --phase observe --out runs/new-234 --offline
python -m boolean_coupling.audit234 --phase score --out runs/new-234
```

Raw arrays and weights remain local and are inventoried; the manifest excludes
unpublished writing. This does not identify a learned circuit, recertify earlier
numerical failures, explain block 1 or solve end-to-end Question B.
