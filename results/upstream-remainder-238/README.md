# 238: upstream remainder transport and targeted generator repair

All37 operations beforeMLP20 are traced with an exact affine-error recurrence.
MLPs16–19 are selected by a frozen rule before intervention outcomes. Recomputing
them at the forecast's own inputs reduces final whole error72.3%→31.6% and margin
5.146→1.340nats, but passes the native-response criterion in0/32. The oracle
four-remainder arm also has0/32. Native reset replays exactly in32/32.

[Complete report](../../docs/analysis/238-upstream-generators-result.md)
· [Frozen protocol](../../docs/experiments/238-upstream-remainder-transport.md)
· [Information budget](../../docs/analysis/238-information-budget.md)

All32 ledger records,32 intervention records, all five outcome arms, FD direction
witnesses and checks, rank/selection seal and source snapshots are retained.
`ledger.csv` contains signed transported projections and local remainder norms.
`native-own-remainders.csv` and its summary are explicitly descriptive comparisons,
not a new accuracy criterion. Both original bootstrap failures remain preserved.

All declared checks pass:8,416 ledger,20,000 intervention,plus9,536 independent
NumPy endpoint comparisons. Original237 numerical failures remain unchanged.
CI checks integrity and syntax; it does not rerun the model or certify new science.

Verify without importing model libraries:

```sh
python -S research/boolean_coupling/reproduce_upstream.py --verify-only --history --plan
```

Regenerate the complete fixed public-input chain:

```sh
python research/boolean_coupling/reproduce_upstream.py --run --out runs/my-upstream-test
```

Reuse an existing local237b prerequisite with its original preserved failures:

```sh
python research/boolean_coupling/reproduce_upstream.py --run --offline \
  --input-237 runs/existing-237b --out runs/my-upstream-test
```

The wrapper runs ledger, analytic audit and independent finite endpoints, seals
the selection before outcomes, then runs interventions and both auditors.
Every output directory must be new. Unexpected failures stop with original logs.
Reusing old contexts is retrospective replication, not fresh confirmation.
Use the [recorded MPS setup](../../research/README.md); no private data/API key
is required. All component and prerequisite work remains counted.

The cached checkpoint is hash-bound without a duplicate weights file.64raw
activation arrays total1,804,836,462bytes and remain local/inventoried. Article
drafts and large arrays stay out ofGitHub. Repository visibility remains private.
