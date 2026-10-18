# Receiver-only curvature: experiment 230

Every registered single-primitive account fails in all 32 observed contexts.
The first receiving row carries the normalization-only correction, but that
correction does not account for complete R. A posthoc normalization/product
combination motivates a separate fresh forecast test; it is not a registered
success of 230.

[Report](../../docs/analysis/230-receiver-curvature-result.md)
· [Protocol](../../docs/experiments/230-receiver-curvature-primitives.md)

`cases.csv` retains every frozen decision, `edges.csv` every signed factorial
edge (1,024), and `positions.csv` all 144 receiving rows. `audit.json` contains
the separate NumPy checks and stratum counts; `cases/*.json` retain every
original qualification record. `composite-description.json` is explicitly
post-observation, including the failed mixed-term-only interpretation. The
manifest binds source, protocol, report and compact evidence bytes. Raw arrays
are local, inventoried by SHA256 and size.

To regenerate prerequisite 229 arrays from public inputs, use
`research/boolean_coupling/reproduce_generator.py --run --out runs/parent`.
Then, in the [recorded environment](../../research/README.md), with
`PYTHONPATH=research`:

```sh
python -m boolean_coupling.run230 --offline --input runs/parent/experiment --out runs/new-230
python -m boolean_coupling.audit230 --run runs/new-230
python -m boolean_coupling.describe230 --run runs/new-230
python -m boolean_coupling.plot_local --kind diagnostic --run runs/new-230 --out runs/new-230/figure
```

Every output/audit must be new. The first two commands are the registered
diagnostic; the latter two describe its saved results without model calls.
The old-context diagnosis is not a new confirmation, a learned circuit or a
False-only forecast. The final-response failure in 228 remains unresolved.
