# 226b: finite downstream generation versus transmission

Read the [result and boundaries](../../docs/analysis/226b-finite-tail-interaction-result.md) and [pre-execution protocol](../../docs/experiments/226b-finite-tail-interaction.md). All 32 old 225b cases are retained. This is retrospective and derivative-free. Raw arrays remain local under `runs/`; this folder is a compact published record, not an independently replicated model execution. See the [joint release record](../tail-interactions-226-227/README.md) for publication provenance.

With the pinned checkpoint already cached and the 225b environment available:

```sh
PYTHONPATH=research .venv/bin/python -m boolean_coupling.tail226b \
  --input runs/baseline-interaction-225b \
  --out runs/finite-tail-interaction-226b-replication
PYTHONPATH=research .venv/bin/python -m boolean_coupling.audit226b \
  --out runs/finite-tail-interaction-226b-replication
```

The existing [225b reproduction guide](../../research/boolean_coupling/README.md) explains how to regenerate the required parent arrays from published prompt pairs and the pinned checkpoint. The runner refuses to overwrite an output directory. These commands perform model work; hash verification alone is not replication. `design.json` records original absolute input paths and their hashes as provenance, not required machine-specific locations.

`audit.json` contains all maxima, counters and sublayer/schema outcomes. `cases.csv` retains every signed margin, vector norm, random control and failed criterion. `readout.json` keeps the exact finite-head decomposition. `local-array-inventory.json` inventories all 32 local NPZ files. The preserved failed 226 preflight and original 226 code/protocol remain available; its indexing error was corrected with unchanged bounds in 226b.

The follow-up [227b primitive test](../mlp24-mixed-primitives-227b/README.md) narrows the MLP24 contribution but retains its separate raw coordinate failures. Original 204/208/224 qualifications remain in force. No learned-circuit or False-only prediction claim.
