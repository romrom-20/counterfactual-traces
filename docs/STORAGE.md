# Model cache and experiment storage

A run folder named `qwen15` is an output namespace, not a checkpoint directory.
Historical runs store activation tensors and intermediate trajectories in its
`states/`, `forecasts/` or `observations/` NPZ archives. Removing that whole folder
can destroy unpublished results; redownloading Qwen does not restore those files.

The provided model loaders use Hugging Face's shared cache. Keep one shared
`HF_HOME` outside experiment outputs, with checkpoint revisions pinned. Snapshot
weight symlinks point at the same cached blob; they are not separate weight copies.
Do not symlink a `qwen15` results folder into the model cache.

For new model jobs, make the shared cache and free-space reserve explicit:

```sh
python research/run_with_shared_cache.py -- \
  python -m boolean_coupling.reproduce296 --offline --out runs/my-repetition
```

Set `PYTHONPATH=research` as required by the scientific entry point. The launcher
sets `HF_HOME`, `HF_HUB_CACHE` and the older hub-cache alias for its child, removes
an inherited deprecated `TRANSFORMERS_CACHE` override, and requires 8 GiB free on
both working and cache volumes. It replaces its process with the command so
cancellation and exit status propagate normally. `--cache-root` can choose another
shared location outside `runs`; `--minimum-free-gib` sets the pre-launch reserve.
It does not copy model weights, override a scientific runner's own guards or
change frozen model calculations. The original loaders already share the cache;
the launcher makes that policy explicit for future jobs.

The reserve is a pre-launch check, not a running output-size cap. New producers
should also bound saved raw data, as recent protocols do. Prefer saving the
specific inputs and vectors needed for the next test over every intermediate
state of every condition. Preserve immutable scientific sources and original
failed checks.

After a historical run is complete and no longer needed as an input, raw
activation archives may be removed with the research owner's authorization.
Before pruning, inventory live producers and open files, require a completed
status and companion result records, preserve logs/designs/launch records,
audits and scalar/vector summaries, and record each removed file's hash and
size. Preserve active, failed/incomplete and still-needed runs. Keep a local
retention marker so later agents do not automatically repeat old experiments
when an intentionally pruned archive is absent.

Published repository verifiers check metadata, source seals and arithmetic;
they do not independently reconstruct omitted activation arrays. Pruning local
arrays leaves those records available but removes the ability to inspect the
original raw vectors locally. Numerical repetition is a separate deliberate
operation; it is not part of routine space reclamation.
