# 264b — complete block-output indexing

Preserve the original264 source, raw output and numerical audit. Its final
vectors and margin contrasts use the correct final output, but its layerwise
loop mistakenly treats the 28 block outputs as if index0 were an embedding.
It therefore omits block1, and labels blocks2–28 as1–27. The numerical identity
checks pass but do not validate this coverage claim. Original264 is not a
complete or correctly labeled layerwise result.

This correction changes only the iteration to indices0–27, explicitly labels
those outputs as blocks1–28, and asserts all28 layers and exact row counts.
The scientific definitions, fixed tolerances, old contexts, input evidence and
all final contrasts in the [original protocol](264-normalization-component-interaction.md)
remain unchanged. No model forwards or new native measurements. Freeze this
separate source before the corrected analysis; do not overwrite264.
