# 281d: resume path binding correction

281c's synthetic preflight caught a local-variable collision: the partial-array
Path variable was overwritten by the reference NPZ reader named prior. The same
281c checkpoint continuation was inadvertently launched without gating on that
preflight result; it failed with the same AttributeError after writing its case31
array, before metadata or any scientific conclusion. Preserve both failed logs,
original code/protocol, data and failure metadata. Do not call281c a successful run.

281d renames only the partial-array Path variable. Archive reader, products281
Engine, all tensors/masks/seeds,80MiB cap, disk reserves and original audit281/
scientific/numerical criteria remain unchanged. It resumes the original281b30
sealed cases, regenerates31/32, and requires exact replay of the original orphan31.
Freeze before retry. An expanded synthetic CPU fixture must pass BOTH a cold32-
case producer AND a30+2 resumed producer including orphan hash replay before the
checkpoint is run; no checkpoint or real study cases enter that fixture.

Record qualification281d-resume-path-binding. The original281 bookkeeping failure,
281b cap failure and281c binding failure remain in the public record. Completed
study budget576forwards; additional failed overhead is2 original plus18 orphan
replay in281b and18 in281c (=38), before the successful30+2 dataset. Synthetic
CPU fixture calls are separately labeled and not checkpoint evidence. No numerical
or scientific failure is repaired by the execution qualifications.
