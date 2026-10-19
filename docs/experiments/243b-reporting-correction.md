# 243b: retain the original scorer's zero-count metadata failure

243 completed every model intervention and all live qualifications. The original
separate scorer checked all 32 cases and then stopped on
`KeyError('unstable_forwards')`: the serialized Counter omits zero-valued keys.
The original source, failure JSON, traceback and sealed outputs remain unchanged.

The new audit243b reads an absent unstable count as zero. It retains every
arithmetic operation, input, threshold, qualification bound and outcome rule.
It additionally records both scorer-source hashes in its output. Freeze this
correction before executing it. Rerun the complete saved-array audit, with no
new model computation. The reproduction wrapper uses the corrected scorer;
original243's reporting failure is retained as historical evidence.
