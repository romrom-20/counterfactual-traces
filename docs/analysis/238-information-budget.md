# Clarification: known source computation versus oracle receiving states

The original228 source-token states were computed **before** ordinary native
True trajectories were measured. They are declared counterfactual component
work, not a leak from later native observations. Some later diagnostic wording
called the cached native-equal source rows oracle information; that is more
restrictive than their actual provenance requires.

[The sealed228 runner](../../research/boolean_coupling/alignment228.py) starts
from the known edited token embedding. At each block it constructs a full input
with the False prefix, the current edited source row and False receiving rows,
then advances the source row. In causal attention, the source cannot read later
receiving positions. Positions before the edit are unchanged. By induction the
computed source row equals its native True value at every block without knowing
any later receiving state.228 subsequently checks that equality against native
measurements. Its forecast seal and source-work counts predate those measurements.

The complete first-block receiving response is also computable from the known
edit: at its input, every other token embedding is unchanged. The original
source-only first block supplies that exact response. Reusing its saved value
is a declared computation budget, not an unknown native receiving-state oracle.

In contrast, the native receiving input beforeMLP20 and native-input local
remainders throughout the network are **genuine oracle diagnostics**. They are
not determined by the one-position source recurrence; predicting them is the
unresolved problem. Retrospective selection based on those native measurements
also cannot count as a fresh forecast.

This clarification changes no old source, input, outcome or numerical bound.
It does not imply a computational speedup: the original source routine performs
full-sequence component calls and its entire cost remains counted. A fresh
prediction must generate the allowed source/first-block quantities from the
known edit, never read unknown native receiving states, seal its rule before
new native measurements, and declare all computation.
