# 286b: Numerical qualification of the 286 source adjoints

This is a retrospective numerical control, after the 286 outcomes were observed.
It does not change 286's source seal, prediction rule, endpoints or tolerances.
In particular, all 192 original cached-versus-full coordinate diagnostics above
1e-4 remain failed. The separately declared cached replay bound was .002.

Before checkpoint execution, freeze this protocol and its three programs. Use
all the same 32 cases, no selection. Observe one ordinary False full computation
per case, retaining each block's causal prefix state. Regenerate the known M
first block with the same False-SiLU-input clamp and forecast write. Neither a
native True state nor a native True gradient is needed for this qualification.

For F and M, recompute blocks 2–28 with full sequence shapes, concatenating the
unchanged causal False prefix before the current suffix at every block. Retain
the suffix inputs. Differentiate the finite full-vocabulary head, then compute
one full-shape block VJP at a time in reverse order. Prefix output adjoints are
zero because no causal prefix output depends on a suffix input. Parameters are
constant; Float32 rounding is treated as identity by autograd, as in 286.

Required checks, fixed before this qualifier's checkpoint run:

- Ordinary full False source/final/scores and regenerated M source equal the
  original 285 values exactly. The unaffected prefix remains exactly unchanged.
- Full-shape F/M final residual, candidate scores, and reverse recomputation
  primal errors are at most 1e-4 in every saved coordinate.
- At every block input 2–28, all coordinates of the source-row margin adjoint
  agree with the cached 286 adjoint within 1e-4 absolute error.
- The dot product with the original C-minus-M source direction agrees within
  1e-4 nat for both F and M source adjoints.

Save every case even if a numerical bound fails; audit and preserve that failure
rather than broadening the bounds. Record whole-suffix adjoint norm comparisons
descriptively. These are reverse adjoints for one scalar margin, not Jacobian
matrices. Both executions use the same installed model/library primitives;
agreement is not independent validation of every primitive derivative. The
existing 286 finite-difference checks remain the independent projected check.

First use the existing unequal-length tiny CPU fixtures, compare blockwise
adjoints against a full-model full-tape reverse calculation at every saved
source-row coordinate, and require 1e-4 agreement. Then freeze all sources.
Use 768 MiB initial / 512 MiB per-case free-space guards and a 32 MiB new-array
cap. Count ordinary observations, first-block calls, finite block forwards,
reverse block recomputations/VJPs and head VJPs explicitly. No speedup claim.
