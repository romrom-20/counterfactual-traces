# 238: tracing upstream generators helps, but four operations do not close Question B

**Recomputing four identified upstream MLPs reduces median final margin error
from 5.146 to 1.340 nats and whole-displacement error from 72.3% to 31.6%. It
still meets the full-response criterion in 0/32 cases.** Supplying their oracle
native-input remainders also fails all32. The four largest generators are useful
repair targets, but are not a sufficient explanation of the complete response.

This is a qualified retrospective test on the32 previously observed228 contexts.
All declared numerical checks pass, including separate NumPy finite endpoints.
The older237 failures remain failed; this study does not recertify them.

## A different quantity from the previous write ledger

235 measured discrepancies between predicted and native writes. A large write
error can reflect an operation receiving the wrong input rather than making a
large local approximation error. Here the target is the wrong receiving state
arriving before MLP20, which237 showed cannot be fixed by exact late computation.

Before that boundary, the originalF forecast is affine at all37 operations from
attention2 through attention20, including MLP2–19. Its first-block receiving
state is exact under the declared known-edit computation. Write the native-minus-
forecast input error as e, the False Jacobian as J, and the native-input local
remainder as R. The exact recurrence is:

`e_next = J e + R`.

R is calculated at the native input; it is an oracle diagnostic. A backward
adjoint transports the terminal error direction through the same originalJ
maps. Each inner product withR gives that generator's signed contribution to
the squared terminal error. Those contributions sum toone after normalization,
within **6.89e-15**. They are defined transport projections, not learned features,
unique causal percentages, or complete transported-vector norms.

## The four largest generators are MLPs16–19

A fixed rule ranks all37 operations by median absolute per-case signed terminal
projection. It selects **MLP18,16,19,17**, in that order. Selection and all ledger
records are audited, hashed and committed before intervention outcomes at
`af75422f15ac02aeb5fd7e204e6fb6103a30729a`.

Their median signed shares are .170,.155,.125,.093. The median sum per case is
**.541**, ranging from .428 to .646. These generators differ from the late MLPs
selected by235's write ledger. They identify actual local approximation
remainders upstream of the inaccurate reset input.

A large projection does not promise vector sufficiency. The four oracle
remainders added at their original prefix sites, then transported by the
original affine maps, lower the parent error only from **19.96 to14.18**.
The original parent is still far from native even with these oracle terms.

## Oracle diagnosis versus a repair at its own inputs

The five arms share a finite MLP20, blocks21–28, final normalization and head.
The target is always the original native True residual and margin.

- **Original prefix:** parent error19.96; final whole relative error72.3%;
  verdict-vector error58.45; margin error5.146nats.
- **Oracle four-remainder injection:** parent14.18; whole31.2%; verdict-vector
  error18.55; margin .861nats. The actual native remainder vectors are supplied.
- **Four finite operations at the forecast's own changing inputs:** parent14.49;
  whole31.6%; verdict-vector error20.70; margin1.340nats. Later states respond
  to each intervention; corrections are not frozen at the terminal boundary.
- **Row-norm-matched orientation control:** parent23.48; whole73.9%;
  verdict-vector error58.79; margin5.054nats. Each control correction is matched
  at that control's own input with a fixed SHA seed.
- **Native input reset:** parent, residual and margin errors are exactlyzero;
 32/32 accuracy. This is a replay control, not a forecast.

Every non-native arm has **0/32** joint accuracy. The frozen bound requires
relative error<=5% and cosine>=.99 at BOTH the entire receiving sequence and
verdict, plus margin error<=.10nats, with24/32 overall and6/8 per stratum.
Both oracle/own-input arms also fail the stricter three-metric halving rule in
all cases because their parent errors do not halve. The improvements are real;
they are not an accurate native-response prediction.

![All generators and every intervention case](../figures/238-upstream-generators.png)

## A useful remaining distinction: generator values depend on the incoming state

An explicitly descriptive comparison examines the finite-minus-affine
correction at each own-input repair site versus the native-input remainder.
At MLP16 their median whole-vector cosine is **.781**, with own/native norm
ratio **.480** and relative difference **.696**. At MLP17/18/19 the corresponding
relative differences are **.658/.639/.484**. These are not interchangeable local
vectors, despite the similar final oracle/own-input aggregate errors.

This does not establish that state dependence alone caused the failed repair:
the four-generator oracle itself failed. Other generators remain missing, and
positive scalar projections do not close the vector. It does identify a concrete
local question: can the False computation predict how a generator changes with
its input? The subsequent239 test freezes normalization, SiLU and gate–up
curvature formulas for that contrast, rather than starting another layer sweep.

## Information budget and numerical qualification

The source-only states in228 were computed from the known edit before native
True measurement. They are allowed component work, not leaked future states.
The first-block receiving state is also computable under that declared budget.
Native receiving inputs and nativeR throughout the network remain genuine
oracles. [Code-based information-budget clarification](238-information-budget.md).

All **8,416 ledger-audit checks** and **20,000 intervention-audit checks** pass.
Independent NumPy analytic derivatives reconstruct every used direction and
native-inputR at1e-9. Reverse bilinear identities have maximum error **2.96e-12**.
Every original baseline/source/native replay is bit-exact.

There are **4,768 used JVP/central-FD comparisons**, with9,536 Torch finite
endpoints. The separate NumPy finite-endpoint audit also evaluates all **9,536
endpoints**, yielding9,536 additional comparisons to saved tangents/FD vectors.
Its maximum coordinate discrepancy is **5.69e-9**. Actual runner JVP/FD error
is at most **7.31e-9**. No publicFloat32 whole-block derivative is used.

Selected finite MLPs get **256 independent Float32 primal reconstructions**
within1e-4. All256 raw Float64 mathematical bridges also pass the same bound;
none is relabeled. All160 finite heads pass.005. False smooth/public primals
have maximum discrepancy **4.14e-5**. These checks do not resolve earlier
Float32 derivative disagreements or237's failed bridges.

Original238 and238b stopped before any model loading because bootstrap metadata
checks used the wrong schema. Both original sources and tracebacks are preserved.
238c corrects only those checks; the original scientific protocol, inputs,
rank rule, criteria and interventions remain unchanged.238d adds independently
reconstructed finite endpoints without changing any bound.

Actual new ordinary models:64False,0True across the two stages. Component work:
608source blocks,4,768JVPs/9,536FD endpoints,1,184VJPs,256selected finite
operations,160finiteMLP20 calls,1,280suffix blocks,160heads. Independent NumPy
endpoint work and reconstruction are additional computation. This is not a
speedup or a learned circuit.32old dependent contexts and one checkpoint do
not establish generality, priority, or competent-verifier shutdown.

## Reproduce

[All cases, source seals and commands](../../results/upstream-remainder-238/README.md)
· [Frozen protocol](../experiments/238-upstream-remainder-transport.md)
· [Corrected runner](../../research/boolean_coupling/transport238c.py)
· [Separate auditor](../../research/boolean_coupling/audit238.py)
· [Independent finite endpoints](../../research/boolean_coupling/auditfd238.py)

```sh
python -S research/boolean_coupling/reproduce_upstream.py --verify-only --history --plan
```

No article draft, checkpoint weight or activation array is included inGitHub.
The complete fixed-input chain is regenerable from the public pinned checkpoint.
