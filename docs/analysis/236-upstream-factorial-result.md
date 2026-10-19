# 236: repairing MLP20/21 helps, but later attention still prevents closure

**The targeted MLP20/21 repair meets the full native-response criterion in
0/32 cases.** Recomputing both operations at the forecast's own changing
inputs lowers median margin error from **6.588 to 4.849 nats**, but the final
whole-receiving displacement still has **74.0%** relative error. Local
approximation errors at these two MLPs are not a sufficient explanation of the
failed end-to-end forecast.

The stronger diagnostic is the native reset. Starting at the exact native input
to MLP20, evaluating every remaining MLP finitely, and retaining only the four
False-affine attention updates 21–24 leaves **0.344 nats** median margin error
and **24.4%** whole-receiving relative error. It also passes 0/32. A fully finite
continuation from the same reset recovers the native result exactly in 32/32.
The remaining attention approximations are therefore insufficient even with
the correct incoming state supplied.

## What changed relative to 235

235's terminal-vector test did not update later forecast states. Here, a genuine
continuation branches at the original F forecast's MLP20 input. Two switches
make MLP20 and/or 21 finite at each branch's own current state. All later states,
attention directions and finite MLP22/23 writes respond to those switches.
The final MLP24, blocks 25–28, normalization and two-score head remain finite.

Each incoming regime includes 00, 20-only, 21-only, both and one independently seeded
row-norm-matched orientation control. The second incoming regime supplies an
oracle native MLP20 input. Those oracle branches are explicitly retrospective
diagnostics, not valid prospective forecasts. All 32 previously observed 228
contexts are included; no new native True model is run.

## Ordinary incoming state: partial improvement, complete accuracy failure

Median final margin errors are **6.588 nats** unchanged, **6.088** with finite
MLP20 only, **5.263** with finite MLP21 only, **4.849** with both, and **6.580**
with the row-matched control. Joint repair reduces the final verdict-vector
error from **104.48 to 59.55**, and parent error from **44.13 to 35.45**.

Those improvements are well outside numerical uncertainty. They still miss the
frozen full-response bound: relative error<=5% and cosine>=.99 at both all
receiving rows and verdict, plus margin error<=.10 nats. Every non-oracle arm
passes **0/32**, including the joint rule. Its median whole relative error is
**0.740**, cosine **0.692**; verdict relative error **0.812**, cosine **0.606**.
The targeted two-operation hypothesis is rejected at these tested bounds.

## Exact incoming state: later approximations introduce substantial error

With a native reset, the same four modes give median margin errors **4.295**,
**3.449**, **1.043**, and **0.344 nats**; the random control gives **3.877**.
The joint branch's whole relative error is **0.244**, cosine **0.971**, with
verdict relative error **0.241**. Every approximate oracle branch also passes
**0/32**. These are failures of the tested continuation even after early carry
error is removed.

Only the fully finite oracle control passes 32/32, with zero parent, final-vector
and score replay error. In oracle-both, MLP20–24 and the complete final suffix
are already finite. The only remaining approximation sites are attention 21–24.
This narrows the unresolved late computation without identifying which of its
normalization, QK, softmax or probability–value operations is responsible.

The native reset substantially improves the joint forecast relative to the
ordinary incoming regime, so earlier state errors matter too. This does not
prove an additive or unique division of responsibility between early carry
and later generation; the network response is state-dependent.

![Every recorded case and control](../figures/236-upstream-factorial.png)

## Qualification and information boundaries

All **10,016 separate audit checks** pass, including **1,664 actual used
JVP/FD comparisons**, **1,024 NumPy finite-MLP reconstructions**, all update
equations, controls, and **352 finite heads**. Central FD uses h=1e-5 and the
unchanged 1e-4 bound. Maximum recorded primal/FD coordinate discrepancy is
**8.894e-5**; analytic update identities retain 1e-9 and score bridges .005.
Original F and fully finite native controls replay exactly. No tolerances,
inputs, directions or outcomes were changed after measurements.

Actual work: 32 ordinary False, 0 ordinary True, 160 source-only attention calls,
1664 JVP/FD comparisons/3328 endpoints, 1376 finite MLP updates, 1536 continuation
blocks, 352 heads. These component computations do not establish a speedup.
Old Float32 derivative disagreements and bridge failures remain unqualified.

## Why many local results have not answered Question B

The investigation had localized several conditional effects, and 234 forecast
one early attention update accurately. Neither establishes that local accounts
compose into the native final displacement. Here the largest late errors are
mostly responses to inaccurate inputs, yet providing accurate inputs still
leaves substantial later approximation error. Fixing one side does not close
the other.

The next mechanistic target is therefore the finite receiving response in
attention 21–24 under this qualified reset—not another demonstration that a
selected late MLP or final normalization matters. Any operation account must
improve both the vector and finite margin, then predict fresh trajectories
before their True states are observed. This study is a qualified negative
result on dependent synthetic templates in one checkpoint, not a learned
circuit, priority claim or solution to end-to-end Question B.

[Frozen protocol](../experiments/236-upstream-mlp-factorial.md)
· [All branches, derivatives and reproduction](../../results/upstream-factorial-236/README.md)
· [Global error budget](235-global-error-result.md)
