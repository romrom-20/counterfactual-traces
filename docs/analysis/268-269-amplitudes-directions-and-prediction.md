# Signed amplitudes, changing directions, and what the forecast explains

**A frozen signed-amplitude rule predicts the native residual and verdict on
32/32 new literal pairs**, while its stronger old-context interaction account
fails its registered criterion. Median fresh errors are 2.85% for the whole
receiving displacement,2.47% at the verdict, and .03618 nats for the verdict
margin. The reference directions and correction MLP operating points remain
fixed; two signed amplitudes per receiving row and layer adapt. Substantial
finite computation and an extra L forecast remain part of the budget.

The question here is how much of a normalization correction must adapt as the
receiving computation changes. The previous factorial allowed the correction
vectors to adapt while fixing the local MLP operating point. This preserved
the signed final interaction on four old contexts. We now split that vector
adaptation into signed amplitude and direction, then test a separately frozen
prediction rule on 32 new literal pairs.

[268 protocol](../experiments/268-signed-amplitudes-and-directions.md) ·
[269 protocol](../experiments/269-sealed-amplitude-forecast.md) ·
[All evidence](../../results/amplitude-direction-268-269/README.md) ·
[Reproduce](../../research/README.md#signed-amplitudes-and-directions-268269)

![All old interaction cases and fresh native predictions](../figures/amplitude/amplitude-prediction.png)

## What is actually being simplified

At each receiving row and layer, the RMSNorm correction has two terms:

```
p = alpha * f0             f0: weighted normalized False pattern
d = k * u                 u: unit weighted displacement direction
```

With h=x-xF, s²=mean(xF²)+eps, a=mean(xF*h)/s², b=mean(h²)/s² and
rho=(1+2a+b)^(-1/2), alpha=rho-1+a and k=(rho-1)||gamma*h/s||. These are
ordinary normalization identities. The signed-amplitude convention matters:
a sign change belongs to alpha or k; u follows gamma*h with positive norm.

An L forecast first computes its approximate receiving trajectory using
False-tangent MLP input normalization, full finite SiLU and cubic receiver
queries. Cache its local normalized point z*, correction unit direction u*,
and signed amplitudes. No native True receiving state supplies this cache.

The correction's finite MLP responses are M(z*+p)-M(z*) and M(z*+d)-M(z*).
The amplitude-only rule keeps z* and u* fixed, while updating alpha and k from
its OWN evolving forecast state. These are two signed numbers **per receiving
row and layer**, not two numbers describing the whole model. The common MLP
backbone and attention keep adapting; full finite learned computation remains.

The direction-only control fixes the amplitudes and updates u. The both-frozen
control fixes both. The random-axis control keeps alpha and k live, substitutes
a seeded fixed D unit axis, and matches the correction INPUT norm at each own
state. It is not a matched MLP-write or neuron-product control. All neurons use
finite SiLU; the sparse-mask confound from 266 is absent.

In the raw files,268/269 A11 means full generator adaptation at the frozen L
operating point; it equals267 A10. New A10 is amplitude-only, A01 direction-only,
A00 both frozen, Ar random D axis. These digits have a different meaning from
267's generator/operating-point crossing.

## The interaction simplification failed its registered criterion

268 tests the signed final policy interaction `I=A-P-D+L`, where P and D install
one correction and A installs both write-additively. This is a contrast of
defined policies, not the native Boolean response itself. Every family uses
the same adaptive backbone and finite public operations. The target is the
full adaptive-generator interaction at the frozen local operating point.

Amplitude-only passes **2/4**, below the registered requirement of at least 3/4.
Its whole-interaction errors are 28.64%,28.31%,12.44%,12.03% in
generic0/generic1/descriptive0/descriptive1 order. The two generic cases fail
the 25% whole-vector bound; their verdict-vector and finite common-anchor
margin checks pass. Median whole error is 20.38%; verdict error 14.99%.
Direction-only, both frozen and random D axis each pass 0/4. The full-generator
self-comparison is 4/4 by construction, not an additional discovery.

The native-response result is different. Amplitude-only passes 4/4 old examples
under the original 5% whole/verdict error, .99 cosine and .10-nat margin rule,
with 2.702% median whole error and .04729-nat margin error. The full adaptive
generator also passes 4/4, at 3.200% whole error and .05545 nats. Both reference
calibrations pass as frozen. The smaller native error of the simplified rule
does not establish that its interaction mechanism is correct.

There is no contradiction. The native displacement and I have different
targets and scales. Since P is identical between these conditions,
`Iamp-Ifull = (Aamp-Afull)-(Damp-Dfull)`. The retained final-vector ledger
checks that identity. In generic cases the A change has norm 1.59–1.66 against
interaction target norms 5.57–5.86; the D change is only.12–.13. A modest change
relative to the larger native response can therefore be a substantial error
relative to the smaller policy interaction. This accounting does not itself
identify which upstream primitive produced the directional change.

## A simple geometric explanation is also inadequate

After the qualified 268 result, a separate posthoc calculation asks whether
the omitted direction mostly lies in the plane spanned by the cached L
direction and the weighted False pattern. No model forwards are added.

At full-generator A inputs, that plane contains about20–29% of the omitted
directional correction's energy, depending on schema and receiving/verdict
scope. At D-only inputs it contains at most1.34%. All rows and layers,
including zero/guard-ineligible quantities, are retained. Most of the omitted
input-space direction is outside this simple plane. These are geometric
measurements, not MLP-output importance or causal plane interventions. They
do not identify learned features or justify a two-dimensional circuit claim.

## Why run a fresh prediction test after that failure?

The four old examples supported a narrower prediction hypothesis even though
the registered interaction account failed .269 explicitly freezes a SEPARATE
test of that hypothesis. It cannot retroactively pass 268. Its32 new literal
pairs have64 unique full-input hashes absent from 1,856 cataloged sequences;
there are eight cases in each schema/order stratum. These are dependent
templates in one checkpoint, not independent tasks or cross-model replication.

All288 forecast/control forwards and their numerical audit are completed
before any of the 32 unmodified True measurements. The public global receipt
binds source, inputs, scores, final-state/array digests and the preaudit. The
native CLI checks the exact remotely visible receipt commit, then records its
start before measuring T. The primary claim is amplitude-only **32/32** under
the unchanged joint native criterion. Every control and failure is reported.

The primary passes **32/32**, including 8/8 in each schema/order stratum. The
largest whole error is 4.265%, verdict error 4.900%, and margin error .098736 nats;
the latter two are close to their unchanged5% and .10-nat boundaries. There
are no guards or excluded examples.

- Finite-MLP/cubic-query reference H:32/32, .982% median whole error and .02232 nat
  margin error. The required reference calibration passes.
- Full generator at the fixed L operating point:29/32,3.298% whole error
  and .04790 nat margin error. All three failures are descriptive/reversed
  cases:two verdict-vector misses (5.028%,5.142%) and one margin miss
  (.10669 nats). All whole-vector bounds pass.
- Primary amplitude-only:32/32,2.849% whole error,2.471% verdict error and
  .03618 nat margin error.
- Direction-only:11/32,7.565% whole error and .08881 nat margin error.
- Both correction inputs frozen:9/32,8.627% whole error and .08139 nat margin error.
- Matched-input-norm random D axis:0/32,6.180% whole error and .21500 nat margin error.
- Tangent-only L seed:0/32,33.36% whole error and .82869 nat margin error.

The public pre-native receipt is
[`fd2a8b18`](https://github.com/romrom-20/counterfactual-traces/commit/fd2a8b18f816d9d1582c6de605c4166e258c9b77).
The source freeze precedes it; all native measurements follow the receipt.
Every prediction score and every outcome is retained. This is a successful
bounded fresh-literal forecast, not a repaired268 interaction result.

The32-versus 29 count should not be inflated into a claim that native direction
changes are harmful. The three control failures sit close to declared bounds,
and this control already has a frozen approximate local operating point and
cubic query approximation. Error compensation is a possible explanation,
not a mechanism established by this comparison. The original fully finite
MLP/cubic-query H reference remains more accurate overall.

## Computation and numerical boundary

268 has 80 complete forwards,2,240 saved policy points and 67,616 passing declared
checks. All original 258 F/C/T and mapped26710/00 trajectories replay exactly.
Signed factorization, unit and crossed D input norms pass the frozen checks;
smooth RMS derivatives are independently checked with CPU64 AD/finite
differences and cubic jets. The original 1e-4 CPU64 gate/up/down bridge still
fails 396/14,784 comparisons. Every failure remains a failure. The public
full-shape Float32 node/write/head controls pass within the declared scope.

269 has 288 complete pre-native forecasts/control forwards and 32 native
measurements. Its pre-native audit passes 239,104 declared checks, maximum
absolute error 1.67064e-7, with 16,128 smooth RMS AD checks,32,256 finite differences,
16,128 independent cubic jets,8,064 receiving points and 288 exact public-head
replays. The final audit replays all32 native heads exactly and checks 8,064
source/prefix records against the measurements. The raw CPU64 projection
bridge still fails 1,247/45,696 at the original 1e-4 bound. All320 CPU64 finite-head
bridges meet .005; public matched-shape actual heads replay exactly. Raw failures
remain visible and have not been converted into passes.

Across268 and 269,400 complete model forwards ran. The posthoc directional
geometry adds zero forwards. Runtime/source/input seals and software versions
are in the evidence. Large arrays stay local with sizes/hashes and regeneration
instructions. The local replay wrapper is separately labelled numerical
replication of published literals; it does not claim another fresh test or
replace the original public pre-native guard.

One chosen fresh algorithm uses one observed False computation, an L seed and
one forecast, plus the specified finite primitive evaluations. Control arms
and duplicate numerical checks are additional work. There is no wall-clock
speedup, compact learned circuit or universal mechanism claim. Query curvature,
finite learned source/KV/MLP computation and the shared backbone remain in the
budget. The older whole-block derivative disputes are still unresolved;
matched-public arithmetic does not certify an independent real-arithmetic
reconstruction of the complete network.

The open mechanistic question is why the learned upstream computation produces
these particular amplitude and direction changes, and which remaining finite
operations can be replaced by a smaller predictive account. A prediction that
passes its specified behavior test can answer part of Question B without
explaining every contrast or identifying the native learned circuit.
