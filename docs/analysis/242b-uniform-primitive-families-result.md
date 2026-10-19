# A uniform primitive rule stays bounded but fails to predict the native response

**Question B remains open.** The primary rule passes **0/32 fresh literal pairs**.
Keeping normalization finite prevents the declared instability seen with affine
normalization, but neither a positive attention approximation nor finite
normalization/products makes quadratic SiLU sufficient across the network.

![All completed forecasts, layer profiles and final errors](../figures/242b-uniform-primitive-families.png)

## Why change the approach?

241 explains errors in a particular linear predictor through oracle remainder
vectors. A compact explanation of those errors need not explain the native
computation.242b instead freezes one computational rule at every block and
generates its own counterfactual states from the known token edit, including
the source Boolean. It receives one ordinary False computation, the weights
and input edit. No native source trajectory, first-block receiver or native
receiving direction is supplied.

This is a stronger approximation budget than the earlier tests that computed
the source trajectory finitely. Its failure does not invalidate their narrower
local successes. It is also not a speedup: dense projections and bilinear
products remain finite, and public hook outputs can be computed then replaced.

## Frozen comparison and outcomes

Arm digits are N,S,G: finite RMSNorm, finite softmax and finite SiLU respectively.
A zero means False-affine RMSNorm, positive normalized quadratic attention tilt,
or quadratic SiLU. All products and the final readout remain finite. The raw
control uses the unnormalized quadratic softmax Taylor polynomial.

Every completed arm fails the joint criterion in all 32 cases:

- **100, primary:** median whole receiving-displacement error **50.30%**,
  verdict-position error **40.43%**, margin error **1.444 nats**.
- **101, finite SiLU:** **17.38%**, **14.87%**, **1.361 nats**.
- **110, finite softmax:** **54.72%**, **48.57%**, **3.093 nats**.
- **100_raw:** **67.03%**, **56.65%**, **3.865 nats**.

Each of 000,001,010 and 011 hits a prespecified instability limit in all 32 cases.
They are retained failures, without fabricated accuracy values. Every case is
still measured. No primary or secondary meets 24/32 overall and 6/8 in each
schema/order stratum. The frozen per-case criteria require both whole and
verdict displacement errors<=5%, cosine>=.99, native norms>=.001 and margin
error<=.10 nats. The edited source row is excluded from those accuracy scores.

The positive softmax tilt and raw polynomial have the same first two local
derivatives. The positive tilt preserves the public False row mass, but retains
higher-order normalizer terms. Their difference therefore cannot be attributed
uniquely to positivity. A better residual approximation also need not produce
a proportionately better verdict margin.

## What this distinguishes

Finite normalization is sufficient for stability **among these tested arms**.
It is not sufficient for an accurate response. Instability can originate at the
edited source, so these arms do not isolate normalization at receiving rows.

Finite SiLU makes a larger improvement than finite softmax in this particular
factorial. It does not establish that softmax is unimportant. The finite-softmax
and finite-SiLU policies change each other's generated states throughout the
network. Restoring one primitive family need not monotonically improve an
otherwise inaccurate computation.

Explicitly posthoc profiles show the primary source-displacement error growing
from a median 2.09% at block 1 to 80.51% at block 28. Receiving errors grow from 2.97%
to 50.30%. With finite SiLU, the final source and receiving errors are 24.40% and
17.38%. These profiles motivate a source-versus-receiver policy intervention;
they do not causally assign the receiving error to the source.

The posthoc global policy identity is

`native - 100 = (110 - 100) + (101 - 100) + (native - 110 - 101 + 100)`.

The median interaction-vector norm is 57.1% of the primary receiving error.
This describes the interaction between two whole-network policies, not a
localized learned circuit or a unique decomposition into semantic mechanisms.
All profiles, signed vector shares, interactions and raw-negative-weight counts
are retained; they were not used to revise the frozen rule or thresholds.

## Numerical qualification and prospective order

Code and 32 pairs were frozen before model work. All 64 token-sequence hashes
were absent from the catalog of 1,408 recorded sequences. Templates remain
dependent; these are fresh literals in the same task and checkpoint.

All 256 forecasts, checks and the global seal were committed and pushed at
`dd 81a 983cd 66f 488aecfd 0a 553670071e 2a 0275a` before the 32 native True runs.
The remote seal bytes were independently retrieved through GitHub's API and
verified before measurement. The saved proof, timestamps and hashes retain that
ordering. [Complete metadata](../../results/uniform-primitive-242b/README.md).

The separate saved-array pre-audit and final audit pass their frozen checks.
There are 3,584 False-primitive primal checks,7,898 CPU64 first AD actions,
6,157 second AD actions and 28,110 finite-difference endpoints.11,706 installed
map calls use 11,482 distinct qualifications, including 3,584 exact zero directions.
32 zero-edit controls replay every False block and score exactly. All observed
pre-edit prefixes remain exact. Final finite-head reconstructions pass .005.

CPU64 derivative checks use normalized directions and 1e-9 AD/1e-4 FD bounds;
installation checks scale back to each actual direction before Float32 casting.
The softmax derivative reference is the public-anchored exponential tilt, not
the original rounded Float32 implementation. These checks do **not** resolve
older Float32 Jacobian disagreements. Original 242 stopped on an unsupported
nested SiLU forward derivative, with no native True measurement. Its source,
traceback and partial evidence remain; 242b uses the equivalent `x*sigmoid(x)`
expression in the independent checker. No tolerance changed.

Work in 242b: 32 ordinary False forwards, 32 instrumented zero controls, 256 modified
forecast forwards (128 complete, 128 terminated), and 32 post-seal native True
forwards. Capturing anchors adds 896 QK products. The qualified installation
calls and derivative work are additional computation. Original 242's partial
work is separately retained and excluded from these 242b counts.

## Boundary and next discriminating question

This rejects the declared uniform low-order account. It does not explain why
the model learns this Boolean coupling, identify learned features, establish
novelty priority or show a competent verifier being disabled. Failed grounding
controls and all older numerical failures remain unchanged.

The next useful distinction is whether finite SiLU's benefit originates in
the changed source token, the receiving rows, or their interaction. A controlled
source/receiver factorial should compute those source states from the known
edit, rather than supply observed native receivers. That would isolate a reason
for the improvement. Another selected-layer correction sweep would not.

[Frozen protocol](../experiments/242b-uniform-primitive-families.md) ·
[Source](../../research/boolean_coupling/run242b.py) ·
[Reproduction](../../research/boolean_coupling/reproduce_uniform.py)
