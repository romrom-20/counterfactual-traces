# The source is inside the Taylor radius, but low orders are still inaccurate

**The outside-radius hypothesis is rejected.** The native Boolean embedding
change has b=.879887, giving a smooth RMSNorm series radius **1.06607**. The
unit local step is inside it. The source is nevertheless much closer to the
boundary than the measured receiver inputs, whose minimum radius is **2.12861**.

This is a retrospective, model-free diagnostic on the already measured 256
states and pinned public embedding weights. The initial source input is one
unique embedding comparison, repeated by the templates; it is not 32 independent
source discoveries. Later states are native inputs, not prospective forecasts.

At the initial source input, relative error in the **normalization update** is:

- Linear: **41.51%**.
- Quadratic: **42.88%**.
- Cubic: **31.77%**.
- Degree 12: **8.54%**.

Across eligible native source case/layer inputs, cubic has **8.61% median local
normalization-update error**. Across receiving inputs it has **.586%**. These
are different local populations, not final residual or margin errors. Every
case/layer and degree 1/2/3/4/6/8/12 is retained, including zero updates.

For a fixed row displacement, the smooth normalized state is proportional to
`(x0+t*d)*(1+2*a*t+b*t²)^(-1/2)`. With positive eps, Cauchy–Schwarz gives
a²<b for nonzero d, so the complex root modulus is `1/sqrt(b)`. All measured
source and receiver inputs satisfy b<1. Convergence does not imply a low-order
truncation is accurate, monotone in degree, or sufficient through the network.
Here t scales a fixed local vector, not a Boolean edit propagated through the
model. The public Qwen implementation is checked to use token embeddings as
the first-block input; saved later inputs and checkpoint hashes are validated.
All **29,536** declared coefficient, geometric and state-boundary checks pass;
maximum discrepancy **1.777e-15**, within their separately specified bounds.

The result constrains a tempting extension of 256: its successful receiver
cubic rule cannot simply be assumed accurate at the source. It does **not**
show that normalization caused the older 242/243 forecast failures, which use
different approximate states and primitive policies. No source intervention or
new native behavioral forecast tests that causal claim here. The particular
learned source/value/MLP displacement remains unresolved.

[Protocol](../experiments/257-source-receiver-radius.md)
· [Every outcome, check and frozen source](../../results/source-receiver-radius-257/README.md)
· [Diagnostic](../../research/boolean_coupling/source_radius257.py)
