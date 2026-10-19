# 237b/237c: the late attention account changes with depth; a final predictor remains missing

**Neither selective late-attention policy recovers the native response.** With
an exact native reset before MLP20 and every remaining MLP finite, restoring the
query–key product lowers median whole-receiving displacement error to **10.8%**,
but passes the joint vector/margin criterion in **0/32** cases. Restoring softmax
alone also passes0/32. Both primitives finite is the complete public attention
computation and replays the native result exactly; it is a control, not a new
explanation or predictor.

**Numerical qualification is explicitly partial.** Original237 failed its
smooth/public final bridge. 237b completed all cases, but its original auditor
stopped on one Float64 MLP bridge. 237c retains that failure while finishing all
checks. Attention derivative/graph checks pass; independent Float32 MLP checks
and public replays pass. The original failed comparisons remain failed.

## What was tested

236 localized a consequential failure to attention 21–24 after an exact native
input reset. Here the early finite normalization/value-product rule is held
fixed while a two-bit factorial restores the finite query–key product, softmax,
or both. The bit orderN,K,S,V specifies normalization, QK, softmax and the
probability/value product;1means finite,0means the False-anchored affine map.
The four policies are1001,1101,1011,1111. This is the False-anchored
normalization/product precursor, not234's source-conditioned K/V transport rule.

Each graph defines a receiving omission within the actual public finite
attention computation at the branch's current input. If C is a smooth graph,
source-only subtraction removes its shared source discrepancy:

`D_b = (C_full(y) - C_b(y)) - (C_full(source) - C_b(source))`.

The intervention installs`A_public(y)-D_b`. Receiving inputs are first installed
in Float32. This explicitly uses finite public attention at every input; it is
not an economical False-only forecast. Full1111 removes zero. Source rows use
saved native228 data, and the native-reset branch additionally uses the native
input before MLP20. Both are declared oracle information.

All four policies, the original236 False-linear rule and an unmodified public
finite control are run under both ordinary forecast and native-reset incoming
states:12arms, all 32 old dependent contexts. MLP20–24 and blocks25–28/readout
are finite. No new nativeTrue full model is run.

## Native reset: neither omitted primitive is dispensable in this policy

Median whole-displacement relative errors are **24.4%** for the old linear
rule, **22.6%** for finite normalization/value-product, **10.8%** when QK is also
finite, and **19.6%** when softmax is also finite. Their median margin errors
are **0.344**, **2.439**, **0.160**, and **2.206nats**, respectively. All four
meet the complete accuracy criterion in **0/32** cases.

The criterion requires relative error<=5% and cosine>=.99 at BOTH the full
receiving sequence and verdict, plus margin error<=.10nats. QK restoration
halves both vector and margin errors against1001 in 16/32, but its final native
vector error remains too large. Improvement and a favorable scalar margin do
not establish full-response accuracy.

Only1111 and the public control recover the native response in 32/32, exactly.
Since1111 is full attention by definition, that result does not identify a
compact mechanism or show how to predict the incoming state.

## Inaccurate incoming state: even complete late computation cannot rescue it

With the original forecast input before MLP20, all six policy/control entries in
that regime have **0/32** joint accuracy. Even unmodified public attention 21–24,
all remaining finiteMLPs and the finite suffix leave **72.3%** median whole
relative error and **5.146nats** median margin error. The old linear continuation
had74.0% and4.849nats. More exact late computation need not improve the margin
when the incoming state is wrong.

That is a separate boundary from the failed native-reset approximation:
the incorrect earlier carry persists even if all later attention is evaluated
finitely. A selective late repair alone cannot solve QuestionB.

![All cases, both incoming regimes and every policy](../figures/237b-late-attention.png)

## Fixed-input vectors show a change with depth

At1001's own incoming state at every late layer, the other three graphs are
also evaluated without changing that input. The signed QK term, softmax term
and their factorial interaction sum exactly to full-minus1001. This is distinct
from the final pipeline factorial, whose branches develop different states.

In the native-reset regime, median whole-vector signed projections onto the
local missing response are **.764 for QK and.242 for softmax at attention 21**.
At attention 23 they are **.176 for QK and.842 for softmax**. The corresponding
ordinary-forecast values are.702/.216 and.215/.840. These are descriptive
projections, not sufficiency tests or unique causal percentages. Medians of
separate components need not add toone. Their accompanying interactions,
norms, remaining relative errors and every case are retained in`local-effects.csv`.

This supports a narrower working question: why does the missing response shift
from QK interaction near 21 to softmax response near23? It does not license a
learned-feature name. A layer-specific explanation must still survive direct
interventions and improve final vector accuracy; selecting a few layers from
these projections alone would be another retrospective correction search.

The explicitly posthoc margin factorial also depends on the incoming state:
QK-only and softmax-only pipeline margin changes have opposite signs in 32/32
ordinary-forecast cases, versus15/32 with the native reset. That is an
operation-policy interaction across evolving trajectories, not a measurement
of two learned competing mechanisms.

## Complete numerical record: passes and failures

- **237 failed** on the first context's smooth/public final residual bridge:
  **0.0001678466796875 >0.0001**. No complete case or mechanism result was saved.
  Its source, protocol, runtime, weights inventory and traceback are preserved.
- **237b's runner completed32/32**, but its original independent audit stopped
  on a Float64 finite-MLP comparison:**0.00011064888120415617 >0.0001**.
  No 237b audit pass is claimed.
- **237c inspected all25,120 comparisons without changing bounds. 25,119 pass;
  that same single Float64 MLP comparison fails.** Its status is explicitly
  `complete_with_failed_checks`, never`passed`.
- All **1,536 distinct primitive analytic/AD/FD comparisons** pass
  (AD<=1e-9, centralFDh1e-5<=1e-4); all **256 legacy used JVPs** pass separate
  NumPy derivative andFD checks. The largest saved primitiveFD discrepancy is
  **2.32e-9**; legacyFD **7.48e-9**. Old public Float32 derivatives are not recertified.
- **2,304 independent NumPy policy corners**, plus1,792 full-corner comparisons,
  reconstruct the smooth graphs within **1.28e-13**. All removal/update/factorial
  identities pass1e-9. Public/smooth local attention bridges pass1e-4,
  maximum **4.88e-5**. All384 finite heads pass.005, maximum **1.00e-5**.
- All **1,536 independent Float32 MLP reconstructions** pass1e-4,
  maximum **1.91e-5**. Separately **1,920 public full-sequence MLP replays are
  bit-exact**. This is same-backend replay, not independent mathematical
  reconstruction. Neither reclassifies the failed Float64 bridge.
- Original236 branches and full native controls replay exactly. No input,
  threshold, outcome or primitive source was changed to turn a failure into
  a pass.237b's arrays are unchanged by237c and the posthoc descriptions.

The attention operation effects have strong separate numerical support. The
whole study nevertheless retains a failed raw mathematical bridge and cannot
be described as passing every preregistered qualification check. Outcome counts
above are conditional diagnostic measurements, not a fully qualified new
end-to-end mechanism or prospective success.

Actual237b work:32ordinaryFalse,0ordinaryTrue,160source attention,2048additional
public attention calls,128False graphs,512source policy corners,1792receiving
policy corners,1280additional full smooth graphs,1536primitive derivative checks
with3072FD endpoints,256legacyJVPs/512FD endpoints,1920finiteMLPs,1536suffix
blocks,384heads. The separate replay adds32False,160source attention and1920MLPs.
No speedup or learned circuit is established. Failed grounding controls, older
derivative disagreements and failed224/227 coordinate bridges remain in force.

## Reproduce and inspect

[Failed237](../../results/late-attention-237-failed/README.md)
· [All237b/237c evidence](../../results/late-attention-237b/README.md)
· [237b frozen definition](../experiments/237b-late-attention-removals.md)
· [237c numerical diagnostic](../experiments/237c-complete-numerical-diagnostic.md)
· [Runner](../../research/boolean_coupling/late237b.py)
· [Complete auditor](../../research/boolean_coupling/audit237c.py)

Standard-library integrity checks:

```sh
python -S research/boolean_coupling/reproduce_late.py --verify-only --history --plan
```

The [research guide](../../research/README.md) gives the actual public-input MPS
pipeline, preserved failures and optional cached-prerequisite path. Article
drafts and raw checkpoint/activation arrays remain excluded.
