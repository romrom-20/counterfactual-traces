# 237: does the early attention rule survive the consequential late blocks?

Frozen after qualified235/236, before237 calculations. All32 inputs/targets are
old228 contexts; this is retrospective. 236 oracle11 supplies exact native input
beforeMLP20, all MLP20–24 operations finite, and only False-linear attention21–24
remains approximate. Its final displacement/margin failure is the target.

## One controlled operation factorial

Repeat forecast and native-oracle incoming regimes from236. All MLP20–24 and
blocks25–28/readout remain finite. At each attention21–24 use a common policy
in bit order N,K,S,V: normalization, QK product, softmax, probability/value
product. One means finite and zero means the first-order map at the smooth
False anchor. Policies1001,1101,1011,1111 cross QK and softmax while holding
normalization and probability/value product finite. No new learned feature is
assumed. 1111 is the full attention computation, a replay control rather than
a compact explanation. Include the original236 False-J receiving rule and
an actual public fully finite attention continuation for both incoming regimes.
All12 arms are retained. No new correction search or fresh success claim.

For each primitive graph C, the receiving update is
`public_source_mid + C(combined_input) - C(source_only_input)`.
Both C evaluations use the same smooth False anchor, causal mask and RoPE;
the combined input installs the arm's current receiving state and the fixed
known source row. This retains the declared finite source primal in all arms
and changes only the conditional receiving response. Source rows are saved
native data from228, an explicitly acknowledged oracle budget. Native-reset
inputs are also oracle diagnostics, never prospective inputs.

At1001's own input at every late layer also evaluate all four policies without
changing the incoming state. Record signed whole-receiving and verdict vectors:
QK=1101−1001, S=1011−1001, I=1111−1101−1011+1001. Their sum is1111−1001.
These fixed-input vectors distinguish the local graph discrepancy from effects
caused by subsequent different trajectories. The uniform policy interventions
provide the final-response causal test; fixed-input projections alone cannot
establish that final errors are repaired. Pipeline factorials have the same
algebra but include evolving-state effects; label them separately.

## Frozen accuracy and accounting

Use236's joint accuracy: final native displacement relative error<=.05 and
cosine>=.99 BOTH whole receiving and verdict, margin error<=.10nats, native
norms>=.001. Require24/32 and6/8 in every schema/order stratum for a policy to
account for the old failure. Report raw accuracy for all12 arms. Report the
stronger contrast with1001: eligible1001 has vector error>=.001 and margin
error>=.10; a repair halves final verdict-vector and margin errors. Do not
replace failed native accuracy with relative improvement. Full finite1111 is
not counted as a selective mechanism, regardless of passing.

For local and pipeline signed effects publish norms, projection on the full
finite-minus1001 discrepancy, and remaining relative vector error. No learned
circuit, universal explanation, or Question-B solution follows from a pass.
If only1111 works, neither omitted primitive is dispensable under this policy.
If oracle succeeds but forecast fails, upstream carry remains a separate gap.

## Numerical qualification, without reviving old derivatives

Primitive graphs are CPU64, anchored at a separately evaluated smooth False
state. Each distinct affine QK/softmax direction gets analytic/AD agreement at
1e-9 and central FD(h1e-5) at1e-4; retain witnesses. Every legacy receiving JVP
gets actual-direction FD at1e-4. No public Float32 whole-block AD is used.
Independent NumPy reconstructs all used source/receiving primitive graphs at
1e-9, all finite MLP20–23 installations at1e-4, every update equation at1e-9,
all saved FD witnesses at1e-4 and all final norm/heads at.005.

Smooth full-source and each1111 receiving output must agree with public
attention within1e-4, including input cast. Full1111 trajectories must agree
with their public finite continuations at final residual1e-4 and scores.005;
public native-oracle must replay228 exactly within those same bounds. Legacy
branches must replay236's11 branches at1e-4/.005. Stop and retain a failure;
never silently change bounds, substitute a native intermediate, or treat a
failed bridge as a mechanism result.

Expected:32 ordinary False,0 ordinary True,160 source attention,128 smooth
False graphs,512 source corner graphs,1792 receiving corner graphs,
256 legacy JVPs/512 FD endpoints,512 additional public attention calls,
1920 finite MLPs,1536 suffix blocks and384 heads. Primitive derivative counts
are direction/cache dependent and published explicitly with their endpoints.
