# 242b: can a uniform primitive rule generate the complete counterfactual response?

Freeze protocol, code and32 fresh literal pairs before model work.241 diagnoses
the original predictor's error using oracle remainder vectors; it does not
identify the native computation. Test a uniform rule across ALL28 blocks, with
no selected layers, fitted correction vectors or native receiving inputs.

## Information and computation budget

Each pair provides one ordinary False computation, known model weights, and
the known one-token Boolean edit. Capture False primitive inputs/outputs only.
Each forecast then runs a modified computation on the edited input, recursively
using its OWN generated states, including the changed source token. No exact
native source trajectory, first-block receiving state or True-state direction
is supplied. Eight forecasts per pair are globally sealed and committed/pushed
to the private repository before ANY ordinary True measurement on these inputs.
The True run is the reference corner, not a pre-seal forecast.

This is a primitive-family sufficiency test, not computational compression or
a speedup. Dense projections, attention products and gate–up products are finite
in every arm. Public hook implementations may calculate elementary finite
outputs that are discarded on replacement; those calls remain counted work.
Even a pass would not identify learned features or, by itself, solve Question B.

## Three-factor intervention

Apply the same setting at every block, from the changed Boolean token onward;
the causal prefix stays unmodified. Final RMSNorm/head always remain finite.
Arm bits are ordered N,S,G:

- N=0: input and MLP RMSNorm use False-affine maps `z0+DN0(x-x0)`;
  N=1: public finite RMSNorm.
- S=0: attention uses a positive normalized quadratic exponential tilt of
  its public False probability vector; S=1: public finite softmax.
- G=0: SiLU uses its False quadratic Taylor map;
  G=1: public finite SiLU.

SiLU uses the public False output as its constant and CPU64 derivatives at the
actual False gate. Softmax uses p=p0/sum(p0), u=dl−E_p[dl], w=1+u+.5u², and
`p_hat = p0*w/E_p[w]`. The weights are positive and retain the public False
row mass. Common logit translations cancel. This rational map matches the
public-anchored exponential tilt through second order but retains higher-order
normalizer terms: it is NOT a pure quadratic Taylor polynomial.

An additional **100_raw** control uses `p0+p0*u+.5*p0*(u²-E_p[u²])`, with the
same local first/second derivatives. Negative weights are allowed in that raw
control and reported. Do not attribute a difference uniquely to positivity:
normalization and the retained higher-order terms change together.

**Primary hypothesis:100**, finite normalization and products, positive quadratic attention tilt
and quadratic SiLU, suffices for the native receiving residual and verdict. Forecast arms
000,001,010,011,100,101,110 and100_raw are all retained. Secondary comparisons distinguish
finite-softmax rescue (110), finite-SiLU rescue (101), and necessity of finite
normalization under finite gates (011). No winning secondary may replace a
failed primary; any new selected rule needs an additional fresh test.

## Frozen outcomes and stability handling

Same global bounds as228: native receiving displacement relative error<=.05,
cosine>=.99 at BOTH whole receiving sequence and final verdict; native norms
>=.001; final true-minus-false margin error<=.10 nats. A rule needs24/32 and6/8
per schema/order stratum. Score positions strictly AFTER the edited token;
the source row is generated internally and retained separately.

All32 pairs are measured after the seal, irrespective of forecast success.
Nonfinite values, a block receiving/source-state norm exceeding100 times the
corresponding False norm, or an approximated primitive's input displacement
infinity norm exceeding100*max(1,False-input infinity norm; UNMASKED logits for softmax), or an approximated
output infinity norm exceeding min(1024,100*max(1,False-output infinity norm)),
terminate that arm
as a prespecified UNSTABLE forecast failure. Save its first failure location,
partial trajectory and checks; continue other arms. This cap is determined
without native data. No case is dropped or assigned a finite accuracy score.
Other numerical/implementation failures stop the experiment for review.

## Qualification before native measurements

Run an instrumented000 zero-edit control for every pair: it must replay False
states/scores exactly. Every forecast block must preserve the pre-edit causal
prefix exactly. Verify model revision, input lengths and the single895→830 edit.
Check every False primitive primal against independent CPU64 arithmetic at1e-4.

For each distinct actually used approximated-primitive direction d, qualify its
False derivative actions in CPU64 using v=d/max(1,||d||inf). Explicitly retain
this scale: these are unit-direction checks, NOT a recertification of the old
public Float32 block Jacobian. Compare analytic first/second actions with
independent Torch JVP/nested-JVP actions at1e-9. For softmax these are the
DERIVATIVES OF THE PUBLIC-ANCHORED EXPONENTIAL TILT, `mass*softmax(log(p0)+d)`
at d=0; public p0 includes Float32 rounding. Separately check the original
mathematical False softmax primal at1e-4. Do not describe the1e-9 check as an
exact derivative check of the original Float32 softmax implementation. Central first FD h1e-5 and
second FD h3e-4 retain1e-4 absolute bounds on the normalized actions. Norm needs
only its first derivative; softmax and SiLU need both. Zero directions use exact
zero identities and may skip redundant FD. Cache by primitive/layer/input and
direction hash; record calls and unique qualifications separately.

Independently reconstruct each installed affine/quadratic map using Torch CPU64
autograd coefficients, scale back to the ACTUAL direction, and independently
recompute the normalized polynomial tilt using Torch CPU64 arithmetic. Cast toFloat32 and
compare with the installed values at1e-4. Save all checks, scales and direction/
FD endpoint hashes. Save False anchor arrays, prefix hashes and each forecast's
complete block-tail trajectory; the pinned weights regenerate omitted parents.
Separate final-head reconstruction retains.005. No tolerance is changed after
seeing outcomes; unexpected qualification failure is retained and stops work.

## Freshness, seals and scope

cases242.json has64 token-sequence hashes absent from the union of available
published historical catalogs and case sets. Templates remain dependent and
use the same checkpoint. This is fresh literal confirmation, not a new task
distribution or independent-team replication. Check the complete forecast seal,
pre-measurement audit, file/array hashes and private Git commit proof before the
measurement phase. Record timing and all ordinary/component work.

Keep all older qualified diagnostics, failed derivatives/bridges and failed
grounding controls unchanged. A pass supports the declared computational rule;
a failure limits this uniform low-order primitive account. Neither outcome
establishes a learned circuit, novelty priority or competent-verifier shutdown.

Original investigation requires a remote private pre-measurement seal. A later
numerical reproduction on these already observed literal pairs may use an
explicitly recorded LOCAL replication seal and cannot claim fresh confirmation.
The run mode is fixed in design.json before forecasting.

## Preserved implementation failure

Original242 stopped in the first case during qualification: public Torch SiLU
does not implement the nested forward AD used by its independent CPU64 checker.
No native True measurement ran.242b changes only that checker expression to
`x*sigmoid(x)`, retaining the protocol, fixtures, hypotheses and tolerances.
The original frozen source, partial arrays, checks and traceback are retained.
This is an implementation failure, not an empirical test of the primary rule.
