# MLP24 generates a consequential downstream interaction

225b isolated the local product interaction that reverses the tested SiLU effect at MLP23, but its quantitative account failed in all 16 descriptive contexts. 226b asks where the remaining baseline dependence is generated. It localizes a consequential new mixed response to **MLP24**, the first downstream MLP. All 16 descriptive cases pass the frozen joint margin/vector reduction rule, including 8/8 in each field order. No other individual tested sublayer passes that joint rule in these cases.

This is retrospective, on the 32 old dependent 225b templates and one pinned Qwen checkpoint. It concerns the magnitude interaction between two defined MLP23 interventions. It does not explain the entire endogenous Boolean response, establish a learned circuit, or supply a False-only predictor. Generic cases have small original magnitude interactions and are ineligible under the .10-nat guard; their zero action counts must not be read as failures to explain a large effect.

## What was separated

The four trajectories begin with products p, p+A+K, p+B, p+B+A+K at MLP23. At any boundary their states are x0,xD,xB,xBoth. The margin contrast W=mBoth-mB-mD+m0 is exactly 225b's E11-E01 gap.

Final RMSNorm/readout is evaluated finitely. At the final additive state z=xB+xD-x0, W splits into a readout-only mixed effect and H(xBoth)-H(z), the effect of the interaction already present in the final residual. Descriptive medians are:

* Actual W: **+.246279nats**.
* Readout-only mixing: **+.053938nats**.
* Residual-interaction effect: **+.199583nats**.

These are separate medians and need not sum. The readout-only account misses by more than .10nats in 16/16 descriptive cases; it is within .10 in all 16 generic cases. This rejects final-normalization-only as a complete account of the descriptive gap under this decomposition. It does not make normalization geometry a new mechanism.

For each attention-plus-residual and MLP-plus-residual update G in blocks24–28, run G at the additive incoming state. The exact finite decomposition is:

    R = G(xB+xD-x0)-G(xB)-G(xD)+G(x0)
    P = G(xBoth)-G(xB+xD-x0)
    I_out = R+P.

R is generated at an additive incoming state; P transmits the incoming interaction finitely. For MLPs, the parents are their actual post-attention residuals. No whole-block Jacobian or derivative approximation appears. Norm growth alone is not used to distinguish generation from transport.

## The causal result

At each sublayer separately, remove R from the actual Both output on receiving rows and continue every remaining operation finitely. Test a norm-matched random residual edit and an exact remove/reinsert control.

MLP24 removal changes descriptive median W from **+.246279 to+.065258nats**, versus **+.255133** for random. The median paired absolute-W ratio is **.26614**. The median final interaction-vector norm ratio is **.63671**. Every descriptive case halves the margin gap, reduces the final interaction norm by at least25%, and beats random on both metrics. These are action criteria, not significance tests or unique causal fractions.

At MLP24, median local mixed-vector norm is3.1587, versus .6319 for the finite transmitted incoming interaction. Their signed vectors are retained. A larger local norm is not itself the causal evidence; the direct removal supplies that evidence.

There is also compensation: removing the corresponding local interaction at MLP27 or MLP28 **increases** the absolute margin gap in 16/16 descriptive cases. Median W after those removals is+.26858/+.32779nats. Later mixed responses therefore do not all push the discrepancy in the same direction. This rules out interpreting every local mixed term as an interchangeable harmful error. It does not prove that MLP24 is the only origin or that a joint account of later operations is unnecessary.

## Numerical scope

All 640 reference block replays, all source/prefix checks, and all remove/reinsert endpoints/scores match exactly. The separate NumPy path validates 12,032 recorded checks, including 6,688 execution checks; these overlap and are not independent replications. Maximum finite-head discrepancy is7.816e-6 against .005. Signed finite decompositions close exactly in the saved CPU64 arithmetic at the 1e-9 bound. Sources and all prior input hashes match after execution.

Work:640 reference blocks,160 additive attention-parent blocks,160 additive MLP calls,480 post-attention intervention MLP calls,1,920 continuation blocks,1,120 finite heads. No ordinary full-model run or new True observation was needed. No derivative is used or recertified. Qualified204, invalid replacement-agent claims, original whole-block disagreements and the failed 224 bridge retain their prior status.

226 first stopped before any intervention because of a prefix-boundary assembly bug: native_tail starts after 22, so after 23 requires[1:], not[2:]. The first replay error was2.158737 against 1e-4. Its source, seal and traceback are preserved. 226b changes that index and artifact names, retaining the design, seeds and bounds.

The next narrower question is which operation within MLP24 creates its mixed product response: input normalization, SiLU mixing, or gate/up multiplication. A finite primitive decomposition and direct product interventions can test that without treating a coordinate projection as a complete explanation.

See the [protocol](../experiments/226b-finite-tail-interaction.md), [runner](../../research/boolean_coupling/tail226b.py), [NumPy auditor](../../research/boolean_coupling/audit226b.py), and [reproduction guide](../../results/finite-tail-interaction-226b/README.md). The original study ran without browser or publication tools. This repository publishes its technical evidence; article drafts remain excluded.
