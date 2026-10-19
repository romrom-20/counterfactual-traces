# 238: which upstream approximation errors generate the wrong state before MLP20?

Freeze before238 measurements.237 shows that exact late computation cannot
repair the earlier wrong input.235's write-error ledger did not distinguish a
new local approximation error from a write responding to inherited input error.
This retrospective diagnostic does. No new nativeTrue full forward is run.
All32 old228 contexts and all37 operations from attention2 through attention20
are retained (MLP2–19 included). The starting post-block1 receiving state is
exact. The target is the input toMLP20, before the successful native-reset budget.

## Exact error recurrence in the original affine prefix

For operationj let A_j be228's original finite-source/False-receiving-affine
rule. BeforeMLP20 every receiving operation inF is affine. Write its linear
partJ_j and original predictionp_j; native input/output aret_j,t_{j+1}.

`e_j=t_j-p_j`, `R_j=t_{j+1}-A_j(t_j)`;
`e_{j+1}=J_j e_j+R_j`.

ComputeR from the saved forecast and one qualifiedJVP at e_j. Independently
reconstructA_j(t_j) with explicit NumPy derivatives. These are oracle local
remainders evaluated at native inputs, not permissible prospective information.
They are distinct from235's write discrepancy.

Reverse transport witha_end=e_end and a_j=J_j^T a_{j+1}. Each operation's signed
contribution to squared terminal error is<a_{j+1},R_j>. Their sum must equal
||e_end||² because initial error iszero. Publish local norms, adjoint norms,
signed projections, and projected amplification; a projection is not a vector
norm or unique causal fraction. This tests the original affine transport,
not the true finite network's derivative along its native trajectory.

Rank all37 operations by median absolute per-case signed terminal projection,
chronological tie-break; selecttop4. Seal ranking, input/source hashes and all
stage-one records BEFORE any intervention outcomes. Both positive and negative
contributors remain eligible. Selection and evaluation use old observed cases;
this is not fresh confirmation or a fitted prediction claim.

## One targeted causal test, with evolving state

Compare five arms ending in finite MLP20 and full blocks21–28/readout:

- OriginalF prefix, followed by the same completely finite suffix.
- Add selected nativeR_j at those four sites inside the original affine prefix;
  propagate their combined correction by the declaredJ maps. This is an oracle
  test of transported-remainder sufficiency, not a predictor.
- Evaluate those four operations finitely at the FORECAST'S OWN evolving input;
  retain the original affine rule at other prefix operations. This tests whether
  an oracle generator diagnosis transfers into a real local repair.
- At the same four sites, use a SHA-seeded independent direction matching each
  receiving row's finite-minus-affine correction norm at the control's OWN input.
- Native input toMLP20 with the same finite suffix, a replay control.

No layer sweep, no choice based on final outcomes, no learned circuit claim.
An oracleR success with own-input failure means the native-input remainder is
not a sufficient portable explanation. Both failing rejects the tested sparse
upstream account. A successful own-input repair still needs a primitive-level
explanation, removal of oracle source information and a sealed fresh test.

Primary accuracy remains final native displacement relative<=.05/cosine>=.99
at BOTH whole receiving and verdict, margin<=.10nats, native norms>=.001,
24/32 and6/8 each schema/order stratum. Report all five raw counts. Also report
whether the oracle/own-input arm halves parent, final verdict-vector and margin
errors against base, and own-input beats the matched control on all three.
Do not turn relative improvement into native accuracy. No cases excluded.

## Qualification and storage

CPU64 only for derivatives. Each actual usedJVP gets centralFD(h1e-5)<=1e-4.
Reverse derivatives get the bilinear identity against the actual error direction
(or a deterministic sinusoid if zero) at1e-9; their pairedJVP is alsoFD checked.
Adjoint scalar closure, combined selected projection and update equations retain
1e-9. False smooth/public primals retain1e-4. NumPy explicitly reconstructs every
used derivative and native-inputR at1e-9 from the pinned public checkpoint.

Selected finite attention primals retain1e-4. Finite publicMLPs are independently
reconstructed in Float32 at1e-4, matching their executed numerical domain.
Also retain Float64 mathematical bridges at the original1e-4 bound, reporting
any failures separately; these cannot be silently reclassified. Finite heads
retain.005; original237b public suffix and native final controls retain1e-4/.005.
The oldFloat32 derivatives and237 failed bridges remain unqualified.

Use the existing cached checkpoint rather than duplicating weights. Record
config/checkpoint hashes, full False attention inputs and exact source provenance.
Store directions,tangents,centralFD vectors and hashes of both evaluatedFD
endpoints; independent NumPy reconstructs derivatives andFDs from those inputs.
All model/component/derivative work is counted. Output directories must be new;
stop and preserve partial arrays/checks on failed qualification or insufficient
disk space. Do not delete historical evidence to make room.
