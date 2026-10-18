# 226b: generation versus transmission of the downstream baseline interaction

This is a retrospective diagnostic on all 32 old 225b cases, at MLP23 only. It does not make a False-only forecast, identify a learned circuit, or rehabilitate earlier whole-block derivatives. Existing scientific files remain unchanged. No publication or push is part of this run.

## Motivation and disclosure

225b established that K controls the tested sign reversal, while B still changes its magnitude. A preliminary saved-array calculation on 18 October found that final RMSNorm/readout alone leaves more than .10 nats unexplained in all 16 descriptive cases. This observation motivates the following test; it is not a preregistered discovery. All model interventions below are specified before their execution.

Use the four full-sequence trajectories for products p, p+A+K, p+B, p+B+A+K. Denote their receiving residuals at any boundary x00, x01, x10, x11. Their conditional margin interaction is W=m11-m10-m01+m00, exactly the quantitative gap E11-E01 from 225b. Prefix and Boolean-source states are identical and protected.

## Finite decomposition, without block derivatives

At the final residual, define z=x10+x01-x00. Split W exactly into:

    readout mixing = H(z)-H(x10)-H(x01)+H(x00)
    residual-interaction effect = H(x11)-H(z).

H is the finite learned final RMSNorm and two-logit head, not a linear projection. This order-specific decomposition is an explicit accounting convention, not a unique causal allocation.

For each of blocks 24 through 28, separately test its attention-plus-residual update and MLP-plus-residual update. Let G be that finite sublayer and y_ab its four observed outputs. Run G at the additive incoming state z=x10+x01-x00. Then define in CPU64:

    R = G(z)-y10-y01+y00
    P = y11-G(z)
    I_out = R+P = y11-y10-y01+y00.

R is the mixing generated at an additive incoming state; P is the finite effect of the existing incoming interaction. For the MLP, x_ab are the actual post-attention residuals. No common Jacobian or derivative identity is assumed. An increment in interaction norm by itself is not called new nonlinear generation. Addition/casting arithmetic is retained, and all actual signed vectors are saved.

## Causal test

At each of the ten sublayers, start from the actual 11 trajectory and install y11-R at its output on receiving positions only. Continue the remaining MLP (if attention was edited), all later blocks and the final head finitely. The other three factorial paths stay fixed. Run a residual-norm-matched, SHA-seeded Gaussian edit and an exact remove/reinsert control at each site. Every site and case is retained; no adaptive correction sweep or output-optimized direction is allowed.

Record W after deletion, its absolute change, the complete final interaction vector, and its norm relative to the original I_final. A joint reduction requires: original |W|>.10 nats; deletion |W| <= .5*|W|; interaction-vector norm <= .75 of original; and both errors below the matched random arm. Report all 32 cases, eligible denominators, schema/order strata, signed effects, and failures. A stable descriptive-site account requires at least 12/16 descriptive passes and at least 6/8 in each descriptive order. These are action criteria, not statistical significance or evidence of a learned feature. They do not license a fresh predictor automatically.

## Qualification and preservation

Seal code, protocol, 225b input arrays/records, runtime and model revision before interventions. Replay all four complete finite suffixes against their saved receiving trajectories with the unchanged 1e-4 coordinate / .005 two-score tolerances. Independently reconstruct the head in NumPy64; close the signed decomposition at 1e-9. Protect source/prefix exactly at installation and check finite continuation against native prefix at 1e-4. The original unqualified whole-block coordinate Jacobians are unused. No new derivative claim is made.

Expected work: 640 reference block replays; 160 additive attention-parent whole blocks; 160 additive MLP calls; 480 post-attention intervention MLP calls; 1,920 continuation blocks; 1,120 finite two-logit heads. These units are separate. No ordinary full-model observation or new native True example is required. Source hashes must match at the end. A separate NumPy auditor checks every case and all controls. Failure stops the run and is preserved rather than widening a bound.

The 225b saved arrays are required for this retrospective extension. The existing standalone 225b runner regenerates them; this experiment does not claim a new self-contained release or fresh confirmation.

## 226 preflight correction

226 stopped on its first reference block before any intervention: maximum coordinate replay discrepancy 2.158737421, against 1e-4. The runner assembled prefix states from the wrong boundary: saved native_tail begins after block22, so the after23 suffix must be native_tail[1:], not [2:]. 226b corrects only this slice and run/artifact/source names. The random seed still ends in 226. All tested contexts, intervention recipes, controls, criteria and tolerances are unchanged. The failed 226 code, seal, copied sources and traceback are preserved. This is an implementation correction, not relaxation of a scientific bound.
