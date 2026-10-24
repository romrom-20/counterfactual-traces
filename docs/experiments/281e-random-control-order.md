# 281e: qualify exact random-control reconstruction

Original audit281 stopped on the first case's random_direction check with
2.7755575615628914e-17 error against its0.0 exact bound. The producer computes
(draw*norm(correction))/norm(draw); the auditor instead used
 draw*(norm(correction)/norm(draw)). These are algebraically equal but not the
same Float64 operation sequence. Preserve original failure-audit.json, log and
frozen audit281; it has not passed its original exact comparison.

281e changes ONLY that reconstruction to the producer's explicit operation order.
No tolerance is relaxed: the exact random-direction bound stays0, norm bound1e-10,
and every other primitive/source/head/endpoint check is unchanged. All predictions,
original design/status/source seals, masks/criteria and failed attempts remain.
Freeze this audit/protocol BEFORE rerunning, bind it to the original failed audit,
completed design/status, and label the qualified outcome281e-generator-order.
The producer remains281d. This is a numerical execution qualification within the
same investigation, not independent-team validation, a repaired old whole-block
Jacobian or a new fresh experiment. If another check fails, preserve it; do not
silently edit either frozen audit or interpret scientific outcomes before audit.
