# 262 — which normalized-input component generates the gate curvature?

Retrospective, zero model forwards or new interventions. Use ALL B/T inputs,
four old contexts/all28 layers from bounded261d. The public global curves remain
261b's existing data; no simplified policy is freshly validated here.

For each MLP input let s²=mean(x0²)+eps, a=mean(x0*d)/s²,
b=mean(d²)/s² and rho=(1+2a+b)^(-1/2). With gamma the learned RMSNorm weight,
the smooth gate/up False pattern is h0=W*(gamma*x0/s), and v=W*(gamma*d/s).
The first-order displacement is J=v-a*h0. The complete curvature splits exactly:

`C_base=(rho-1+a)*h0` and `C_direction=(rho-1)*v`.

These are ordinary RMSNorm geometry conditional on the known incoming d, not
two learned mechanisms or semantic features. Verify their sum against every
saved smooth curvature vector1e-9, and the scalar normalization identity1e-10.
Retain the already measured public-versus-smooth difference A from261d; its
failed absolute bridges remain failed. Reconstruct original input-conditional
ranked masks from frozen258 helper, not an observed error/readout direction.

PRIMARY question: does C_base alone predict the complete smooth gate curvature
with<=25% vector error and cosine>=.95 in>=90% of eligible case/layer inputs,
separately at B/T and whole/selected scopes? Curvature norm guard1e-8. Preserve
all rows; verdict/up scopes and direction-only outcomes are secondary.

Also freeze a deliberately simpler secondary: C_energy=-(b/2)*h0, motivated by
an approximately orthogonal displacement and small-step denominator expansion.
Report the same local curvature criteria without replacing the primary.
Record all a/b/rho and normalized radial alignments, not just favorable layers.

For J+C_base, J+C_direction and J+C_energy, bound public gate/up prediction error
by (norm(C-C_candidate) +/- norm(A))/norm(delta_public), lower clipped0.
Check triangle bounds1e-9 and report <=5% robust passes, >5% robust rejects or
uncertainty with the same1e-8 public norm guard. These bounds use observed A,
not a separately predicted numerical repair or a new global forecast.

Bind all261b/261d arrays and audit hashes, checkpoint weights, source/protocol
before calculating these new components. No numerical tolerance changes or
learned-circuit/upstream-state-generation/full Question B claim follows. A local
factor account needs direct own-state primitive interventions before any final
behavioral attribution or fresh forecast claim.
