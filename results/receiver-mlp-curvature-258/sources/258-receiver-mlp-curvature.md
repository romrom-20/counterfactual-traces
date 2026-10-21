# 258 — can a fixed 1% finite-SiLU budget predict the native response?

Hold the qualified 256 cubic-query rule fixed. Experiment240 found a target-blind
1% SiLU-tail mask at one MLP, but the incoming forecast state was inaccurate and
that conditional repair did not predict the native response. Test a uniform
1% rule at EVERY receiving MLP from its own changing state, on fresh literals.
No selected sites, fitted gains, native states or prior correction trajectory.

An unexecuted/unfrozen hinge-versus-smooth draft was archived locally before
model work: its forced cancellation near zero makes it a weak mechanistic test.
The actual258 rule below is frozen before any model forward.

## Fixed rule

Cache the public False gate g0 and activation phi0. At the current forecast gate
 g1/up value, set d=g1-g0. Let s=sigmoid(g0),
`f1=s*(1+g0*(1-s))`, `f2=s*(1-s)*(2+g0*(1-2*s))`.
Use `Q=phi0+f1*d+.5*f2*d²` unless a coordinate is selected for finite SiLU.

For interval[g0,g1], let v be the maximum sigmoid(g)*(1-sigmoid(g)) on that
interval, attained at its point closest to zero. The smooth quadratic remainder
bound is `B=v*(3+max(abs(g0),abs(g1)))*abs(d)^3/6`.
Rank `B*abs(up_current)*norm(down_column)` over all receiving neuron-position
coordinates in EACH layer. Restore exactly max(1,floor(.01*N)) coordinates,
with stable argsort ties, using only False/current forecast/weights. No unknown
True gate, readout, actual remainder or score is used to select. Every layer
gets at most1% for these dimensions; no larger-budget secondary is promoted.

## Arms and frozen criteria

All arms retain cubic receiver queries, finite source/prefix, receiver K/V,
softmax, MLP RMSNorm, gate/up/down projections, products and final head.
Only receiving rows strictly AFTER the changed Boolean have SiLU approximated.

- **S, primary:** quadratic except exact public SiLU on ranked1% coordinates.
- **Q:** quadratic on all receiving coordinates.
- **R:** quadratic with exact SiLU on seeded random coordinates matching the
  ranked mask's count in EACH row at R's OWN current inputs. Seed SHA256 of
  casekey:layer:row:receiver-silu-control. No manifold/learned-feature claim.
- **C:** public finite SiLU everywhere, the cubic-query reference.
- F ordinary False; Z zero-edit S must replay all tails/scores exactly.

32 fresh literal pairs,64 full token hashes absent from every existing catalog
and case fixture; same dependent templates/one pinned checkpoint. Global
forecasts+separate pre-audit+public remote seal BEFORE native T/current-QK O.
Source trajectories must match T exactly for C/Q/S/R; only C's first-block FULL
receiving tail is required native-exact. MLP approximations can differ there.

Unchanged joint criterion: BOTH whole-receiving and verdict displacement error
<=.05, cosine>=.99, native norms>=.001, finite margin error<=.10 nats. Primary
requires24/32 overall and6/8 each schema/order stratum. Report identical C/Q/R
rules; never replace S by a secondary. C must meet the same overall rule for
an accurate-query-reference interpretation. All cases remain in denominators.
Nonfinite or >100-times-False tail norm stops/preserves evidence.

## Local signed vectors and intervention

At every C/Q/S/R MLP input, compute finite-minus-quadratic write G at that SAME
input. Project the masked activation-tail product through the actual learned
 down matrix to get restored vector V and unselected vector U. Save G,V,U for
all receiving rows/layers; verify V+U closes G within1e-4. For R, V uses its
random mask. Record full vector norms, cosines and signed projections onto G
at whole and verdict scopes. Relative local quantities use norm(G)>=1e-8
eligibility; zero cases remain. Fractions/denominators and per-row mask counts
are retained, not just a favorable readout projection.

S/R actually install their predicted product and continue the network. Final
native residual/margin criteria test compositional prediction, beyond local
same-input tail capture. Different arms evolve different states, so local
vector shares are not unique global causal percentages or learned features.

## Qualification

Every live modified layer checks: False analytic first/second SiLU coefficients;
CPU64 quadratic remainder bound excess<=1e-8; public finite/anchor reconstruction
within1e-4; first FD along max-absolute-normalized direction (step1e-5,bound1e-5)
and second coefficient FD (step1e-3,bound1e-5); exact mask budget/count/control,
installed product and unchanged source/prefix; signed down-vector closure1e-4.

Separate saved witnesses: first case per schema/order at blocks1/14/28, all8
arms. Reconstruct selection from public down-column norms, False/current gates
and up, CPU64 AD coefficients1e-9/FD1e-5, products and learned down-vector
projections1e-4. False anchors retained at witnesses; every live layer checked
but not all neuron arrays archived. No full independent forward or original
Float32/whole-block derivative recertification.

Actual installed query-input SHA256 fingerprints are retained at EVERY layer,
with raw arrays at fixed witnesses. Separate reconstruction must match those
fingerprints exactly. Cubic-query coefficients checked by generic jets1e-10; normalized-direction
RMS FD as256 (first steps1e-3/1e-4,bounds1e-3/1e-5; second1e-3,bound1e-5).
Composed query/attention P/Y1e-4; finite head .005. Original raw projection,
rotary and QK coordinate bounds1e-4 retained with failures separately reported.

Budget:192 forecast forwards F/Z/C/Q/S/R,64 native/control T/O;4,480 each linear,
quadratic/cubic query maps and SiLU selection/decomposition calls;8,960 SiLU FD
checks;896 False MLP anchors;14,336 extra receiving down projections;1,792 extra
QK/softmax/PV each Z/O;96 witness QK. Audits/weight-column norms additional.
All public SiLU outputs execute before hook replacement; this is a finite-node
information budget, NOT actual compute savings or a compact complete MLP.
Learned source/K/V/MLP projections still finite; no circuit/novelty/generalization
claim or full solution to Question B follows automatically from success.
