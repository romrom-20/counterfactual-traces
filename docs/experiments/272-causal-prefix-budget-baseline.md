# 272 — A causal-prefix baseline for the Question B computation budget

Freeze helper, runner, auditor and this protocol before new272 study tensors or
forwards. All32 OLD269 literal pairs, one pinned Qwen checkpoint and dependent
synthetic templates. This is a retrospective architectural/cost baseline;
no fresh confirmation, novelty priority, learned circuit or complete Question B
claim. Preserve every prior result, original numerical failure and source seal.

## Why run this rather than another correction factorial?

Question B's native prediction target has repeatedly given way to contrasts
between approximate intervention policies. Those contrasts can diagnose an
approximation without explaining the native learned computation.258 and269
already pass bounded native prediction criteria under substantial finite work.
Neither establishes a smaller useful budget against the natural causal cache
baseline. The edit in269 occurs only5–6 tokens before the end of169–177-token
inputs. That is an important property of this cohort, not a general input edit.

At every causal block, positions strictly before the edit are unchanged.
Their cached keys and values can be reused. With the same total length and
absolute positions, recomputing the edited token and every subsequent token
executes the entire affected native subgraph. This is standard causal reuse,
not a new learned explanation. It addresses the predictive/budget interpretation
of Question B while leaving learned feature generation unresolved.

## Rule and controls

F: one full ordinary False computation with use_cache=True; record its residual
outputs and KV cache. That single computation supplies P's observed state.

P: copy ONLY cache positions[0,k), where k is the changed token. Discard all
cached positions at/after k; supply known True IDs[k:N], full attention-mask
lengthN, global position IDs/cache positions[k:N], and the pinned weights.
Execute all affected operations finitely. No native True states/controllers or
L forecast are provided. Prefix cloning is included in marginal timing.

Z: same cropped prefix, replay False IDs[k:N]. Must reproduce False up to raw
Float32 diagnostics. S: a deliberately stale-edit control, keep False cache
through k and run unchanged IDs[k+1:N]. It should replay False and fail native
prediction; it is not a valid edited model. This checks why cached edited states
cannot be reused. T: ordinary full True forward with use_cache=False, after P.
Already-observed269 F/T endpoints and scores must replay EXACTLY in full-shape
F/T, binding the reference. Cases are OLD, so this ordering is not a blind seal.

Run F,Z,S,P,T per case:64 profiled full forwards and96 profiled suffix calls.
Then3 paired timing repetitions of P/T, alternating order by fixed case index
and repetition:96 more full and96 more suffix calls. No hooks/captures in timed
calls; same loaded model, GPU synchronization before/after, input tensors on
GPU already. Include cache cloning inside P timing. Shared observed F/model
loading/tokenization are excluded from BOTH marginal timings. No performance
claim is inferred from operation counts alone.

## Frozen criteria and recording

Primary: P passes32/32 under the ORIGINAL native joint rule: whole receiving AND
verdict targetnorm>=.001, relative error<=5%, cosine>=.99, actual finite-head
margin error<=.10nat. Additionally, instrumented learned dense MACs must be<=5%
of full T in EVERY case, including the one-token full-vocabulary head. Both
are required. P still recomputes ALL affected nonlinearities and projections.

Independently reconstruct dense counts from pinned config:
per token per layer2H²+2H*Kwidth+3H*intermediate, plus H*vocabulary for the last
head. Verify all measured linear calls/shapes and attention Q/K lengths. Record
dense QK/PV MACs, pairs, rows, prefix bytes and call counts. Cache reduces rows,
not the number of layers or learned matrix calls. Report head/memory/copy costs
and that edited-late structure is special. An earlier edit can erase this saving.

Secondary latency claim requires median per-case speedup>=1.25 and at least
24/32 cases faster. Compute each ratio from its three paired medians, retain
all96 raw timing pairs. If it fails, report the failure and measured latency;
do not relabel theoretical MAC savings as wall-clock speedup.

Require exact native F/T unchanged-prefix residual hashes and KV hashes at all
28 blocks. Copied-cache lengths/positions/provenance must be exact and the full
False cache unchanged before/after all calls. P/Z reused prefix KV must match
F exactly. Save profiles with every layer's query/key lengths and projection
input shapes. Full F/T binds original269 arrays/metadata by hash.

Preserve changed-shape Float32 raw diagnostics: P versus T, Z/S versus F at all
28 suffix residuals, including source, with ORIGINAL1e-4 coordinate tolerance.
Replay actual final RMS/head at each arm's EXECUTED sequence length; record
exact0 comparisons and the original.005 head bridge separately. Record timed
P/T score agreement at1e-4. These raw diagnostics are separate from the primary
native rule; failures remain failures and qualify any real-arithmetic-exact
claim. No old derivative is used or recertified, and no tolerance is enlarged.

Bind sources, installed Qwen/cache/mask implementations, checkpoint/runtime,
all fixtures and original269 records. Save all five28-block suffix trajectories,
scores, prefix/profile hashes and timing records. KV tensors remain unsaved but
regenerate from F; publish their sizes/hashes. New output directory only,
raw case cap220MiB,512MiB free reserve. Preserve failures; no deletion of prior
arrays. Every context stays in the denominator; a stopped run is not a pass.

The historical completion/claim audit and all report/packaging/plotting are
posthoc, separately labelled. Old invalid replacement-agent claims remain
invalid;204 stays qualified retrospective; earlier raw derivative failures
remain open. A successful272 baseline would motivate demanding more from a
mechanistic predictor, not claiming that cache reuse solves learned causation.
