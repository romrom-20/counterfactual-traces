# 257 — does the Boolean source step leave the RMS Taylor radius?

Retrospective, model-free follow-up to 256. Hypothesis: the native Boolean
embedding change has b>=1 at the source, despite b<1 at receiving inputs. This
would qualify a local source/receiver approximation distinction. Reject this
specific explanation if the source b is below one. Do not infer that it explains
the complete failed 242/243 forecasts, which have different evolving states.

Read source embeddings 895/830 from the pinned public checkpoint converted to
the experiment's Float32, and verify the installed public Qwen code uses those
embeddings as the first-block input. This is one unique embedding comparison,
not 32 independent facts. Later native source and receiving inputs come from
the already measured 256 trajectories. Check original source/array identities.

At every layer and source/receiver scope, record b, 1/sqrt(max b), and smooth
normalization-update Taylor error at degrees 1,2,3,4,6,8,12. Coefficients use a
generic power-series recurrence, checked against the qualified first three
coefficients within 1e-10 and its defining polynomial identity within 1e-8.
The exact False receiving-state boundary must replay within zero. Preserve
all cases/scopes/layers, including zero updates. No model forward, forecast,
behavioral intervention, feature attribution or whole-network theorem occurs.

Convergence radius is for scaling a fixed local row vector, not scaling an
input edit through the model. A local radius above one does not establish that
cubic truncation is accurate or that the remaining learned model is understood.
