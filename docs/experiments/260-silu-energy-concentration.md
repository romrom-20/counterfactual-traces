# 260 — is local sparse capture concentration or cancellation?

Retrospective diagnostic after258/259, frozen before calculating these energy
statistics. Use all24 C/S forecast MLP witnesses from258: four contexts, blocks
1/14/28, two arms. No new model forwards, native measurements or interventions.
C/S first-block inputs coincide; this is not24 independent observations.

For each receiving neuron/position define its omitted write contribution
`c_j = Float32((finite_SiLU - quadratic_SiLU) * up)_j * down_column_j`.
Use the existing qualified component convention, original bound1e-4 and the
original ranked1% mask. Neither the mask nor its budget changes.

Report the individual-term energy `E=sum_j ||c_j||²`, selected/unselected
energies and their fractions. Separately report norms of the SUMMED write
vectors G/V/U, coherence `||sum c_j||²/E`, cross energy `2<V,U>`, remaining
relative vector error and signed share. Positions are orthogonal coordinates
in the whole-receiving vector. Include whole and verdict scopes, norm guard
1e-8 and energy guard1e-20; retain ineligible entries.

Frozen question: does the ranked1% mask contain>=90% of the individual energy
in at least18/24 whole-input witnesses? Count ineligible witnesses as not
passing. Report the same verdict-row count without promoting it to the primary.
Success would support local concentration before cross-neuron cancellation;
failure would reject this quantitative concentration account, not negate258's
actual native forecast. Report cross energy regardless of its sign.

Reconstruct original mask exactly, bridge complete/selected writes to258 saved
vectors within1e-4, check complete vector closure1e-10 and the energy identity
within1e-8. Bind original input/witness/weight hashes, source/protocol and all
CSV outcomes. Do not infer learned features, explain upstream gate generation,
establish a cheaper predictor, or call Question B solved from this diagnostic.
