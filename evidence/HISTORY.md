# Preserved scientific history

The active tree focuses on Question B. Earlier probe, sentiment and benchmark pipelines are retained in the historical snapshot rather than presented as current projects. Unpublished article drafts are excluded from all active branch snapshots and bundled evidence.

## Before the Question B cleanup

The pre-cleanup scientific/code tree is preserved at tag [`archive/before-question-b-cleanup`](https://github.com/romrom-20/counterfactual-traces/tree/archive/before-question-b-cleanup), with article drafts withheld. It contains the retired source, scripts, tests, configurations, results and original citation metadata. The MIT license and original copyright notice remain unchanged.

`retired-files.json` inventories the 782 legacy files removed from the active tree, with original blob IDs, SHA256 hashes and sizes. Removing them from the current reading path is not a reassessment of every old result.

## Publication-filtered evidence

[The archive notes](README.md) explain the 2,867 retained historical files and one withheld unpublished article. [publication-withdrawal.json](publication-withdrawal.json) records the declared omissions and original hashes.

The 225/225b scientific computations, protocols, model inputs, numerical bounds and audit results are unchanged. The public manifest is a separately identified publication revision: drafts and their links are withdrawn, the bundle omits the historical article, and the verifier checks all retained original hashes plus the exact declared omission. The original full release and article copies remain local. This revision does not rerun or numerically recertify an experiment.

## What GitHub checks mean

The workflow runs lint, compiles published Python sources and verifies study/archive hashes. It rejects tracked article files under `docs/drafts/`. It does not rerun the MPS experiment.

Narrow style exceptions preserve the scientific sources; correctness-oriented lint remains active. The former probe/sentiment test suite is archived. An auxiliary local run had 189 passes and 9 missing-private-fixture failures; those failures have not been relabeled as successes. Passing current checks means the retained release is intact, not that all scientific hypotheses succeeded.
