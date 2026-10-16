# Preserved history

The active tree now focuses on Question B. Earlier probe, sentiment and benchmark pipelines are retained in Git history rather than presented as current projects.

## Before this cleanup

The complete previous repository tree is preserved at commit [`01578b8eff29e8ebd4a5fb6aa1e01ff658c17258`](https://github.com/romrom-20/counterfactual-traces/tree/01578b8eff29e8ebd4a5fb6aa1e01ff658c17258), with the tag `archive/before-question-b-cleanup`. It contains the retired source, scripts, tests, configurations, result bundles and original citation metadata. The MIT license and original copyright notice remain unchanged.

`retired-files.json` inventories the files removed from the active tree, with their original Git blob IDs, SHA256 hashes and sizes. They are recoverable from that commit; removal from the current reading path is not a retraction or reassessment of every old result.

## The scientific preservation bundle

[through-224b.tar.gz](through-224b.tar.gz) preserves the earlier 2,868-file scientific closure, including failures, disqualified attempts and superseded drafts. Its [original evidence notes](README.md) explain its scope and verification. Do not treat preservation as endorsement of superseded conclusions.

The published 225b sources, protocols, metadata, logs, figures and draft remain byte-identical to their original [136-file release manifest](../results/baseline-interaction-225b/manifest.json). This cleanup does not rerun, reinterpret or numerically recertify that experiment. Large raw activation arrays remain excluded and hash-inventoried.

## What GitHub checks mean

The current workflow runs lint, compiles the published Python sources and verifies every published study hash plus all historical bundle hashes. It does not import or download the model and does not rerun the Apple Silicon/MPS experiment.

The former workflow stopped at 41 style-lint findings, including exact frozen copies of the scientific sources, before its test step. Those sealed files retain narrowly scoped exceptions for their existing import/lambda/semicolon style; correctness-oriented lint rules remain active. The old probe/sentiment test suite belongs to the archived project. Passing current checks means the focused release is internally intact, not that all scientific hypotheses succeeded.

Before retiring that suite, an auxiliary local run completed with 189 passing tests and 9 failures. All nine failures required missing `.context/exp055-private-predictions.jsonl` or `.context/exp076-private-schema-corpus.jsonl` fixtures. Those failures are not presented as resolved by this cleanup. The old suite remains in the tagged snapshot; it is not a public reproduction test for the standalone Question B experiment.
