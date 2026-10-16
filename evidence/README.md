# Historical evidence through 224b

`through-224b.tar.gz` contains the exact 2,868-file preservation closure plus its manifest. It includes scientific code, protocols, reports, numerical audit outputs, original failure logs, presentation sources and recorded run metadata. Its SHA256 is in `through-224b.sha256`.

The archive deliberately contains superseded drafts, failed numerical checks and explicitly invalid replacement-agent attempts. Their presence is preservation, not endorsement. In particular,204 is a qualified retrospective diagnostic; old whole-block derivative disagreements remain unresolved;224 failed its final-coordinate bridge;224b uses a separately registered actual-node background. Read the current result/post for the supported claims.

Large raw activation arrays and checkpoint weights are excluded. Original raw files remain hash-inventoried; the archive does not make those arrays publicly downloadable. The new225b implementation regenerates its native states from exact published prompts and the pinned public checkpoint, so it does not require those historical arrays.

Verify all preserved file hashes without extracting or loading a model:

```sh
python research/boolean_coupling/reproduce.py --verify-only --history
```

To inspect the history manually:

```sh
mkdir -p .context/history
tar -xzf evidence/through-224b.tar.gz -C .context/history
```

The extracted root is `.context/history/through-224b/`. Its original README and `.context/` records remain byte-identical inside that root. The main repository's new landing page is presentation for this release; it does not alter the historical snapshot. Verification is a preservation check, not an independently executed model replication.
