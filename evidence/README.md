# Historical evidence through 224b

`through-224b.tar.gz` is a publication-filtered copy of the earlier scientific preservation archive. It retains **2,867 of the original 2,868 files**, plus the original manifest. One unpublished article draft is withheld. Every retained file remains byte-identical to its original recorded hash; this is no longer described as the complete original closure. The original full copy remains local.

The exact omissions and original hashes are recorded in [publication-withdrawal.json](publication-withdrawal.json). The current archive checksum is in `through-224b.sha256`.

The archive includes superseded scientific hypotheses, failed numerical checks and invalid replacement-agent attempts. Preservation is not endorsement. Experiment 204 remains a qualified retrospective diagnostic, old whole-block derivative disagreements remain unresolved, 224 failed its final-coordinate bridge, and 224b tests a separately registered actual-node background.

Large raw activation arrays and checkpoint weights are excluded. The standalone 225b implementation regenerates its states from exact public inputs and the pinned public checkpoint, without this historical archive.

Verify the retained files and declared omission without extracting or loading a model:

```sh
python research/boolean_coupling/reproduce.py --verify-only --history
```

This is a preservation check, not model replication. Article drafts are kept locally and excluded from the repository and bundled archive.
