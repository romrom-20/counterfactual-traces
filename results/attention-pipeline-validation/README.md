# Local end-to-end pipeline validation

The published reproduction wrapper completed both model experiments and their audits again, using explicit local seals. Every saved numeric array matches the original runs exactly. This is same-team local repeatability, not independent replication. Raw-logit absolute failures remain failed.

The posthoc first-block check also verifies that every252 A/Q/K forecast starts with the exact native True first-block tail, before its later feedback policy diverges. No model calls were used to compare the saved arrays. See `validation.json` for all case hashes/checks and `pipeline.json` for the executed commands.
