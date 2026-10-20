# 248: later context matters more than carrying the changed verdict state

From the same boundary before block 10, run the **entire finite suffix**, rather
than removing one attention write. Cross native False/True earlier context with
native False/True verdict state. Each trajectory generates all its future states.
The supplied native True boundary states are oracles.

The two crossed conditions behave very differently:

- **True earlier context, False verdict state (P):** median verdict-vector error
  4.92%, cosine .99877, whole receiving error 4.01%, margin error .204 nats.
- **False earlier context, True verdict state (R):** verdict-vector error
  101.78%, cosine -.296, whole receiving error 101.05%, margin error 10.90 nats.

The frozen verdict-state-carry primary passes **0/32**. Context-only also passes
**0/32**: its verdict vector passes in 16 cases, but its margin misses the .10-nat
bound in every case. Its small vector error is not a successful complete account.
Full receiving-plus-verdict accuracy also passes 0/32 in both conditions.

The context-only vector's median signed projection onto the final native verdict
response is .9973. Carry alone contributes -.0164, with interaction .0169 in the
specified finite allocation. The interaction norm is 4.89% of the native verdict
response in median, yet the joint negligible-interaction criterion still fails.
Small full-vector fractions need not imply sufficiently small score errors.

These observations support continued context-dependent computation after the
boundary over an account in which the changed verdict state alone carries the
response unchanged. They do **not** establish which later read is necessary,
whether one late read or several are responsible, or whether the path is direct
from the source or mediated through other token rows. Cross-position influence
in this model travels through attention; this is not a claim about a learned
semantic circuit.

Native False/True endpoints replay every suffix coordinate and both scores
exactly. Earlier rows remain identical for F/R and P/T at all 19 blocks. The
separate finite-head and saved-array audit passes **480 checks**, maximum 9.14e-6.
Budget: 32 ordinary False calls, 2,432 suffix blocks and 128 full-vocabulary heads.
No new ordinary True call does not make this False-only: True incoming states
are reused. All cases are old, dependent templates in one checkpoint.

[Protocol](../experiments/248-native-carry-read.md) ·
[All cases and audit](../../results/native-carry-read-248/README.md)

![Native context, carry and source controls](../figures/247-249/248-249-native-states.png)
