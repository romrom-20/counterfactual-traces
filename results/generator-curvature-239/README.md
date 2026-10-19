# 239: False-curvature model of the native/forecast generator contrast

The finite-normalization/quadratic-SiLU model passes the local contrast rule
at MLP17/18 in32/32 each, but fails MLP16 and misses11 MLP19 verdict cases.
The four-site account fails; the pure quadratic control fails every site.
A posthoc exact SiLU-tail identity explains the model's mathematical omission,
with MLP16 error concentrated at the first receiver. Concentration uses the
observed error and does not establish a predictive mask, circuit or final effect.

[Full report](../../docs/analysis/239-generator-curvature-result.md)
· [Frozen protocol](../../docs/experiments/239-curvature-generator-contrast.md)
· [Upstream causal prerequisite](../upstream-remainder-238/README.md)

All32 cases, all four sites, both models, both scopes and all three components
are retained. All numerical checks pass:384Hessian/FD qualifications,3328
separate allocation/contrast checks,512Float32primals and512rawFloat64bridges.
The tail description is explicitly posthoc and has no new model call or causal
intervention. No ordinary new False/True model is run by239.

Standard-library integrity verification:

```sh
python -S research/boolean_coupling/reproduce_curvature.py --verify-only --history --plan
```

Reproduce the full fixed public-input prerequisite chain and239:

```sh
python research/boolean_coupling/reproduce_curvature.py --run --out runs/my-curvature-test
```

Reuse a qualified local238 prerequisite and cached checkpoint:

```sh
python research/boolean_coupling/reproduce_curvature.py --run --offline \
  --input-238 runs/existing-238 --out runs/my-curvature-test
```

Every output must be new; unexpected failures preserve logs and stop. These
old-context diagnostics are not fresh confirmations.239 alone uses CPU64/NumPy
with cached checkpoint tensors; regenerating its earlier prerequisites needs
the [recorded MPS environment](../../research/README.md). No private data/API key
is required. Raw activations remain local/inventoried, checkpoint weights are
not duplicated, and article drafts stay offGitHub. Repository remains private.
