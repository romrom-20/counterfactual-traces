# Signed correction amplitudes and directions 268–269

[Report](../../docs/analysis/268-269-amplitudes-directions-and-prediction.md) ·
[268 protocol](../../docs/experiments/268-signed-amplitudes-and-directions.md) ·
[269 protocol](../../docs/experiments/269-sealed-amplitude-forecast.md) ·
[Reproduction](../../research/README.md#signed-amplitudes-and-directions-268269)

268's amplitude-only interaction hypothesis fails 2/4. Its old native forecast
passes 4/4, motivating a separate frozen new-literal predictive test.269's
amplitude-only primary passes 32/32, with 2.85% median whole-displacement error.
The local MLP operating point and correction directions are cached from an
extra L forecast; two signed amplitudes per receiving row/layer still adapt.
The shared backbone and substantial finite computation remain.

The failed interaction hypothesis stays failed. Forecast accuracy does not
identify a learned circuit or solve general Question B. New literals remain
dependent synthetic templates in one small checkpoint.

`amplitude-direction-268/` preserves every old interaction/native outcome and
original numerical failure. `directional-geometry-268/` is a separately labelled
posthoc, zero-forward input geometry/identity calculation. Most of the omitted
direction is outside the simple False-pattern/cached-direction plane.

[269 records](../signed-amplitude-forecast-269/) include the unchanged public
`pre-native/` receipt, all32 forecasts and 32 native observation metadata, final
outcomes and all raw failures. The seal was committed/pushed as
[`fd2a8b18`](https://github.com/romrom-20/counterfactual-traces/commit/fd2a8b18f816d9d1582c6de605c4166e258c9b77)
before any unmodified native True measurements. No private article is included.

Raw activation arrays are excluded and inventoried. Verify source/record
integrity and all frozen case decisions using only the standard library:

```sh
python -S research/boolean_coupling/verify_amplitude.py
```

CI verifies this record, not independent model replication. The original
whole-block derivative disputes stay unresolved; matched-public Float32
qualification does not recertify them. Full raw projection failures remain.
