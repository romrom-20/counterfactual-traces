# 283: predict the missing-product response at the carried operating point

Freeze code and protocol before checkpoint calculations.32 OLD282 contexts,
16 dependent paired backgrounds; this is a retrospective mechanism test.282's
carry-only primary remains failed. No other arm rescues the new P primary.

## Fixed update and precise derivative

F nativeFalse; B sourceMLP1+2 False-cubic; K adds the missing product R(C) at the
carried input xC=Float32(xF+c). All26 later source/all receiving operations and
finite readout remain. Each new intervention uses B's SAME current MLP2 midpoint,
gate/up values; only its product or matched output write changes.

The attention step v=actualBmid-xC is known from B's own False-derived computation.
One additional finiteCcorner supplies public Float32 norm/gate/up/SiLU/cubic
values, before any full-current source2 activation or nativeTrue run. Let
eC=phiC-cubicC in Float64, and P3 be the False-coefficient smooth cubic. Calculate
the Float64 RMS derivative at xC, including epsilon and learned norm weight:

`dz = weight*(v/s - xC*mean(xC*v)/s**3)`.

`dg=Wgate*dz; du=Wup*dz; ePrime=SiLU'(gC)-P3'(gC)`.

`gateDelta=ePrime*dg*uC; upDelta=eC*du`.

PPRIMARY adds `R(C)+gateDelta+upDelta` to B's product. H adds R(C)+gateDelta;
J adds R(C)+upDelta. K adds only R(C). G0–G2 retain K's product and add seeded
random output writes normmatched to P-K's actual local write increment. FZ is
False zero. No correction iterations, alternate orders or tuned step sizes.

This is a derivative of a DEFINED ANCHORED SMOOTH SURROGATE, not a Jacobian of
rounded public Float32 execution or the whole block. Explicitly define
z(x)=RMS64(x), g(x)=gC+Wgate*(z(x)-z(xC)), u(x)=uC+Wup*(z(x)-z(xC)),
e(x)=eC+[SiLU64(g(x))-SiLU64(gC)]-[P3(g(x))-P3(gC)]. The differentiated product
is e(x)*u(x). R(C) itself retains the observed finite-minus-cubic Float32 products;
do not replace its rounding remainder with eC*uC. The actual candidate product
is Float32(Bproduct+definedCorrection).

Only AFTER K/P/H/J/G/FZ, run C1 (source2finite/source1cubic) and obtain full R(U),
then U adds R(U), N adds R(U)-gateDelta-upDelta, TnativeTrue last. U=C1 must be
exact. F/B/K/C1/T replay282 exactly. N is a full-calibration removal diagnostic,
not a forecast.14fullforwards/case=448;32extra finiteCmaps/64localdownmaps plus
32normJVPs/64CPU64learned gate/up matrix-vector products. Audits are extra.

## Frozen primary and retained native target

P must jointly satisfy:
- Original native receiving(exclude source)+verdict norms>=.001, relative<=.05,
  cosine>=.99 and actual finite margin error<=.10nat.
- Full C1-B local source2 write norm>=.001, relative<=.25, cosine>=.95.
- Actual C1-B receiving/verdict effects: same .001/.25/.95, abs margin>=.02nat,
  margin error<=.005+.25abs(target).
- Paired descriptive-minus-generic C1-B local write contrast .001/.25/.95,
  paired exactly by assignment/block/reverse/target;16 dependent pairs.
- NEW, ADDITIONAL incremental C1-K local write target .001/.25/.95, predicted
  by P-K. This is R(U)-R(C), including the small standalone attention remainder;
  it is different from282's mixed-only X. It does not replace282's failed target.

Require>=24/32 and>=6/8 in every schema/order stratum. Ineligible cases stay
failures in32. H/J/random/control success cannot rescue P. Every arm gets every
endpoint. Separately test actual incremental C1-K receiving/verdict/margin and
full-removal U-N versus U-K at the unchanged .02nat materiality rule; these can
be ineligible and do not become primary by dropping the margin guard. Native
K already passes in282; no universal native necessity from added precision.

Report signed LOCALWRITE/RECEIVING/VERDICT effects from H-K and J-K, the difference
P-K-(H-K)-(J-K) and unpredicted C1-P. Include closure<=1e-10, component norms and
signed projections. These are actual finite intervention effects; their signed
sum is not an independent learned-feature account or causal fraction. The RMS
direct/scale tangent identity is ordinary normalization geometry.

## Numerical and storage gate

Before freeze, complete the new producer AND separate audit on the existing
32-case unequal-token tinyCPU reference from282c. No real checkpoint/cases.
The synthetic test must exercise all14arms, archive/zero/full/install controls,
new derivatives and paired accounting; scientific primary success is not required.
Only then freeze. Any failed check stops attribution and is preserved.

Independently qualify the defined RMS and product JVP using TorchDoubleAD and
central differences h=1e-4 and5e-5 along the ACTUAL attention direction. Each
coordinate error bound1e-4. Independently validate64False anchors/192SiLU
coefficient vectors with DoubleAD/twoFD as before. Public finiteC primitives,
installed products/down projections and full-vocabulary heads must replay EXACT;
CPU64 head<=.005. CPU64C norm/gate/SiLU and polynomial checks retain1e-4. New raw
up/product/down bridges keep original1e-4 comparisons. Up is declared diagnostic
with282b-style dimension-derived rounding envelope; product/down raw diagnostics
also retained. This neither passes older failed up/whole-block/logit checks nor
independently proves the GPU reduction. There is no whole-block Jacobian claim.

All original references/data/seals are immutable. Lossless identical-array aliases
may reference resolved282logical arrays or earlier unique new arrays; verify
dtype/shape/content hashes and all281d dependencies. No deletion/recompression.
Initial640MiB/percase512MiB free reserve, rawcap64MiB; keep bytecode off. Any
producer/source/qualification correction must be separately named and frozen.
No fresh, efficient, learned-circuit, priority or general Question B claim.
Fresh support would require new contexts and globally sealed forecasts and
numerical qualification before full-current calibration/nativeTrue measurements.
