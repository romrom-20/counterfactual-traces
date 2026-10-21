# 255 — can the query correction be reduced to local scalar geometry?

This is a retrospective, model-free diagnostic on all 32 already observed 254
contexts and all 28 layers. It does not supply a fresh native-response forecast.

For False state x and displacement d, let s²=mean(x²)+eps,
a=mean(x*d)/s² and b=mean(d²)/s². The exact smooth RMSNorm remainder is

`gamma/s * [(r-1+a)*x + (r-1)*d]`, where `r=(1+2*a+b)^(-1/2)`.

This is an algebraic identity, not a discovered learned feature. Through second
order in d, the remainder is

`Q2 = gamma/s * [(1.5*a²-.5*b)*x-a*d]`.

The third-order addition is

`Q3 = gamma/s * [(1.5*a*b-2.5*a³)*x+(1.5*a²-.5*b)*d]`.

Evaluate the identity and Q2/Q2+Q3 errors at D0/P1/P2/native inputs. Retain all
case/layer/row-scope outcomes; relative local accuracy is eligible only above
1e-8 remainder norm. Do not select sites or fit coefficients. Compare the
closed coefficients to a separate generic polynomial-jet recurrence (1e-10)
and the second-order coefficient to centered finite differences at step 1e-3
(1e-5) at four fixed strata cases, blocks 1/14/28, all four states. Check the
exact identity within 1e-10 everywhere. Failure stops, with evidence retained.

Local Taylor accuracy motivates, but cannot replace, a globally sealed fresh
forecast. A subsequent test must compute these terms at its own evolving
states without a preceding corrected trajectory or native True measurements.
Retaining finite keys, values, MLPs and source computations remains a rich
component budget even if the extra correction trajectories can be removed.
