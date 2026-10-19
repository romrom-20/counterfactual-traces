"""False-state MLP curvature formulas: homogeneous quadratic and finite norm models."""

import numpy as np
import torch
from torch.nn import functional as F


def derivatives(g):
    s = 1/(1+np.exp(-g))
    return g*s, s*(1+g*(1-s)), s*(1-s)*(2+g*(1-2*s))


def anchor(w, x):
    rms = np.sqrt(np.mean(x*x, -1, keepdims=True)+w["eps"])
    z = x/rms*w["norm"]
    g, u = (z@w[n+"_weight"].T+w.get(n+"_bias", 0) for n in ("gate", "up"))
    phi, slope, curve = derivatives(g)
    return dict(rms=rms, z=z, g=g, u=u, phi=phi, slope=slope, curve=curve)


def formula(w, x, d, a):
    q0 = a["rms"]**2
    q1, q2 = 2*np.mean(x*d, -1, keepdims=True), np.mean(d*d, -1, keepdims=True)
    s0 = 1/a["rms"]
    s1, s2 = -.5*q1/q0**1.5, .375*q1*q1/q0**2.5-.5*q2/q0**1.5
    z1 = w["norm"]*(d*s0+x*s1)
    z2 = w["norm"]*(d*s1+x*s2)
    g1, u1 = (z1@w[n+"_weight"].T for n in ("gate", "up"))
    g2, u2 = (z2@w[n+"_weight"].T for n in ("gate", "up"))
    exact_z = (x+d)/np.sqrt(np.mean((x+d)**2, -1, keepdims=True)+w["eps"])*w["norm"]
    dz = exact_z-a["z"]
    dg, du = (dz@w[n+"_weight"].T for n in ("gate", "up"))
    products = dict(quadratic_norm=a["slope"]*g2*a["u"]+a["phi"]*u2,
        quadratic_silu=.5*a["curve"]*g1*g1*a["u"], quadratic_product=a["slope"]*g1*u1,
        finite_norm=a["slope"]*(dg-g1)*a["u"]+a["phi"]*(du-u1),
        finite_silu=.5*a["curve"]*dg*dg*a["u"],
        finite_product=(a["slope"]*dg+.5*a["curve"]*dg*dg)*du)
    return {k:v@w["down_weight"].T for k, v in products.items()}


def torch_function(w):
    p = {k:torch.from_numpy(np.asarray(v)) for k, v in w.items()}

    def linear(x, name):
        return F.linear(x, p[name+"_weight"], p.get(name+"_bias"))

    def function(x):
        z = x*torch.rsqrt(x.square().mean(-1, keepdim=True)+p["eps"])*p["norm"]
        return x+linear(F.silu(linear(z, "gate"))*linear(z, "up"), "down")
    return function
