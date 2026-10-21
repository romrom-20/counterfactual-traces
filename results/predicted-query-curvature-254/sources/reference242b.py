"""Independent CPU64 AD/FD qualification of the uniform primitive interventions."""

import hashlib
from collections import Counter

import numpy as np
import torch

from .audit229 import checked


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


class Reference:
    def __init__(self, bank):
        self.bank, self.checks, self.witnesses = bank, [], []
        self.cache, self.functions, self.counts = {}, {}, Counter()
        for label, entry in bank.items():
            x = torch.from_numpy(entry["x"].astype(float))
            if entry["kind"] == "norm":
                weight = torch.from_numpy(entry["weight"].astype(float))
                eps = float(entry["eps"])
                def fn(v, w=weight, e=eps):
                    return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + e) * w
            elif entry["kind"] == "silu":
                def fn(v):
                    return v * torch.sigmoid(v)
            else:
                checked(self.checks, "original_False_softmax_primal", torch.softmax(x, dim=-1).numpy(), entry["y"], 1e-4)
                y = torch.from_numpy(entry["y"].astype(float))
                mass = y.sum(-1, keepdim=True)
                log_y = torch.where(y > 0, torch.log(y), torch.full_like(y, -torch.inf))
                x = torch.zeros_like(x)
                def fn(v, m=mass, logp=log_y):
                    return m * torch.softmax(logp + v, dim=-1)
            self.functions[label] = (x, fn)
            checked(self.checks, "False_primitive_primal", fn(x).numpy(), entry["y"], 1e-4)
            self.counts["False_primitive_primals"] += 1

    def installed(self, label, delta, proposed, analytic, mode="taylor"):
        self.counts["installation_calls"] += 1
        key = label + ":" + mode + ":" + digest(delta)
        if key in self.cache:
            expected = self.cache[key]
            self.counts["qualification_cache_hits"] += 1
        else:
            entry = self.bank[label]
            scale = max(1., float(np.max(abs(delta))))
            witness = dict(label=label, mode=mode, actual_direction_sha256=digest(delta), scale=scale,
                qualified_direction="actual direction divided by max(1,infinity norm)")
            if not np.count_nonzero(delta):
                expected = entry["y"].astype(np.float32)
                witness["zero_direction"] = True
                self.counts["zero_directions"] += 1
            else:
                x, fn = self.functions[label]
                v = torch.from_numpy(delta / scale)
                primal, first = torch.func.jvp(fn, (x,), (v,))
                a1, a2 = analytic(delta / scale)
                checked(self.checks, "CPU64_first_action", first.numpy(), a1, 1e-9)
                plus, minus = (fn(x + sign * 1e-5 * v).numpy() for sign in (1, -1))
                checked(self.checks, "CPU64_first_FD", (plus - minus) / 2e-5, first.numpy(), 1e-4)
                witness.update(first_plus_sha256=digest(plus), first_minus_sha256=digest(minus))
                self.counts["AD_first"] += 1
                self.counts["FD_endpoints"] += 2
                second = torch.zeros_like(first)
                if entry["kind"] != "norm":
                    def first_fn(z):
                        return torch.func.jvp(fn, (z,), (v,))[1]
                    _, second = torch.func.jvp(first_fn, (x,), (v,))
                    checked(self.checks, "CPU64_second_action", second.numpy(), a2, 1e-9)
                    plus2, minus2 = (fn(x + sign * 3e-4 * v).numpy() for sign in (1, -1))
                    checked(self.checks, "CPU64_second_FD", (plus2 - 2 * primal.numpy() + minus2) / (3e-4)**2, second.numpy(), 1e-4)
                    witness.update(second_plus_sha256=digest(plus2), second_minus_sha256=digest(minus2))
                    self.counts["AD_second"] += 1
                    self.counts["FD_endpoints"] += 2
                if entry["kind"] == "softmax" and mode == "tilt":
                    base = torch.from_numpy(entry["y"].astype(float))
                    p = base / base.sum(-1, keepdim=True)
                    d = torch.from_numpy(delta)
                    u = d - (p * d).sum(-1, keepdim=True)
                    w = 1 + u + .5 * u.square()
                    expected = (base * w / (p * w).sum(-1, keepdim=True)).numpy().astype(np.float32)
                else:
                    expected = (entry["y"].astype(float) + scale * first.numpy() + .5 * scale**2 * second.numpy()).astype(np.float32)
                witness["zero_direction"] = False
            self.cache[key] = expected
            self.witnesses.append(witness)
            self.counts["distinct_directions"] += 1
        checked(self.checks, "installed_primitive_map", expected, proposed, 1e-4)
