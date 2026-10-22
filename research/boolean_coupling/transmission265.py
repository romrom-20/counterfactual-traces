"""Operation-defined same-input MLP corners; no whole-block Jacobian."""

import hashlib

import numpy as np
import torch

from .input263 import plans
from .silu258 import predict

CORNERS = ("L", "P", "D", "H")


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def full(value, length, start):
    result = torch.zeros((1, length, value.shape[-1]), dtype=torch.float32, device="mps")
    result[:, start+1:] = torch.tensor(value[None].astype(np.float32), device="mps")
    return result


def project(value, weight, length, start):
    with torch.no_grad():
        return torch.nn.functional.linear(full(value, length, start), weight)[0, start+1:].cpu().numpy()


def corners(base, point, gamma, eps, matrices, columns, key, layer, length, start):
    parts = plans(base["x"].astype(float), point.astype(float), base["z"].astype(float), gamma.astype(float), eps)
    linear, p, d = (parts[k] for k in ("linear", "base", "direction"))
    inputs = dict(L=linear, P=linear+p, D=linear+d, H=linear+p+d)
    nodes = {}
    for name, value in inputs.items():
        z = value.astype(np.float32)
        gate, up = (project(z, matrices[branch], length, start) for branch in ("gate", "up"))
        with torch.no_grad():
            phi = torch.nn.functional.silu(full(gate, length, start))[0, start+1:].cpu().numpy()
        quad, mask, _, _, _, _ = predict(base["gate"].astype(float), base["phi"].astype(float), gate.astype(float), up.astype(float), columns, key, layer)
        nodes[name] = dict(z=z, gate=gate, up=up, phi=phi, quadratic=quad.astype(np.float32), mask=mask)
    common = nodes["H"]["mask"]
    for value in nodes.values():
        for mode, mask in (("adaptive", value["mask"]), ("fixed", common)):
            phi = np.where(mask, value["phi"], value["quadratic"]).astype(np.float32)
            value["used_"+mode] = phi
            value["product_"+mode] = phi*value["up"]
    return nodes


def additive(nodes, mode):
    return (nodes["P"]["product_"+mode].astype(float)+nodes["D"]["product_"+mode].astype(float)-nodes["L"]["product_"+mode].astype(float)).astype(np.float32)


def components(nodes):
    phi = {k:v["used_fixed"].astype(float) for k, v in nodes.items()}
    up = {k:v["up"].astype(float) for k, v in nodes.items()}
    product = {k:v["product_fixed"].astype(float) for k, v in nodes.items()}
    ideal = {k:phi[k]*up[k] for k in CORNERS}
    terms = dict(gate_side=(phi["H"]-phi["P"]-phi["D"]+phi["L"])*up["H"],
        gate_up_cross=(phi["P"]-phi["L"])*(up["D"]-up["L"])+(phi["D"]-phi["L"])*(up["P"]-up["L"]),
        up_nonadditivity=(phi["P"]+phi["D"]-phi["L"])*(up["H"]-up["P"]-up["D"]+up["L"]))
    ideal_mixed = ideal["H"]-ideal["P"]-ideal["D"]+ideal["L"]
    public_product = product["H"]-product["P"]-product["D"]+product["L"]
    terms["product_rounding"] = public_product-ideal_mixed
    return terms, ideal_mixed, public_product


def random_write(change, key, layer):
    result = []
    for row, value in enumerate(change.astype(float)):
        seed = hashlib.sha256(f"{key}:{layer}:{row}:local-mixing265".encode()).hexdigest()
        random = np.random.default_rng(int(seed[:16], 16)).normal(size=value.shape)
        result.append(random/np.linalg.norm(random)*np.linalg.norm(value))
    return np.array(result, dtype=np.float32)
