"""Sampled native-shape primitive VJPs, independently checked by saved-array algebra."""

import numpy as np
import torch

from .core225b import array


def probes(engine, key):
    data, zeros = {}, []
    layers, width = len(engine.tapes[0]) - 1, engine.model.config.hidden_size
    rng = np.random.default_rng(int(key[:16], 16))
    eta = float(np.float32(1 / np.sqrt(width)))
    data["norm_eta"] = np.asarray(eta)
    weights = []
    for j in range(layers):
        weights.append(array(engine.model.model.layers[j + 1].input_layernorm.weight))
    data["norm_weights"] = np.stack(weights)
    for anchor in (0, 1):
        saved = {
            n: []
            for n in (
                "norm_x",
                "norm_out",
                "norm_AD",
                "q",
                "k",
                "q_AD",
                "k_AD",
                "p",
                "v",
                "p_AD",
                "v_AD",
                "softmax_AD",
                "softmax_eta",
            )
        }
        heads, coords, scales = [], [], []
        for j in range(layers):
            layer = engine.model.model.layers[j + 1]
            a = engine.attention[anchor][j]
            head = int(rng.integers(a["q"].shape[1]))
            coordinate = int(rng.integers(a["v"].shape[-1]))
            heads.append(head)
            coords.append(coordinate)
            scales.append(a["scaling"])
            with torch.enable_grad():
                x = engine.tapes[anchor][j].clone().requires_grad_(True)
                norm = layer.input_layernorm(x)
                loss = norm[0, 0].sum() * eta
                grad = torch.autograd.grad(loss, x)[0]
                saved["norm_x"].append(array(x[0, 0]))
                saved["norm_out"].append(array(norm[0, 0]))
                saved["norm_AD"].append(array(grad[0, 0]))
                engine.counts["primitive_norm_VJPs"] += 1
                q, k = (a[n].clone().requires_grad_(True) for n in ("q", "k"))
                raw = q @ k.transpose(2, 3) * a["scaling"]
                qg, kg = torch.autograd.grad(raw[0, head, -1, engine.start], (q, k))
                saved["q"].append(array(q[0, head, -1]))
                saved["k"].append(array(k[0, head, engine.start]))
                saved["q_AD"].append(array(qg[0, head, -1]))
                saved["k_AD"].append(array(kg[0, head, engine.start]))
                qg[0, head, -1] = 0
                kg[0, head, engine.start] = 0
                zeros.append(
                    dict(
                        anchor=anchor,
                        layer=j + 2,
                        primitive="qk",
                        error=max(float(qg.abs().max().cpu()), float(kg.abs().max().cpu())),
                    )
                )
                p, v = (a[n].clone().requires_grad_(True) for n in ("p", "v"))
                product = p @ v
                pg, vg = torch.autograd.grad(product[0, head, -1, coordinate], (p, v))
                saved["p"].append(array(p[0, head, -1]))
                saved["v"].append(array(v[0, head, :, coordinate]))
                saved["p_AD"].append(array(pg[0, head, -1]))
                saved["v_AD"].append(array(vg[0, head, :, coordinate]))
                pg[0, head, -1] = 0
                vg[0, head, :, coordinate] = 0
                zeros.append(
                    dict(
                        anchor=anchor,
                        layer=j + 2,
                        primitive="pv",
                        error=max(float(pg.abs().max().cpu()), float(vg.abs().max().cpu())),
                    )
                )
                score = a["masked"].clone().requires_grad_(True)
                prob = torch.nn.functional.softmax(score, dim=-1, dtype=torch.float32).to(
                    score.dtype
                )
                soft_eta = rng.normal(size=score.shape[-1]).astype(np.float32)
                soft_eta /= np.linalg.norm(soft_eta)
                tensor_eta = torch.tensor(soft_eta, device=score.device, dtype=score.dtype)
                sg = torch.autograd.grad((prob[0, head, -1] * tensor_eta).sum(), score)[0]
                saved["softmax_AD"].append(array(sg[0, head, -1]))
                saved["softmax_eta"].append(soft_eta)
                sg[0, head, -1] = 0
                zeros.append(
                    dict(
                        anchor=anchor,
                        layer=j + 2,
                        primitive="softmax",
                        error=float(sg.abs().max().cpu()),
                    )
                )
                engine.counts["primitive_attention_VJPs"] += 3
            del x, norm, grad, q, k, raw, qg, kg, p, v, product, pg, vg, score, prob, sg
        data.update({f"probe{anchor}_" + n: np.stack(values) for n, values in saved.items()})
        data[f"probe{anchor}_heads"] = np.asarray(heads)
        data[f"probe{anchor}_coordinates"] = np.asarray(coords)
        data[f"probe{anchor}_scales"] = np.asarray(scales)
    return data, zeros


def primitive_records(data, eps):
    """Float64 CPU formulas, using saved native parents and library adjoints."""
    rows = []
    eta, weights = float(data["norm_eta"]), data["norm_weights"].astype(float)
    for anchor in (0, 1):
        p = {
            n: data[f"probe{anchor}_" + n].astype(float)
            for n in (
                "norm_x",
                "norm_out",
                "norm_AD",
                "q",
                "k",
                "q_AD",
                "k_AD",
                "p",
                "v",
                "p_AD",
                "v_AD",
                "softmax_AD",
                "softmax_eta",
                "scales",
            )
        }
        for j in range(len(weights)):
            x, w = p["norm_x"][j], weights[j]
            scale = np.sqrt(np.mean(x * x) + eps)
            expected_norm = w * x / scale
            expected_grad = w * eta / scale - x * np.dot(w * eta, x) / (len(x) * scale**3)
            prob, se = p["p"][j], p["softmax_eta"][j]
            for name, actual, expected in (
                ("norm_primal", p["norm_out"][j], expected_norm),
                ("norm_VJP", p["norm_AD"][j], expected_grad),
                ("q_VJP", p["q_AD"][j], p["scales"][j] * p["k"][j]),
                ("k_VJP", p["k_AD"][j], p["scales"][j] * p["q"][j]),
                ("probability_VJP", p["p_AD"][j], p["v"][j]),
                ("value_VJP", p["v_AD"][j], prob),
                ("softmax_VJP", p["softmax_AD"][j], prob * (se - np.dot(prob, se))),
            ):
                err = float(np.max(abs(actual - expected)))
                rows.append(
                    dict(
                        anchor=anchor,
                        layer=j + 2,
                        primitive=name,
                        error=err,
                        bound=1e-4,
                        passes=bool(np.isfinite(err) and err <= 1e-4),
                    )
                )
    return rows
