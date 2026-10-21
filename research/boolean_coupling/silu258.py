"""False-anchored SiLU quadratic and target-blind finite-coordinate budget."""

import hashlib

import numpy as np


def sigmoid(value):
    return np.exp(-np.logaddexp(0., -value))


def predict(base, anchor, point, up, columns, key, layer, random=False):
    sigma = sigmoid(base)
    first = sigma*(1+base*(1-sigma))
    second = sigma*(1-sigma)*(2+base*(1-2*sigma))
    delta = point-base
    quadratic = anchor+first*delta+.5*second*delta*delta
    closest = np.clip(0., np.minimum(base, point), np.maximum(base, point))
    variance = sigmoid(closest)*sigmoid(-closest)
    bound = variance*(3+np.maximum(abs(base), abs(point)))*abs(delta)**3/6
    score = bound*abs(up)*columns
    count = max(1, int(.01*score.size))
    rank = np.zeros(score.size, dtype=bool)
    rank[np.argsort(score.ravel(), kind="stable")[-count:]] = True
    rank = rank.reshape(score.shape)
    mask = rank.copy()
    if random:
        mask[:] = False
        for row in range(mask.shape[0]):
            seed = int(hashlib.sha256(f"{key}:{layer}:{row}:receiver-silu-control".encode()).hexdigest()[:16], 16)
            picked = np.random.default_rng(seed).choice(mask.shape[1], int(rank[row].sum()), replace=False)
            mask[row, picked] = True
    return quadratic, mask, rank, bound, first, second
