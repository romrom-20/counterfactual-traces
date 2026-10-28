"""Unequal-shape explicit dense-operator fixtures and sign/range corruption tests."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .audit229 import sha
from .audit294 import qualify
from .cache272 import save_json
from .change294 import ROOT, SOURCES, apply, basis, evaluate, orth, statistics


def main(out):
    assert not out.exists()
    out.mkdir()
    torch.set_num_threads(2)
    checks = []
    rejected = []
    for hidden, width in ((17, 23), (19, 29)):
        rng = np.random.default_rng(hidden)
        w = {
            n: rng.normal(0, 0.2, shape)
            for n, shape in [
                ("gate_weight", (width, hidden)),
                ("up_weight", (width, hidden)),
                ("down_weight", (hidden, width)),
            ]
        }
        af, bf, ae, be = rng.normal(size=(4, 2, width))
        o = rng.normal(size=(2, hidden // 2, hidden))
        n = rng.normal(size=(2, hidden))
        qf, _ = basis(w, af, bf, o)
        qe, _ = basis(w, ae, be, o)
        parent = dict(
            False_a=af,
            False_b=bf,
            current_a=ae,
            current_b=be,
            current_omega=o,
            False_omega=o.copy(),
            False_Q=qf,
            current_Q=qe,
            nonlinear=n,
        )
        data, rows = evaluate(w, parent)
        rows += qualify(w, parent, data)
        assert all(r["passes"] for r in rows)
        for row in range(2):

            def dense(a, b):
                return w["down_weight"] @ (
                    a[:, None] * w["gate_weight"] + b[:, None] * w["up_weight"]
                )

            dm = dense(ae[row], be[row]) - dense(af[row], bf[row])
            assert (
                np.max(
                    abs(
                        apply(
                            w, (ae - af)[row : row + 1], (be - bf)[row : row + 1], o[row : row + 1]
                        )[0]
                        - o[row] @ dm.T
                    )
                )
                < 1e-10
            )
            q = orth((o[row] @ dm.T)[None])
            for _ in range(2):
                q = orth(orth(q @ dm) @ dm.T)
            actual = data["CHANGE_RANGE_Q"][row]
            assert np.linalg.norm(actual - (actual @ q[0].T) @ q[0]) < 1e-10
            s = statistics(data["CHANGE_RANGE_projected"][row], n[row])
            assert abs(s["signed_target_fraction"] - (1 - s["relative"] ** 2)) < 1e-10
        for tag in ("sign", "range"):
            bad = {k: v.copy() for k, v in data.items()}
            if tag == "sign":
                bad["change_y0"] *= -1
            else:
                bad["CHANGE_RANGE_Q"] = bad["CHANGE_RANGE_Q"][..., ::-1].copy()
            assert any(not r["passes"] for r in qualify(w, parent, bad)), tag
            rejected.append(f"{hidden}_{tag}")
        checks.extend(rows)
    save_json(
        out / "qualification.json",
        dict(
            passed=True,
            synthetic=True,
            checkpoint_read=False,
            model_forwards=0,
            shapes=[[17, 23], [19, 29]],
            required_checks=len(checks),
            failed_checks=0,
            rejected=rejected,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
            scientific_outcomes_not_required=True,
        ),
    )
    print("294 CPU qualification PASS", len(checks), rejected, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
