"""Unequal-shape fixtures, clipping crossings and coherent negative controls."""

import argparse
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit295 import check, qualify
from .cache272 import save_json
from .drive295 import ROOT, SOURCES, evaluate, kernel
from .joint292 import proposal
from .source279 import derivatives


def main(out):
    assert not out.exists()
    rng = np.random.default_rng(295)
    checks = []
    rejected = []
    for hidden, inter in [(17, 23), (19, 29)]:
        gf = rng.normal(0, 4, (2, inter))
        ge = gf + rng.normal(0, 2, (2, inter))
        gf[0, 0], ge[0, 0] = -30, 30
        f = dict(g=gf, u=rng.normal(size=(2, inter)), phi=gf / (1 + np.exp(-gf)), **derivatives(gf))
        u = f["u"] + rng.normal(size=(2, inter))
        phi, slope = proposal("P", ge, f)
        p = dict(
            g=ge,
            u=u,
            phi=phi.astype(np.float32),
            slope=slope,
            False_g=gf,
            False_u=f["u"],
            False_phi=f["phi"],
            False_slope=f["d1"],
        )
        wd = rng.normal(0, inter**-0.5, (hidden, inter))
        old = rng.normal(size=(2, hidden))
        dg, du = ge - gf, u - f["u"]
        smooth = (f["u"] * (phi - f["phi"] - f["d1"] * dg) + (phi - f["phi"]) * du) @ wd.T
        p.update(
            nonlinear=smooth,
            write=smooth,
            False_write=np.zeros_like(smooth),
            linear=np.zeros_like(smooth),
        )
        data = evaluate(wd, p, old)
        rows = qualify(wd, p, data, old)
        assert all(r["passes"] for r in rows), [r for r in rows if not r["passes"]]

        def product(t):
            ph, _ = proposal("P", gf + t * dg, f)
            return ph * (f["u"] + t * du)

        for t in [0.25, 0.5, 0.75]:
            analytic = kernel(t, f, dg, du) + f["u"] * f["d1"] * dg + f["phi"] * du
            for step in [1e-4, 5e-5]:
                check(
                    rows,
                    "FD_" + str(t) + "_" + str(step),
                    (product(t + step) - product(t - step)) / (2 * step),
                    analytic,
                    1e-5,
                    1e-5,
                )
        assert all(r["passes"] for r in rows), [r for r in rows if not r["passes"]]
        for label in ["sign", "drive"]:
            bad = {k: v.copy() for k, v in data.items()}
            if label == "sign":
                bad["ENDPOINT_DRIVE"] *= -1
                bad["endpoint_product"] *= -1
            else:
                bad["gate_change"] = np.roll(bad["gate_change"], 1, axis=-1)
            assert any(not r["passes"] for r in qualify(wd, p, bad, old))
            rejected.append(f"{hidden}_{label}")
        zero = dict(p, g=gf.copy(), u=f["u"].copy(), phi=f["phi"].astype(np.float32), slope=f["d1"])
        z = evaluate(wd, zero, np.zeros_like(old))
        for name in [
            "ENDPOINT_DRIVE",
            "MIDPOINT_DRIVE",
            "SIMPSON_DRIVE",
            "INTEGRAL16",
            "SHUFFLED_DRIVE",
        ]:
            assert np.array_equal(z[name], np.zeros_like(old)), name
        checks.extend(rows)
    out.mkdir()
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
            zero_drive=True,
            source_hashes={n: sha(ROOT / n) for n in SOURCES},
            scientific_outcomes_not_required=True,
        ),
    )
    print("295 CPU qualification PASS", len(checks), rejected, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args().out.resolve())
