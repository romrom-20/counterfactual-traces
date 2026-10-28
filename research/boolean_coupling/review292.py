"""Posthoc signed vector records from sealed arrays; no model calls or new forecasts."""

import argparse
from pathlib import Path

import numpy as np

from .publish285 import read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/joint-gate-292"


def stats(prediction, target):
    p, t = np.asarray(prediction, float).ravel(), np.asarray(target, float).ravel()
    pp, tt, pt, ee = float(p @ p), float(t @ t), float(p @ t), float((p - t) @ (p - t))
    return dict(
        prediction_squared_norm=pp,
        target_squared_norm=tt,
        prediction_target_dot=pt,
        error_squared_norm=ee,
        norm=tt**0.5,
        relative=ee**0.5 / max(tt**0.5, 1e-30),
        cosine=pt / max((pp * tt) ** 0.5, 1e-30),
        signed_target_fraction=pt / max(tt, 1e-30),
    )


def main(run):
    d = read(run / "design.json")
    assert read(run / "native-audit.json")["status"] == "passed"
    forecasts, natives = (read(run / (n + "-status.json")) for n in ("forecast", "native"))
    final, layers = [], []
    for c, fe, te in zip(d["cases"], forecasts["cases"], natives["cases"], strict=True):
        key = c["key"]
        paths = [run / phase / (key + ".npz") for phase in ("forecasts", "native")]
        assert all(sha(p) == e["arrays_sha256"] for p, e in zip(paths, (fe, te), strict=True))
        with np.load(paths[0]) as f, np.load(paths[1]) as t:
            base = f["F_final"].astype(float)
            native = t["final"].astype(float)
            for arm in ("P", "D", "C", "B", "A", "T"):
                pred = native if arm == "T" else f[arm + "_final"].astype(float)
                for contrast, anchor in [("native", base), ("effects", f["A_final"].astype(float))]:
                    for scope, rows in [("receiving", slice(1, None)), ("verdict", -1)]:
                        final.append(
                            dict(
                                key=key,
                                arm=arm,
                                contrast=contrast,
                                scope=scope,
                                **stats((pred - anchor)[rows], (native - anchor)[rows]),
                            )
                        )
            scopes = [
                ("source_states", "states", 0),
                ("receiving_states", "states", slice(1, None)),
                ("source_verdict_MLP_writes", "writes", slice(None)),
            ]
            for scope, field, rows in scopes:
                p = f["P_" + field].astype(float) - f["F_" + field].astype(float)
                target = t[field].astype(float) - f["F_" + field].astype(float)
                for j in range(len(p)):
                    layers.append(
                        dict(
                            key=key, scope=scope, layer=j + 1, **stats(p[j, rows], target[j, rows])
                        )
                    )
    write(
        DEST / "posthoc-vector-review.json",
        dict(
            experiment=292,
            posthoc=True,
            model_calls=0,
            script_sha256=sha(Path(__file__)),
            native_audit_sha256=sha(run / "native-audit.json"),
            forecast_archives={e["key"]: e["arrays_sha256"] for e in forecasts["cases"]},
            native_archives={e["key"]: e["arrays_sha256"] for e in natives["cases"]},
            final=final,
            layers=layers,
            boundary="Signed Float64 accounting of already sealed predictions. Neither a new forecast nor primitive causal localization.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    main(p.parse_args().run.resolve())
