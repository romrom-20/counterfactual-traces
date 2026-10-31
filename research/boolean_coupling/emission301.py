"""POSTHOC description of native emitted Boolean differences; no new model calls."""

import argparse
from pathlib import Path

import numpy as np

from .audit229 import sha
from .audit272 import read, table
from .publish285 import write
from .subspace293 import statistics

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/alias-source-routes-301/native-phase"


def main(run, dest):
    d = read(run / "design.json")
    repeated = d.get("numerical_repetition", False)
    assert len(d["cases"]) == 32, "This profile is for the checkpoint cohort, not tiny I/O fixtures"
    if repeated:
        assert read(run / "summary.json")["passed"] and not d["tiny_fixture"]
        assert (
            d["frozen_source_hashes"]
            == read(ROOT / "results/alias-source-routes-301/pre-native/design.json")[
                "source_hashes"
            ]
        )
        entries = []
        for c in d["cases"]:
            m = read(run / (c["key"] + ".json"))
            files = {e["path"]: e["sha256"] for e in m["files"]}
            entries.append(
                (
                    dict(arrays_sha256=files[c["key"] + "-forecast.npz"]),
                    dict(arrays_sha256=files[c["key"] + "-native.npz"]),
                )
            )
    else:
        fs = read(run / "forecast-status.json")
        ns = read(run / "native-status.json")
        entries = list(zip(fs["cases"], ns["cases"], strict=True))
    rows, records, residuals = [], [], []
    for case, (fe, ne) in zip(d["cases"], entries, strict=True):
        fp = (
            run / (case["key"] + "-forecast.npz")
            if repeated
            else run / "forecasts" / (case["key"] + ".npz")
        )
        np_ = (
            run / (case["key"] + "-native.npz")
            if repeated
            else run / "observations" / (case["key"] + ".npz")
        )
        assert sha(fp) == fe["arrays_sha256"] and sha(np_) == ne["arrays_sha256"]
        with np.load(fp) as z:
            f = dict(z)
        with np.load(np_) as z:
            n = dict(z)
        assert len(f["A_F_trace"]) == 28
        for j in range(28):
            da = n[f"A_T_emit{j}"].astype(float) - f[f"A_F_emit{j}"].astype(float)
            db = n[f"B_T_emit{j}"].astype(float) - f[f"B_F_emit{j}"].astype(float)
            if j == 0:
                assert np.array_equal(da, db), (
                    "Equal initial Boolean embeddings must emit equal first-layer value differences"
                )
            s = statistics(db, da)
            records.append(
                dict(
                    key=case["key"],
                    layout=case["group"]["block"],
                    mention_order=case["group"]["mention_order"],
                    layer=j + 1,
                    first_layer_exact_equal=bool(j == 0 and np.array_equal(da, db)),
                    **s,
                )
            )
            ha = n["A_T_trace"][j, 0].astype(float) - f["A_F_trace"][j, 0].astype(float)
            hb = n["B_T_trace"][j, 0].astype(float) - f["B_F_trace"][j, 0].astype(float)
            residuals.append(
                dict(
                    key=case["key"],
                    layout=case["group"]["block"],
                    mention_order=case["group"]["mention_order"],
                    layer=j + 1,
                    **statistics(hb, ha),
                )
            )
        rows.append(
            dict(key=case["key"], forecast_arrays_sha256=sha(fp), native_arrays_sha256=sha(np_))
        )
    write(
        dest / "emission-records.json",
        dict(
            experiment=301,
            posthoc=True,
            model_forwards=0,
            numerical_repetition=repeated,
            first_layer_exact_pairs=32,
            source_script_sha256=sha(Path(__file__)),
            inputs=rows,
            statistics=records,
            residual_statistics=residuals,
            boundary="Saved native Boolean value differences. Descriptive input-spelling dependence, not a learned feature, necessary circuit, new forecast or rescue of failed hypotheses.",
        ),
    )
    table(
        dest / "emission-profile.csv",
        [
            dict(
                layer=j,
                median_cosine=float(np.median([r["cosine"] for r in records if r["layer"] == j])),
                median_relative=float(
                    np.median([r["relative"] for r in records if r["layer"] == j])
                ),
                median_norm_ratio=float(
                    np.median(
                        [
                            np.sqrt(
                                r["prediction_squared_norm"] / max(r["target_squared_norm"], 1e-30)
                            )
                            for r in records
                            if r["layer"] == j
                        ]
                    )
                ),
                residual_median_relative=float(
                    np.median([r["relative"] for r in residuals if r["layer"] == j])
                ),
                residual_median_cosine=float(
                    np.median([r["cosine"] for r in residuals if r["layer"] == j])
                ),
            )
            for j in range(1, 29)
        ],
    )
    print(dict(posthoc=True, records=len(records), first_layer_exact_pairs=32, model_forwards=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEST)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main(args.run.resolve(), args.out.resolve())
