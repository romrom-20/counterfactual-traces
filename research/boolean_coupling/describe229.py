"""Explicitly post-observation position-resolved descriptions; zero model calls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

NAMES = ("R", "C", "QK", "SM", "AVs", "AVd")


def main(run):
    assert not (run/"position-effects.csv").exists() and not (run/"description.json").exists()
    status = json.loads((run/"status.json").read_text())
    rows, global_rows = [], []
    for entry in status["cases"]:
        meta = json.loads((run/"cases"/(entry["key"]+".json")).read_text())
        data = dict(np.load(run/"cases"/(entry["key"]+".npz")))
        error = data["E"]
        values = {k:data[k] if k in ("R", "C") else data["smooth_vector_"+k] for k in NAMES}
        for position in range(len(error)):
            e = error[position]
            row = dict(key=entry["key"], **meta["group"], receiving_offset=position,
                token_position=meta["start"]+position, verdict_position=position == len(error)-1,
                error_norm=float(np.linalg.norm(e)))
            for name, value in values.items():
                row[name+"_norm"] = float(np.linalg.norm(value[position]))
                row[name+"_signed_share"] = float(np.vdot(value[position], e)/np.vdot(e, e)) if np.vdot(e, e) else None
            rows.append(row)
        global_rows.append(dict(key=entry["key"], error_norm=float(np.linalg.norm(error)),
            **{k+"_signed_share":float(np.vdot(v, error)/np.vdot(error, error)) for k, v in values.items()},
            **{k+"_norm":float(np.linalg.norm(v)) for k, v in values.items()}))
    with (run/"position-effects.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run/"outcomes.csv").open() as f:
        outcomes = list(csv.DictReader(f))
    lookup = {(r["key"], r["arm"]):r for r in outcomes}
    ratios = {name:{metric:float(np.median([float(lookup[e["key"], name][metric])
               /float(lookup[e["key"], "base"][metric]) for e in status["cases"]]))
               for metric in ("vector_error", "margin_error")} for name in ("repair_R", "repair_C")}
    description = dict(explicitly_post_observation=True, model_calls=0, cases=global_rows,
        all_receiving_medians={key:float(np.median([r[key] for r in global_rows])) for key in global_rows[0] if key != "key"},
        ratio_of_errors_medians=ratios,
        base_margin_minmax=[float(fn([float(r["margin_error"]) for r in outcomes if r["arm"] == "base"])) for fn in (min, max)],
        note="Global shares and error ratios are descriptive; frozen eligibility/criteria unchanged. "
             "A signed projection share is not a variance-explained or causal percentage.")
    (run/"description.json").write_text(json.dumps(description, indent=2)+"\n")
    print(json.dumps({k:v for k, v in description.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    main(parser.parse_args().run.resolve())
