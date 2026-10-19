"""Explicitly posthoc signed-margin and local-vector summaries of partially qualified237b."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(run):
    assert json.loads((run/"complete-numerical-diagnostic.json").read_text())["status"] == "complete_with_failed_checks"
    assert not (run/"margin-factorial.csv").exists()
    rows = []
    for entry in json.loads((run/"status.json").read_text())["cases"]:
        key = entry["key"]
        meta = json.loads((run/"cases"/(key+".json")).read_text())
        with np.load(run/"cases"/(key+".npz")) as data:
            for regime in ("forecast", "oracle"):
                values = {b:float(np.diff(data[regime+"_"+b+"_scores"].astype(float))[0]) for b in ("1001", "1101", "1011", "1111")}
                q, s = values["1101"]-values["1001"], values["1011"]-values["1001"]
                interaction = values["1111"]-values["1101"]-values["1011"]+values["1001"]
                target = values["1111"]-values["1001"]
                assert abs(q+s+interaction-target) <= 1e-9
                rows.append(dict(key=key, **meta["group"], regime=regime, QK=q, softmax=s, interaction=interaction,
                    full_minus_base=target, opposite_QK_softmax=q*s < 0, posthoc=True))
    with (run/"margin-factorial.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run/"local-effects.csv").open() as file:
        local = list(csv.DictReader(file))
    summary = dict(posthoc=True, new_model_calls=0, margin={regime:dict(
        opposite_QK_softmax=sum(r["opposite_QK_softmax"] for r in rows if r["regime"] == regime),
        medians={n:float(np.median([r[n] for r in rows if r["regime"] == regime])) for n in ("QK", "softmax", "interaction", "full_minus_base")})
        for regime in ("forecast", "oracle")}, local={regime:{str(layer):{component:{scope:{name:float(np.median(
            [float(r[name]) for r in local if r["regime"] == regime and int(r["layer"]) == layer
             and r["component"] == component and r["scope"] == scope])) for name in ("norm", "signed_share", "remaining_relative")}
             for scope in ("all", "verdict")} for component in ("QK", "S", "interaction")}
             for layer in range(21, 25)} for regime in ("forecast", "oracle")})
    (run/"posthoc-description.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["margin"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
