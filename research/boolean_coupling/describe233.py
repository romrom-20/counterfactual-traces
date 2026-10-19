"""Explicitly post-observation additive-versus-mixed comparison for the 233 operator."""

import argparse
import json
from pathlib import Path

import numpy as np

from .audit229 import sha


def main(run):
    assert not (run/"additive-description.json").exists()
    status = json.loads((run/"status.json").read_text())
    rows = []
    for entry in status["cases"]:
        path = run/"cases"/(entry["key"]+".npz")
        assert sha(path) == entry["arrays_sha256"]
        data = {k:v.astype(float) for k, v in dict(np.load(path)).items()}
        meta = json.loads(path.with_suffix(".json").read_text())
        row = dict(key=entry["key"], **meta["group"])
        joint = data["baseline"]+data["T_1010"]-data["T_0000"]
        additive = joint-data["M_1010"]
        for scope in ("all", "verdict"):
            select = slice(None) if scope == "all" else -1
            target = (data["native_mid"]-data["false_mid"])[select]
            for name, value in (("joint", joint), ("additive", additive)):
                delta = (value-data["false_mid"])[select]
                relative = float(np.linalg.norm(delta-target)/np.linalg.norm(target))
                cosine = float(np.vdot(delta, target)/(np.linalg.norm(delta)*np.linalg.norm(target)))
                row[scope+"_"+name+"_relative"] = relative
                row[scope+"_"+name+"_accuracy"] = bool(np.linalg.norm(target) >= .001 and relative <= .025 and cosine >= .995)
        rows.append(row)
    numeric = [k for k in rows[0] if k not in ("key", "block", "schema", "reverse", "assignment", "target")]
    output = dict(explicitly_post_observation=True, model_calls=0, source_sha256=sha(__file__),
        note="Not a registered 233 arm; the additive-removal hypothesis is tested prospectively only in 234.",
        counts={k:sum(r[k] for r in rows) for k in numeric if k.endswith("accuracy")},
        medians={k:float(np.median([r[k] for r in rows])) for k in numeric}, cases=rows)
    (run/"additive-description.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k:v for k, v in output.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
