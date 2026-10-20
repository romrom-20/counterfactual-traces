"""Explicitly posthoc position/contribution sizes; no model or tokenizer calls."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audit229 import sha
from .core225b import write_json


def main(run,regional):
    assert json.loads((run/"audit.json").read_text())["status"]=="passed"
    assert not (run/"position-summary.json").exists()
    rows=[]
    for ent in json.loads((run/"status.json").read_text())["cases"]:
        key=ent["key"]
        p,q=run/"cases"/(key+".npz"),regional/"cases"/(key+".npz")
        with np.load(p) as d,np.load(q) as r:
            interaction=r["FF_trajectory"][-1,1:].astype(float)-r["FQ_trajectory"][-1,1:]-r["QF_trajectory"][-1,1:]+r["QQ_trajectory"][-1,1:]
            delta=d["D_final"][1:].astype(float)-r["FF_trajectory"][-1,1:]
            rows.append(dict(key=key,first_receiver_D_energy=float(np.vdot(d["D"][0],d["D"][0])/np.vdot(d["D"],d["D"])),
                terminal_D_change_relative_norm=float(np.linalg.norm(delta)/np.linalg.norm(interaction)),
                terminal_D_change_signed_share=float(np.vdot(delta,interaction)/np.vdot(interaction,interaction)),
                input246_sha256=sha(p),input244_sha256=sha(q)))
    with (run/"position-effects.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run/"position-summary.json",dict(posthoc=True,model_calls=0,source_sha256=sha(__file__),
        audit_sha256=sha(run/"audit.json"),csv_sha256=sha(run/"position-effects.csv"),
        metrics={k:dict(median=float(np.median([r[k] for r in rows])),minimum=min(r[k] for r in rows),maximum=max(r[k] for r in rows))
            for k in ("first_receiver_D_energy","terminal_D_change_relative_norm","terminal_D_change_signed_share")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--input-244",type=Path,required=True)
    args=parser.parse_args()
    main(args.run.resolve(),args.input_244.resolve())
