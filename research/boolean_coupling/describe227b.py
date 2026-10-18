"""Explicitly posthoc algebraic check of the remaining gate/up product term."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    target = out / "posthoc-product-algebra.json"
    assert not target.exists()
    rows, hashes = [], {}
    for p in sorted((out / "cases").glob("*.npz")):
        q = dict(np.load(p))
        meta = json.loads(p.with_suffix(".json").read_text())
        assert sha(p)==meta["arrays_sha256"]
        hashes[p.name] = sha(p)

        def a(name, field):
            return q[name+"_"+field].astype(float)

        phisum = a("b","phi")+a("d","phi")-a("zero","phi")
        usum = a("b","up")+a("d","up")-a("zero","up")
        ideal = ((a("b","phi")-a("zero","phi"))*(a("d","up")-a("zero","up"))
                 +(a("d","phi")-a("zero","phi"))*(a("b","up")-a("zero","up")))
        error = q["RP"]-ideal
        arithmetic = ((a("Y","product")-a("Y","phi")*a("Y","up"))
                      +(a("Y","phi")-phisum)*a("Y","up")
                      +phisum*(a("Y","up")-usum))
        for name, sign in (("b",-1),("d",-1),("zero",1)):
            arithmetic += sign*(a(name,"product")-a(name,"phi")*a(name,"up"))
        closure = float(np.max(abs(error-arithmetic)))
        assert closure<=1e-9
        rows.append(dict(key=meta["key"],schema=meta["group"]["schema"],
            raw_ideal_error=float(np.max(abs(error))), raw_ideal_pass=bool(np.max(abs(error))<=1e-4),
            relative_L2_arithmetic=float(np.linalg.norm(error)/np.linalg.norm(q["RP"])),
            arithmetic_closure=closure))
    result = dict(explicitly_posthoc=True, model_calls=0, derivative_claim=False,
        source_sha256=sha(__file__), input_arrays=hashes, cases=rows,
        summary=dict(n=len(rows),raw_failures=sum(not r["raw_ideal_pass"] for r in rows),
            maximum_raw_error=max(r["raw_ideal_error"] for r in rows),
            maximum_arithmetic_closure=max(r["arithmetic_closure"] for r in rows),
            median_relative_L2_arithmetic=float(np.median([r["relative_L2_arithmetic"] for r in rows]))))
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out.resolve())
