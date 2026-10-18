"""Post-observation identity check locating the first forecast discrepancy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run):
    target=run/"first-attention-boundary.json"
    assert not target.exists()
    cases=json.loads((Path(__file__).parent/"cases228.json").read_text())
    rows=[]
    inputs={}
    for case in cases:
        fp=run/"forecasts"/(case["key"]+".npz")
        op=run/"observations"/(case["key"]+".npz")
        for p in (fp,op):
            meta=json.loads(p.with_suffix(".json").read_text())
            assert sha(p)==meta["arrays_sha256"]
            inputs[str(p.relative_to(run))]=sha(p)
        f,o=dict(np.load(fp)),dict(np.load(op))
        native1=o["true_states"][0].astype(float)-f["false_boundaries"][1].astype(float)
        e1=float(np.max(abs(f["L_delta_out"][0]-native1)))
        for arm in ("F","R"):
            assert np.max(abs(f[arm+"_delta_out"][0]-f["L_delta_out"][0]))==0
        actual=o["true_mids"][1].astype(float)-f["false_mids"][1].astype(float)
        pred=f["L_pred_mid"][1]-f["false_mids"][1].astype(float)
        error=pred-actual
        rows.append(dict(key=case["key"],**case["group"],first_block_coordinate_error=e1,
            attention2_error_norm=float(np.linalg.norm(error[-1])),
            attention2_native_norm=float(np.linalg.norm(actual[-1])),
            attention2_relative=float(np.linalg.norm(error[-1])/np.linalg.norm(actual[-1])),
            attention2_max_coordinate_error=float(np.max(abs(error[-1])))))
    result=dict(explicitly_post_observation=True,model_calls=0,source_sha256=sha(__file__),inputs=inputs,cases=rows,
        summary=dict(n=len(rows),first_block_maximum_coordinate_error=max(r["first_block_coordinate_error"] for r in rows),
            **{name:float(np.median([r[name] for r in rows])) for name in
               ("attention2_error_norm","attention2_native_norm","attention2_relative","attention2_max_coordinate_error")}))
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    main(parser.parse_args().run.resolve())
