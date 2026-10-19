"""Explicitly post-observation recombination of sealed forecast components; no model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    assert not (run/"hybrid-description.json").exists(), "Never overwrite evidence."
    status = json.loads((run/"status.json").read_text())
    rows, inputs = [], {}
    for entry in status["observations"]:
        paths = [run/k/(entry["key"]+".npz") for k in ("forecasts", "observations")]
        for path in paths:
            record = json.loads(path.with_suffix(".json").read_text())
            assert sha(path) == record["arrays_sha256"]
            inputs[str(path.relative_to(run))] = sha(path)
        f, o = (dict(np.load(p)) for p in paths)
        meta = json.loads(paths[0].with_suffix(".json").read_text())
        false_linear = f["source_mid"].astype(float)+f["F_tangent"]
        nf = f["FNV_prediction"]-false_linear
        ns = f["SNV_prediction"]-f["SL_prediction"]
        dt = f["SL_prediction"]-false_linear
        dn = ns-nf
        error = o["native_mid"].astype(float)-f["FNV_prediction"]
        identity = float(np.max(abs(dt+dn-(f["SNV_prediction"]-f["FNV_prediction"]))))
        assert identity <= 1e-9
        predictions = dict(F_linear=false_linear, source_linear_false_correction=f["SL_prediction"]+nf,
                           false_linear_source_correction=false_linear+ns,
                           original_FNV=f["FNV_prediction"], original_SNV=f["SNV_prediction"])
        target_delta = o["native_mid"].astype(float)-f["false_mid"].astype(float)
        row = dict(key=entry["key"], group=meta["group"], identity_error=identity)
        for scope in ("all", "verdict"):
            target = target_delta if scope == "all" else target_delta[-1]
            e = error if scope == "all" else error[-1]
            denominator = float(np.linalg.norm(target))
            for name, prediction in predictions.items():
                delta = prediction-f["false_mid"].astype(float)
                delta = delta if scope == "all" else delta[-1]
                relative = float(np.linalg.norm(delta-target)/denominator)
                cosine = float(np.vdot(delta, target)/(np.linalg.norm(delta)*denominator))
                row[scope+"_"+name+"_relative"] = relative
                row[scope+"_"+name+"_cosine"] = cosine
                row[scope+"_"+name+"_meets_old_accuracy"] = bool(denominator >= .001 and relative <= .025 and cosine >= .995)
            for name, value in (("tangent_change", dt), ("correction_change", dn)):
                v = value if scope == "all" else value[-1]
                row[scope+"_"+name+"_share_of_FNV_error"] = float(np.vdot(v, e)/np.vdot(e, e))
                row[scope+"_"+name+"_norm"] = float(np.linalg.norm(v))
        rows.append(row)
    numeric = [k for k in rows[0] if k not in ("key", "group")]
    decision = [k for k in numeric if k.endswith("_meets_old_accuracy")]
    result = dict(explicitly_post_observation=True, model_calls=0, source_sha256=sha(Path(__file__)), inputs=inputs,
        note="Hybrids recombine sealed predictor terms after observation; these are not registered "
             "232 forecast arms or interventions on a native learned circuit. They do not establish necessity or uniqueness.",
        counts={k:sum(r[k] for r in rows) for k in decision},
        medians={k:float(np.median([r[k] for r in rows])) for k in numeric}, cases=rows)
    (run/"hybrid-description.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k, v in result.items() if k not in ("cases", "inputs")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    main(parser.parse_args().run.resolve())
